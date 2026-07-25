"""eval_engine.py — the PANTRY-REFILL engine, kept alive for OFFLINE eval only.

Lifted verbatim from commit ff78ece (`nanochat/fast_engine.py` lines 334-1139),
which is the last commit before `6603715 RL: retire the streaming + handoff
engines` removed it. Restored deliberately: eval is the one regime it is still
the right machine for.

WHY IT IS A SEPARATE MODULE RATHER THAN A RESTORE INTO fast_engine.py
--------------------------------------------------------------------
Eval is many prefixes x few rows; RL-train is few prefixes x K rows. Pantry's
value is mid-round ADOPTION — a retiring row's slot is refilled from a pantry of
waiting jobs, so the batch stays full across a long tail of uneven completions.
That is exactly what the current fast_engine CANNOT do: its SoA decode scheduler
(7276dbb) is built on "batch positions are STABLE between bucket drops", which is
what lets a quiet window cost one pinned D2H and no block-table re-upload.
Adoption mutates the batch mid-window and breaks that invariant, so Pantry could
not be ported onto the new GraphDecoder without giving up the property the
training engine was rewritten to get. It therefore keeps its OWN era's KVPool
(with the lend/reclaim protocol), GraphDecoder (begin_window/collect_window),
Node, Seq and PrefillGraph — frozen at ff78ece.

The genuinely shared, byte-identical low-level layer is IMPORTED from
fast_engine, not copied: PAGE, VmmBuffer, nblocks, decode_body, prefill_body.
That also keeps the torch custom-op registrations (fa_kvcache_paged, kv_scatter)
happening exactly ONCE, so both modules can be imported in the same process.
Everything this module needs from the model side (nanochat/gpt.py,
flash_attention.py, common.py, tasks/gsm8k.py) is unchanged since ff78ece —
verified with `git diff ff78ece HEAD --` on each — so the lift needs no porting.

NOT for training: it is never coresident with a training step, and nothing in
the RL loop imports it. Offline checkpoint scoring only.
"""

import math
import time
from collections import deque

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from nanochat.fast_engine import PAGE, VmmBuffer, nblocks, decode_body, prefill_body

class KVPool:
    """All decode-time KV memory as two big tensors — REFCOUNTED free-list pool.
    Block 0 = null block. K/V live on VMM reservations: stable VA for the
    captured graphs, physical lent back to training between rounds."""

    def __init__(self, config, pool_bytes: int, device_index: int = 0):
        n_layer = config.n_layer
        n_kv = config.n_kv_head
        head_dim = config.n_embd // config.n_head
        bpb = 2 * n_layer * PAGE * n_kv * head_dim * 2  # bytes per block (K+V, bf16)
        self.num_blocks = max(2, pool_bytes // bpb)
        shape = (n_layer, self.num_blocks, PAGE, n_kv, head_dim)
        self.k_buf = VmmBuffer(self.num_blocks * (bpb // 2), dev=device_index)
        self.v_buf = VmmBuffer(self.num_blocks * (bpb // 2), dev=device_index)
        self.k = self.k_buf.tensor(shape)
        self.v = self.v_buf.tensor(shape)
        self.free = list(range(self.num_blocks - 1, 0, -1))
        self.refs = [0] * self.num_blocks

    def lend(self) -> float:
        """After a round's gen: release the pool's PHYSICAL pages to the driver
        so training can use them. Contents destroyed (fine — re-prefilled next
        round); VA + captured graph pointers stay valid. Returns seconds."""
        t = time.perf_counter()
        torch.cuda.synchronize()
        self.k_buf.unmap()
        self.v_buf.unmap()
        return time.perf_counter() - t

    def reclaim(self) -> float:
        """Before a round's gen: re-acquire physical. empty_cache() first — torch
        hoards train's freed segments as cache, and cuMemCreate draws from the
        same device free pool. Returns seconds."""
        t = time.perf_counter()
        torch.cuda.empty_cache()
        self.k_buf.map()
        self.v_buf.map()
        return time.perf_counter() - t

    def alloc(self, n: int) -> list[int]:
        if n <= 0:
            return []
        if n > len(self.free):
            raise RuntimeError(f"KV pool exhausted ({n} wanted, {len(self.free)} free)")
        out = [self.free.pop() for _ in range(n)]
        for b in out:
            assert self.refs[b] == 0
            self.refs[b] = 1
        return out

    def addref(self, block_ids: list[int]) -> None:
        for b in block_ids:
            assert self.refs[b] > 0, f"addref on free block {b}"
            self.refs[b] += 1

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            self.refs[b] -= 1
            assert self.refs[b] >= 0, f"double free of block {b}"
            if self.refs[b] == 0:
                self.free.append(b)


class SmearStore:
    """Per-row persistent smear state (prev-token post-norm/pre-smear embedding),
    indexed by slot id. Rows hold a slot from mint to retire; the decode window
    gathers slots -> dense graph buffer before replays and scatters back after."""

    def __init__(self, n_slots: int, n_embd: int, device):
        self.data = torch.zeros(n_slots, n_embd, dtype=torch.bfloat16, device=device)
        self.free = list(range(n_slots - 1, -1, -1))

    def alloc(self) -> int:
        assert self.free, "SmearStore exhausted — raise MAX_SEQS/PANTRY_BLOCKS slack"
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(slot)


# -----------------------------------------------------------------------------
# §5. Node / Seq — prefix-shared admission units (speedrun verbatim + smear seed)
# -----------------------------------------------------------------------------
class Node:
    """One problem's shared context for a round: ``prompt[:-1]``. The K sibling
    rows alias its prefix pages; each row's forced first decode input is
    ``prompt[-1]`` (KV lands at plen in the row's private page), so the first
    SAMPLED token already comes out of the decode graph — pure natural decode."""
    __slots__ = ("fed_ids", "plen", "cand_jobs", "blocks", "seq_len",
                 "n_full", "partial", "rows_left", "pantry_pages", "seed_emb")

    def __init__(self, context_ids: list[int], cand_jobs: list[dict]):
        self.fed_ids = context_ids
        self.plen = len(context_ids)
        self.cand_jobs = cand_jobs
        self.blocks: list[int] = []
        self.seq_len = 0
        self.n_full = self.plen // PAGE
        self.partial = (self.plen % PAGE) != 0
        self.rows_left = len(cand_jobs)
        self.pantry_pages = 0
        self.seed_emb: torch.Tensor | None = None  # (C,) bf16, set after prefill

    def new_pages(self) -> int:
        return nblocks(self.plen) + (max(len(self.cand_jobs) - 1, 0) if self.partial else 0)


class Seq:
    """One rollout row. Context KV is SHARED (aliased prefix pages); ``forced``
    (= prompt[-1]) is the first decode input; ``gen`` collects sampled tokens."""
    __slots__ = ("job", "node", "forced", "plen", "allow", "blocks", "n_shared",
                 "seq_len", "gen", "next_tok", "need", "done", "priv_pantry", "slot")

    def __init__(self, job: dict, node: "Node", macro_n: int):
        self.job = job
        self.node = node
        self.forced = job["forced"]
        self.plen = node.plen
        self.allow = job["allow"]
        self.blocks: list[int] = []
        self.n_shared = 0
        self.seq_len = 0
        self.gen: list[int] = []
        self.next_tok = -1
        self.need = nblocks(self.plen + 1 + self.allow + macro_n)
        self.done = False
        self.priv_pantry = 0
        self.slot = -1


# -----------------------------------------------------------------------------
# §6. In-graph sampler (house nucleus draw: top-k -> top-p -> Gumbel-max).
#     Temperature / top-p live in 0-D CUDA buffers so eval can retune them
#     WITHOUT re-capturing the decode graphs.
# -----------------------------------------------------------------------------
def sample(logits: torch.Tensor, inv_temp: torch.Tensor, top_p: torch.Tensor,
           top_k: int) -> torch.Tensor:
    vals, idx = logits.topk(top_k, dim=1)          # logits already fp32 (softcapped)
    vals = vals * inv_temp
    probs = torch.softmax(vals, dim=-1)
    vals = vals.masked_fill((probs.cumsum(-1) - probs) > top_p, float("-inf"))
    probs = torch.softmax(vals, dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs).exponential_()
    return idx.gather(1, probs.div(q).argmax(dim=-1, keepdim=True)).squeeze(1)


# -----------------------------------------------------------------------------
# §7. Captured graphs: varlen prefill + bucketed paged decode
# -----------------------------------------------------------------------------
class PrefillGraph:
    """ONE captured CUDA graph for every refill. Packs node contexts, scatters KV
    into the paged pool, and emits each context's last-position pre-smear
    embedding (static output ``seed_out``); the caller clones seeds per node
    right after each replay."""

    def __init__(self, model, pool: KVPool, prefill_t: int, prefill_seqs: int,
                 compile_body: bool = True, fullgraph: bool = True):
        self.model, self.pool = model, pool
        cfg = model.config
        self.prefill_t, self.prefill_seqs = prefill_t, prefill_seqs
        n_kv = cfg.n_kv_head
        head_dim = cfg.n_embd // cfg.n_head
        self.k_flat = pool.k.view(cfg.n_layer, -1, n_kv, head_dim)
        self.v_flat = pool.v.view(cfg.n_layer, -1, n_kv, head_dim)
        dev = "cuda"
        self.ids = torch.zeros(prefill_t, dtype=torch.long, device=dev)
        self.pos = torch.zeros(prefill_t, dtype=torch.long, device=dev)
        self.slot = torch.zeros(prefill_t, dtype=torch.long, device=dev)
        self.notstart = torch.zeros(prefill_t, 1, dtype=torch.bfloat16, device=dev)
        self.cu = torch.zeros(prefill_seqs + 2, dtype=torch.int32, device=dev)
        self.gather = torch.zeros(prefill_seqs, dtype=torch.long, device=dev)
        self._scratch = pool.alloc(1)[0]
        self.body = (torch.compile(prefill_body, dynamic=False, fullgraph=fullgraph)
                     if compile_body else prefill_body)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.seed_out: torch.Tensor | None = None
        self.replays = 0
        self.real_tok = 0

    def _forward(self):
        return self.body(self.model, self.ids, self.pos, self.cu, self.slot,
                         self.notstart, self.gather, self.k_flat, self.v_flat,
                         self.prefill_t)

    def capture(self) -> None:
        self.ids[:] = 0
        self.pos[:] = 0
        self.slot[:] = self._scratch * PAGE
        self.notstart[:] = 0
        self.cu[:] = self.prefill_t
        self.cu[0] = 0
        self.gather[:] = 0
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.seed_out = self._forward()
        self.graph = g

    def run(self, items: list[Node]) -> None:
        i = 0
        while i < len(items):
            chunk, tot = [], 0
            while i < len(items) and len(chunk) < self.prefill_seqs:
                Lp = len(items[i].fed_ids)
                assert Lp <= self.prefill_t, f"context of {Lp} tokens exceeds PREFILL_T={self.prefill_t}"
                if chunk and tot + Lp > self.prefill_t:
                    break
                chunk.append(items[i])
                tot += Lp
                i += 1
            self._replay(chunk, tot)

    def _replay(self, chunk: list[Node], tot: int) -> None:
        ids, pos, slot, cu, gather, nstart = [], [], [], [0], [], []
        for s in chunk:
            Lp = len(s.fed_ids)
            s.blocks = self.pool.alloc(nblocks(Lp))
            s.seq_len = Lp
            ids.extend(s.fed_ids)
            pos.extend(range(Lp))
            slot.extend(s.blocks[p // PAGE] * PAGE + p % PAGE for p in range(Lp))
            nstart.extend([0.0] + [1.0] * (Lp - 1))
            cu.append(cu[-1] + Lp)
            gather.append(cu[-1] - 1)
        n = len(chunk)
        dev = "cuda"
        self.ids[:tot] = torch.tensor(ids, dtype=torch.long, device=dev)
        self.ids[tot:] = 0
        self.pos[:tot] = torch.tensor(pos, dtype=torch.long, device=dev)
        self.pos[tot:] = 0
        self.slot[:tot] = torch.tensor(slot, dtype=torch.long, device=dev)
        self.slot[tot:] = self._scratch * PAGE
        self.notstart[:tot, 0] = torch.tensor(nstart, dtype=torch.bfloat16, device=dev)
        self.notstart[tot:] = 0
        cu = cu + [self.prefill_t] * (self.prefill_seqs + 2 - len(cu))
        self.cu.copy_(torch.tensor(cu, dtype=torch.int32, device=dev))
        self.gather[:n] = torch.tensor(gather, dtype=torch.long, device=dev)
        self.gather[n:] = 0
        self.graph.replay()
        # Clone each node's smear seed NOW — seed_out is overwritten next replay.
        for j, nd in enumerate(chunk):
            nd.seed_emb = self.seed_out[j].clone()
        self.replays += 1
        self.real_tok += tot


class GraphDecoder:
    """One captured CUDA graph per row-count bucket over shared static buffers.
    Each replay = ONE decode step; the driver replays MACRO_N times per window,
    recording sampled tokens device-side between replays."""

    def __init__(self, model, pool: KVPool, store: SmearStore, max_blocks: int,
                 max_seqs: int, macro_n: int, buckets: tuple, temperature: float,
                 top_p: float, top_k: int, compile_body: bool = True,
                 extra_compile_slots: int = 8):
        self.model, self.pool, self.store = model, pool, store
        self.max_blocks, self.max_seqs, self.macro_n = max_blocks, max_seqs, macro_n
        self.buckets, self.top_k = tuple(sorted(buckets)), top_k
        if compile_body:
            need = len(self.buckets) + extra_compile_slots
            for attr in ("recompile_limit", "cache_size_limit", "accumulated_cache_size_limit"):
                if hasattr(torch._dynamo.config, attr):
                    setattr(torch._dynamo.config, attr,
                            max(getattr(torch._dynamo.config, attr), need))
        self.body = torch.compile(decode_body, dynamic=False) if compile_body else decode_body
        dev = "cuda"
        C = model.config.n_embd
        self.input_ids = torch.zeros(max_seqs, 1, dtype=torch.long, device=dev)
        self.cache_seqlens = torch.zeros(max_seqs, dtype=torch.int32, device=dev)
        self.block_table = torch.zeros(max_seqs, max_blocks, dtype=torch.int32, device=dev)
        self.prev_emb = torch.zeros(max_seqs, C, dtype=torch.bfloat16, device=dev)
        self.tok_buf = torch.zeros(max_seqs, dtype=torch.long, device=dev)
        self.token_record = torch.zeros(max_seqs, macro_n, dtype=torch.long, device=dev)
        self.inv_temp = torch.tensor(1.0 / temperature, dtype=torch.float32, device=dev)
        self.top_p_t = torch.tensor(top_p, dtype=torch.float32, device=dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mempool = None
        self._scratch = pool.alloc(1)[0]
        self._win_rows = 0

    def set_sampling(self, temperature: float | None = None, top_p: float | None = None) -> None:
        """Live sampler knobs — the captured graphs read these buffers."""
        if temperature is not None:
            assert temperature > 0, "captured sampler is stochastic; temp must be > 0"
            self.inv_temp.fill_(1.0 / temperature)
        if top_p is not None:
            self.top_p_t.fill_(top_p)

    def _macro_body(self, b: int) -> None:
        logits, x_pre = self.body(self.model, self.input_ids[:b], self.cache_seqlens[:b],
                                  self.block_table[:b], self.pool.k, self.pool.v,
                                  self.prev_emb[:b])
        tok = sample(logits, self.inv_temp, self.top_p_t, self.top_k)
        self.tok_buf[:b] = tok
        self.input_ids[:b, 0] = tok
        self.prev_emb[:b].copy_(x_pre)
        self.cache_seqlens[:b] += 1

    def capture_all(self) -> None:
        for b in sorted(self.buckets, reverse=True):
            t = time.perf_counter()
            self.input_ids[:] = 0
            self.block_table[:] = self._scratch
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.cache_seqlens[:] = 0
                    self._macro_body(b)
            torch.cuda.current_stream().wait_stream(s)
            self.cache_seqlens[:] = 0
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, **({"pool": self._mempool} if self._mempool else {})):
                self._macro_body(b)
            self._mempool = self._mempool or g.pool()
            self.graphs[b] = g
            print(f"    bucket {b:3d}: {time.perf_counter() - t:5.1f}s", flush=True)

    def begin_window(self, inputs: list[int], cache_lens: list[int],
                     block_rows: list[list[int]], slots: list[int]) -> None:
        B = len(inputs)
        self._win_rows = B
        bucket = next(x for x in self.buckets if x >= B)
        mb = self.max_blocks
        dev = "cuda"
        slots_t = torch.tensor(slots, dtype=torch.long, device=dev)
        self.input_ids[:B, 0] = torch.tensor(inputs, dtype=torch.long, device=dev)
        self.cache_seqlens[:B] = torch.tensor(cache_lens, dtype=torch.int32, device=dev)
        self.block_table[:B] = torch.tensor([r + [0] * (mb - len(r)) for r in block_rows],
                                            dtype=torch.int32, device=dev)
        self.prev_emb[:B] = self.store.data[slots_t]
        if bucket > B:
            self.cache_seqlens[B:bucket] = 0
            self.block_table[B:bucket] = 0
        g = self.graphs[bucket]
        for i in range(self.macro_n):
            g.replay()
            self.token_record[:B, i] = self.tok_buf[:B]
        # Persist the window's final smear state back to the rows' slots
        # (device-ordered; safe to enqueue before the host reads tokens).
        self.store.data[slots_t] = self.prev_emb[:B]

    def collect_window(self) -> list[list[int]]:
        return self.token_record[:self._win_rows].tolist()


# -----------------------------------------------------------------------------
# §8. FastEngine — the fused generation driver (speedrun §6 run_gen)
# -----------------------------------------------------------------------------
def default_buckets(ceiling_rows: int, max_seqs: int, step: int = 16) -> tuple:
    roundup = lambda n: min(max_seqs, -(-n // step) * step)
    top = roundup(max(ceiling_rows, 1))
    coarse = {b for b in (64, 128) if b < top}
    return tuple(sorted(coarse | {top}))


class PantryRefillEngine:
    """Paged prefix-shared CUDA-graph engine over a live nanochat GPT.

    Life cycle: __init__ builds the pool + graphs (untimed setup; call with the
    KV pool memory available), then per round:
        pool.reclaim() -> run_round(nodes) -> pool.lend()
    `make_nodes` builds the admission units from (meta, prompt_ids, k, allow)
    specs. Rows never retain pool pages across rounds (leak-asserted).
    """

    def __init__(self, model, tokenizer, *,
                 kv_pool_gb: float,
                 max_seqs: int,
                 max_tokens: int,
                 max_prompt_len: int,
                 macro_n: int = 8,
                 buckets: tuple | None = None,
                 prefill_t: int = 2048,
                 prefill_seqs: int = 12,
                 pantry_blocks: int = 96,
                 starve_jobs: int = 12,
                 pass1: int = 0,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 top_k: int = 512,
                 stop_detect: bool = True,
                 stop_strings: tuple = ("\nQuestion:", " Question:", "\nProblem:"),
                 window_tokens: int = 6,
                 compile_decode: bool = True,
                 compile_prefill: bool = True,
                 prefill_fullgraph: bool = True,
                 extra_compile_slots: int = 8,
                 print_every: int | None = None,
                 device_index: int = 0):
        assert pass1 == 0 or pass1 < max_tokens, "PASS1 must be < MAX_TOKENS (or 0 to disable)"
        self.model, self.tok = model, tokenizer
        self.max_tokens, self.pass1 = max_tokens, pass1
        self.pass1_eff = pass1 or max_tokens
        self.macro_n, self.prefill_t, self.prefill_seqs = macro_n, prefill_t, prefill_seqs
        self.pantry_cap, self.starve_jobs = pantry_blocks, starve_jobs
        self.stop_detect, self.stop_strings, self.window_tokens = stop_detect, tuple(stop_strings), window_tokens
        self.print_every = print_every

        # Terminal set for the nanochat chat format: assistant_end ends the turn;
        # a sampled bos would start a fresh document — both retire the row.
        self.terminal_ids = frozenset({tokenizer.encode_special("<|assistant_end|>"),
                                       tokenizer.get_bos_token_id()})
        self.gate_ids = frozenset(
            ids[-1] for s in self.stop_strings for ids in [tokenizer.encode(s)] if ids)

        self.kv_pool_gb = kv_pool_gb
        self.sampler_cfg = (temperature, top_p, top_k)
        self.pool = KVPool(model.config, int(kv_pool_gb * 2 ** 30), device_index)
        self.max_blocks = nblocks(max_prompt_len + 1 + max_tokens + macro_n) + 1
        # Admission ceiling: worst-case per-row private reserve at the pass-1 budget.
        min_priv = nblocks(2 + self.pass1_eff + macro_n)  # smallest plausible context
        self.ceiling_rows = max(1, (len(self.pool.free) - pantry_blocks) // min_priv)
        if buckets is None:
            buckets = default_buckets(min(self.ceiling_rows, max_seqs), max_seqs)
        assert max(buckets) <= max_seqs
        self.buckets = tuple(sorted(buckets))
        self.max_seqs = max_seqs

        n_slots = max_seqs + 4 * pantry_blocks + 64
        self.store = SmearStore(n_slots, model.config.n_embd, "cuda")
        with torch.no_grad():
            self.gd = GraphDecoder(model, self.pool, self.store, self.max_blocks,
                                   max_seqs, macro_n, self.buckets, temperature,
                                   top_p, top_k, compile_body=compile_decode,
                                   extra_compile_slots=extra_compile_slots)
        self.pfg = None
        self._compile_prefill = compile_prefill
        self._prefill_fullgraph = prefill_fullgraph
        # v1.1 lesson (speedrun): overlapped side-stream refills deadlock
        # mid-decode in a fused gen+train process; refills stay serialized.
        self.prefill_stream = torch.cuda.current_stream()

    @torch.no_grad()
    def capture(self, warm_context: list[int]) -> None:
        """Capture decode buckets + the prefill graph; warm one prefill replay."""
        temp, top_p, top_k = self.sampler_cfg
        print(f"  engine config: kv_pool {self.kv_pool_gb:g} GB "
              f"({self.pool.num_blocks} blocks x {PAGE} tok) | max_seqs {self.max_seqs} | "
              f"buckets {self.buckets} | macro_n {self.macro_n} | "
              f"prefill T={self.prefill_t} x{self.prefill_seqs} seqs | "
              f"pantry {self.pantry_cap} blocks | starve_jobs {self.starve_jobs} | "
              f"max_tokens {self.max_tokens}"
              + (f" (pass1 {self.pass1})" if self.pass1 else "")
              + f" | temp {temp:g} top_p {top_p:g} top_k {top_k} | "
              f"stop_detect {int(self.stop_detect)} | ceiling_rows {self.ceiling_rows}",
              flush=True)
        print("  capture+compile decode buckets:", flush=True)
        self.gd.capture_all()
        self.pfg = PrefillGraph(self.model, self.pool, self.prefill_t, self.prefill_seqs,
                                compile_body=self._compile_prefill,
                                fullgraph=self._prefill_fullgraph)
        t = time.perf_counter()
        self.pfg.capture()
        print(f"    prefill graph (T={self.prefill_t}, "
              f"{'compiled' if self._compile_prefill else 'capture-only'}): "
              f"{time.perf_counter() - t:5.1f}s", flush=True)
        warm = Node(list(warm_context), [])
        self.pfg.run([warm])
        self.pool.release(warm.blocks)
        warm.blocks = []
        self.pfg.replays = self.pfg.real_tok = 0
        torch.cuda.synchronize()

    def set_sampling(self, temperature: float | None = None, top_p: float | None = None) -> None:
        self.gd.set_sampling(temperature, top_p)

    def make_nodes(self, specs: list[tuple]) -> list["Node"]:
        """specs: (meta, prompt_ids, k, allow). Context = prompt[:-1]; forced
        first decode input = prompt[-1]; K natural draws at the given budget."""
        nodes = []
        for meta, prompt_ids, k, allow in specs:
            assert len(prompt_ids) >= 2, "prompt too short for the split-last-token trick"
            assert allow <= self.max_tokens, "allow exceeds engine MAX_TOKENS (block table width)"
            p1 = self.pass1_eff if self.pass1 else allow
            jobs = [dict(meta=meta, prompt_ids=list(prompt_ids), forced=prompt_ids[-1],
                         allow=min(p1, allow), budget=allow,
                         final=(self.pass1 == 0 or allow <= self.pass1_eff))
                    for _ in range(k)]
            nodes.append(Node(list(prompt_ids[:-1]), jobs))
        return nodes

    # -- the round driver ------------------------------------------------------
    @torch.no_grad()
    def run_round(self, nodes_all: list["Node"], rnd: int = 0,
                  on_retire=None) -> tuple[list[dict], dict]:
        """One round of generation through the persistent engine. Returns
        (rows, stats). Each row: dict(meta, completion_token_ids, completion_text,
        terminal, stop_reason, finish_reason). `on_retire(row)` fires as rows
        finish (e.g. inline grading)."""
        pfg, gd, pool, store = self.pfg, self.gd, self.pool, self.store
        MACRO_N = self.macro_n
        pfg.replays = pfg.real_tok = 0
        pantry: deque[tuple[Seq, torch.cuda.Event]] = deque()
        pantry_blocks = 0
        node_q = deque(nodes_all)
        running: list[Seq] = []
        rows: list[dict] = []
        rolls_done = tok_total = 0
        stop_fires = refills = adopt_stalls = bnd_copies = n_ext = 0
        prefill_s = adopt_s = 0.0
        n_target = sum(len(n.cand_jobs) for n in nodes_all)
        print_every = self.print_every or max(1, n_target // 8)

        def reserved_owned() -> int:
            return sum(s.need - len(s.blocks) for s in running)

        def plan_refill() -> list[Node]:
            take: list[Node] = []
            tok_n = cost_sum = 0
            blocks_left = min(self.pantry_cap - pantry_blocks,
                              len(pool.free) - reserved_owned())
            full = False
            for nd in node_q:
                L = nd.plen
                if tok_n + L > self.prefill_t:
                    full = True
                    break
                if cost_sum + nd.new_pages() > blocks_left or len(take) + 1 > self.prefill_seqs:
                    break
                take.append(nd)
                cost_sum += nd.new_pages()
                tok_n += L
            # Continuation nodes trickle in — batch until the chunk is token-full
            # or the pantry is actually hungry (liveness preserved via STARVE).
            if take and (full or len(pantry) <= self.starve_jobs):
                return take
            if not take and not running and not pantry and node_q:
                nd = node_q[0]
                assert nd.new_pages() <= len(pool.free), \
                    f"first node needs {nd.new_pages()} pages, only {len(pool.free)} free"
                return [nd]
            return []

        def mint_rows(nd: Node) -> list[Seq]:
            nonlocal pantry_blocks, bnd_copies
            seqs = []
            full_pages = nd.blocks[:nd.n_full]
            src_bnd = nd.blocks[nd.n_full] if nd.partial else None
            for j, job in enumerate(nd.cand_jobs):
                s = Seq(job, nd, MACRO_N)
                pool.addref(full_pages)
                s.blocks = list(full_pages)
                s.n_shared = nd.n_full
                if nd.partial:
                    if j == 0:
                        s.blocks.append(src_bnd)
                    else:
                        dst = pool.alloc(1)[0]
                        pool.k[:, dst].copy_(pool.k[:, src_bnd])
                        pool.v[:, dst].copy_(pool.v[:, src_bnd])
                        s.blocks.append(dst)
                        s.priv_pantry = 1
                        bnd_copies += 1
                s.seq_len = nd.plen
                s.slot = store.alloc()
                seqs.append(s)
            # Seed every sibling's smear state with the node's last-context
            # pre-smear embedding (computed by the prefill graph).
            slots_t = torch.tensor([s.slot for s in seqs], dtype=torch.long, device="cuda")
            store.data[slots_t] = nd.seed_emb.unsqueeze(0).expand(len(seqs), -1)
            pool.release(full_pages)
            nd.pantry_pages = nd.new_pages()
            pantry_blocks += nd.pantry_pages
            return seqs

        def refill_varlen() -> None:
            nonlocal refills, prefill_s
            take = plan_refill()
            if not take:
                return
            _t = time.perf_counter()
            for _ in take:
                node_q.popleft()
            refills += 1
            with torch.cuda.stream(self.prefill_stream):
                pfg.run(take)
                minted = [s for nd in take for s in mint_rows(nd)]
            ev = torch.cuda.Event()
            ev.record(self.prefill_stream)
            for s in minted:
                pantry.append((s, ev))
            prefill_s += time.perf_counter() - _t

        def adopt() -> None:
            nonlocal pantry_blocks, adopt_s, adopt_stalls
            _t = time.perf_counter()
            reserved = reserved_owned()
            row_cap = min(self.max_seqs, max(self.buckets))
            while pantry and len(running) + 1 <= row_cap:
                s, ev = pantry[0]
                if not ev.query():
                    adopt_stalls += 1
                    break
                if len(pool.free) < reserved + (s.need - len(s.blocks)):
                    break
                reserved += s.need - len(s.blocks)
                pantry_blocks -= s.priv_pantry
                s.node.rows_left -= 1
                if s.node.rows_left == 0:
                    pantry_blocks -= nblocks(s.node.plen)
                pantry.popleft()
                s.next_tok = s.forced
                running.append(s)
            adopt_s += time.perf_counter() - _t

        def suspend(s: Seq) -> None:
            """Two-pass: pass-1 budget reached with budget remaining — free the
            row's pages and re-queue as a K=1 continuation node. Context =
            prompt ⊕ gen[:-1]; forced first decode input = gen[-1]."""
            nonlocal n_ext
            s.done = True
            pool.release(s.blocks)
            s.blocks = []
            store.release(s.slot)
            j = s.job
            gen_prefix = list(j.get("gen_prefix", [])) + list(s.gen)
            cont = dict(meta=j["meta"], prompt_ids=j["prompt_ids"],
                        gen_prefix=gen_prefix, forced=gen_prefix[-1],
                        allow=j["budget"] - len(gen_prefix), budget=j["budget"],
                        final=True)
            node_q.append(Node(list(j["prompt_ids"]) + gen_prefix[:-1], [cont]))
            n_ext += 1

        def retire(s: Seq, eos: bool, stop: str | None = None) -> None:
            nonlocal rolls_done
            s.done = True
            pool.release(s.blocks)
            s.blocks = []
            store.release(s.slot)
            j = s.job
            full_gen = list(j.get("gen_prefix", [])) + list(s.gen)   # pass-1 ⊕ pass-2
            body = full_gen[:-1] if eos else full_gen  # trailing terminal excluded from text
            full_text = self.tok.decode(body)
            if stop is not None:
                cut = len(full_text)
                for ss in self.stop_strings:
                    i = full_text.find(ss)
                    if i != -1:
                        cut = min(cut, i)
                full_text = full_text[:cut]
                finish_reason, stop_reason, terminal = "stop", stop, "stop_string"
            elif eos:
                finish_reason, stop_reason, terminal = "stop", None, "emitted_eos"
            else:
                finish_reason, stop_reason, terminal = "length", None, "truncated"
            row = dict(meta=j["meta"], round=rnd, completion_token_ids=full_gen,
                       completion_text=full_text, finish_reason=finish_reason,
                       stop_reason=stop_reason, terminal=terminal)
            rows.append(row)
            if on_retire is not None:
                on_retire(row)
            rolls_done += 1

        def _stop_hit(s: Seq) -> str | None:
            ids = s.gen[-self.window_tokens:]
            if len(ids) < self.window_tokens:  # pad from the pre-gen stream
                pre = list(s.job["prompt_ids"]) + list(s.job.get("gen_prefix", []))
                ids = pre[len(ids) - self.window_tokens:] + ids
            tail = self.tok.decode(ids)
            hits = [ss for ss in self.stop_strings if ss in tail]
            if not hits:
                return None
            return max(hits, key=lambda ss: tail.rfind(ss))

        def apply_window(batch: list[Seq], toks: list[list[int]]) -> bool:
            nonlocal tok_total, stop_fires
            any_done = False
            for s, row in zip(batch, toks):
                if (self.terminal_ids.isdisjoint(row)
                        and (not self.stop_detect or self.gate_ids.isdisjoint(row))
                        and len(s.gen) + MACRO_N < s.allow):
                    s.gen.extend(row)
                    tok_total += MACRO_N
                    s.seq_len += MACRO_N
                    s.next_tok = row[-1]
                    continue
                for t in row:
                    s.gen.append(t)
                    tok_total += 1
                    if t in self.terminal_ids:
                        retire(s, True)
                        any_done = True
                        break
                    if len(s.gen) >= s.allow:
                        # pass-1 budget with budget remaining -> extend; else truncated.
                        if s.job["final"]:
                            retire(s, False)
                        else:
                            suspend(s)
                        any_done = True
                        break
                    if self.stop_detect and t in self.gate_ids:
                        hit = _stop_hit(s)
                        if hit is not None:
                            stop_fires += 1
                            retire(s, False, stop=hit)
                            any_done = True
                            break
                if not s.done:
                    s.seq_len += MACRO_N
                    s.next_tok = row[-1]
            return any_done

        t0 = time.perf_counter()
        last_roll_print = 0
        while node_q or pantry or running:
            if not running:
                if node_q:
                    refill_varlen()
                if pantry:
                    adopt()
                assert running or pantry or node_q, "nothing running and nothing to admit"
                if not running:
                    continue
            batch = running
            inputs, cache_lens, block_rows, slots = [], [], [], []
            for s in batch:
                grow = (s.seq_len + MACRO_N + PAGE - 1) // PAGE - len(s.blocks)
                if grow > 0:
                    s.blocks.extend(pool.alloc(grow))
                inputs.append(s.next_tok)
                cache_lens.append(s.seq_len)
                block_rows.append(s.blocks)
                slots.append(s.slot)
            gd.begin_window(inputs, cache_lens, block_rows, slots)
            if node_q:
                refill_varlen()
            toks = gd.collect_window()
            if apply_window(batch, toks):
                running = [s for s in running if not s.done]
            if pantry:
                adopt()
            if rolls_done - last_roll_print >= print_every:
                el = time.perf_counter() - t0
                # all Python-side counters — no GPU sync, no throughput cost
                print(f"    [r{rnd}] roll {rolls_done:4d}/{n_target} | tok {tok_total:>10,} | "
                      f"{tok_total / max(el, 1e-9):7,.0f} tok/s | rows {len(running):3d} | "
                      f"pantry {pantry_blocks:3d}b/{len(pantry):2d}j | "
                      f"free {len(pool.free):4d} | {el:6.1f}s", flush=True)
                last_roll_print = rolls_done

        gen_s = time.perf_counter() - t0
        assert pantry_blocks == 0 and not pantry, "pantry not drained"
        leaked = self.pool.num_blocks - 1 - len(self.pool.free) - 2  # null + gd/pfg scratch
        assert leaked == 0, f"{leaked} KV blocks leaked"
        assert len(store.free) == len(store.data), "smear slots leaked"
        return rows, dict(
            gen_s=gen_s, gen_tok=tok_total, refills=refills, replays=pfg.replays,
            prefill_tok=pfg.real_tok, bnd_copies=bnd_copies, stop_fires=stop_fires,
            adopt_stalls=adopt_stalls, prefill_s=prefill_s, adopt_s=adopt_s,
            n_extended=n_ext)

    # -- eager debug path (parity gate) ---------------------------------------
    @torch.no_grad()
    def debug_greedy(self, prompt_ids: list[int], max_new_tokens: int,
                     stop_at_terminal: bool = True) -> list[int]:
        """Greedy generation through the SAME paged forward path, eagerly (no
        graphs, no sampler): prefill prompt[:-1] via prefill_body semantics,
        then argmax-decode starting from the forced prompt[-1]. Used by the
        Phase-2 parity gate against the reference engine."""
        model = self.model
        dev = "cuda"
        ctx = prompt_ids[:-1]
        Lp = len(ctx)
        blocks = self.pool.alloc(nblocks(Lp + 1 + max_new_tokens + 1))
        cfg = model.config
        n_kv, hd = cfg.n_kv_head, cfg.n_embd // cfg.n_head
        k_flat = self.pool.k.view(cfg.n_layer, -1, n_kv, hd)
        v_flat = self.pool.v.view(cfg.n_layer, -1, n_kv, hd)
        ids = torch.tensor(ctx, dtype=torch.long, device=dev)
        pos = torch.arange(Lp, dtype=torch.long, device=dev)
        slot = torch.tensor([blocks[p // PAGE] * PAGE + p % PAGE for p in range(Lp)],
                            dtype=torch.long, device=dev)
        notstart = torch.ones(Lp, 1, dtype=torch.bfloat16, device=dev)
        notstart[0] = 0
        cu = torch.tensor([0, Lp], dtype=torch.int32, device=dev)
        gather = torch.tensor([Lp - 1], dtype=torch.long, device=dev)
        seed = prefill_body(model, ids, pos, cu, slot, notstart, gather,
                            k_flat, v_flat, Lp)
        prev_emb = seed.clone()                       # (1, C)
        block_row = blocks + [0] * (self.max_blocks - len(blocks))
        block_table = torch.tensor([block_row], dtype=torch.int32, device=dev)
        cache_seqlens = torch.tensor([Lp], dtype=torch.int32, device=dev)
        input_ids = torch.tensor([[prompt_ids[-1]]], dtype=torch.long, device=dev)
        out = []
        for _ in range(max_new_tokens):
            logits, x_pre = decode_body(model, input_ids, cache_seqlens, block_table,
                                        self.pool.k, self.pool.v, prev_emb)
            prev_emb.copy_(x_pre)
            cache_seqlens += 1
            nxt = int(torch.argmax(logits, dim=-1).item())
            out.append(nxt)
            if stop_at_terminal and nxt in self.terminal_ids:
                break
            input_ids[0, 0] = nxt
        self.pool.release(blocks)
        return out


# TODO - Implement...
