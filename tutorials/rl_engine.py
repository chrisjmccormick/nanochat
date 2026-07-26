"""
fast_engine.py 
- Paged
- Prefix-sharing 
- Compiled and CUDA-graph captured
- Fused generation and training 
- In-place optimizer updates 
- Smear managed across decode steps

- TODO - lm_head -> slice vocab -> fp32 -> 15*tanh(./15).
    - Slice vocab?

- Smear:
   - Have to hang on to the previous token's post-norm embedding.
   - Prefill has to provide the last embedding from the prompt.
     - Same one used across all K rollouts?

"""

import math
import time
from collections import deque

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
# FA2 primitives (varlen + paged kvcache with block_table) from the community
# `kernels` hub — nanochat's uv env ships `kernels`, not the flash-attn wheel.
from kernels import get_kernel
_fa2 = get_kernel("kernels-community/flash-attn2")
# Kernel revisions differ: newer builds expose the fns at the module top level,
# older ones nested them under `.flash_attn_interface`. Prefer top-level (which
# the current kernels-community/flash-attn2 provides) and fall back to the
# submodule so both layouts work.
_fa2i = _fa2 if hasattr(_fa2, "flash_attn_varlen_func") else _fa2.flash_attn_interface
flash_attn_varlen_func = _fa2i.flash_attn_varlen_func
_fa_kvcache_raw = _fa2i.flash_attn_with_kvcache

from nanochat.gpt import norm, apply_rotary_emb, linear

PAGE = 256  # KV page size (tokens); FA2 paged KV requires a multiple of 256


# -----------------------------------------------------------------------------
# §1. Custom ops (compile-safe wrappers around the FA2 paged/scatter primitives)
# -----------------------------------------------------------------------------
# TODO - This was added in order to get some feature we added to compile:
#          - Does the same issue exist in the community FA2? And community FA3?
#              - If not, we can drop this.
@torch.library.custom_op("nanochat_fast::fa_kvcache_paged", mutates_args=("k_cache", "v_cache"))
def fa_kvcache_paged(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor, cache_seqlens: torch.Tensor,
                     block_table: torch.Tensor, window_left: int, window_right: int) -> torch.Tensor:
    return _fa_kvcache_raw(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                           block_table=block_table, causal=True,
                           window_size=(window_left, window_right))


@fa_kvcache_paged.register_fake
def _(q, k_cache, v_cache, k, v, cache_seqlens, block_table, window_left, window_right):
    return torch.empty_like(q)


@torch.library.custom_op("nanochat_fast::kv_scatter", mutates_args=("k_flat", "v_flat"))
def kv_scatter(k_flat: torch.Tensor, v_flat: torch.Tensor, slot_map: torch.Tensor,
               k: torch.Tensor, v: torch.Tensor) -> None:
    k_flat.index_copy_(0, slot_map, k)
    v_flat.index_copy_(0, slot_map, v)


@kv_scatter.register_fake
def _(k_flat, v_flat, slot_map, k, v):
    return None


# -----------------------------------------------------------------------------
# §2. Engine-side forward paths over the live GPT submodules
# -----------------------------------------------------------------------------
# `B` / batch_size in this context refers to the number of actively decoding rollouts;
# it will be equal to one of the compiled bucket sizes that we've specified. The bucket
# size is dictated by the number of remaining rollouts to generate.
#
# TODO - Put this where appropriate:
# A single page corresponds to the full KV cache for 256 tokens of a single rollout.
# i.e., in nanochat d24, it contains 12 key and 12 value vectors per layer, so 
# 24 x (12 + 12) = 576 vectors per token, and 576 x 256 = 147,456 vectors per page.
# At bf16 and a head size of 128, that's exactly 36MB per page.

# ```349:353:nanochat/nanochat/fast_engine.py
#         shape = (n_layer, self.num_blocks, PAGE, n_kv, head_dim)
#         self.k_buf = VmmBuffer(self.num_blocks * (bpb // 2), dev=device_index)
#         self.v_buf = VmmBuffer(self.num_blocks * (bpb // 2), dev=device_index)
#         self.k = self.k_buf.tensor(shape)
#         self.v = self.v_buf.tensor(shape)
# ```
#
# k_pool / v_pool — shape ( L, nblocks, PAGE, H_kv,   D)
#                         (24,  ~1500?,  256,   12, 128)
#
# I think it's a slightly cleaner mental model if you imagine the blocks as the
# first dimension, because one block contains all layers. But we need to slice
# the block by layer for the attention calls.
#
# Dim       Code name    Meaning
# --------  -----------  ---------------------------------------------------------
# L         n_layer      Transformer layer; k_pool[i]/v_pool[i] per layer in decode_body
# nblocks   num_blocks   Total KV pages in pool (shared); block 0 = null/parked
# PAGE      PAGE (256)   Tokens per page (FA2 paged-KV requirement)
# H_kv      n_kv_head    Key/value head count (GQA)
# D         head_dim     Length of a key (or value) vector.
#

# TODO - Block table...
# Each rollout row doesn’t own a slice of `nblocks` directly — `block_table (B, max_blocks)` maps row → block IDs 
# into that shared pool. As sequences grow, they allocate more pages from the free list.

# ### `prev_emb` — shape `(B, C)`




def decode_body(model, input_ids, cache_seqlens, block_table, k_pool, v_pool, prev_emb):
    """One paged decode step for B rows.

    input_ids (B,1) long | cache_seqlens (B,) int32 | block_table (B,MB) int32 |
    k_pool/v_pool (L, nblocks, PAGE, H_kv, D) | prev_emb (B, C) bf16 — previous
    token's post-norm/pre-smear embedding.
    Returns (softcapped fp32 logits (B, vocab), x_pre (B, C) this token's
    post-norm/pre-smear embedding — caller writes it back into prev_emb).
    """
    cfg = model.config
    B = input_ids.shape[0]
    head_dim = cfg.n_embd // cfg.n_head

    # Input embeddings
    x = F.embedding(input_ids, model.wte)         # (B,1,C) bf16

    # RMS norm
    x = norm(x)

    # TODO - The `:` means all tokens, what's the `0` indexing?
    #        And this is making a copy?
    x_pre = x[:, 0]                               # post-norm, PRE-smear

    # Per token, how much of the previous embedding to add to the current one. (0 - 1.0 x lambda)
    # TODO - Typical value of lambda?
    smear_amnt = model.smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[..., :24], model.smear_gate))
    
    # For each token, add some of the previous token to it.
    x = x + smear_amnt * prev_emb.unsqueeze(1)

    # Lookup RoPE embeddings for all tokens. 
    # TODO - Is it not consistent with the index, since we have continuous batching / pages
    #        from different rollouts?
    positions = cache_seqlens.to(torch.long)      # (B,)
    cos = model.cos[0, :, 0][positions].unsqueeze(1).unsqueeze(1)  # (B,1,1,D/2)
    sin = model.sin[0, :, 0][positions].unsqueeze(1).unsqueeze(1)

    # Note: the model is fully flattened now — every per-layer weight hangs
    # directly off the GPT module (c_q[i], mlp_fc[i], ...), so the layer loop
    # below is the whole forward, written out with F.linear on raw parameters.

    x0 = x
    backout_layer = cfg.n_layer // 2
    x_backout = None
    for i in range(cfg.n_layer):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        xn = norm(x)
        q = linear(xn, model.c_q[i]).view(B, 1, cfg.n_head, head_dim)
        k = linear(xn, model.c_k[i]).view(B, 1, cfg.n_kv_head, head_dim)
        v = linear(xn, model.c_v[i]).view(B, 1, cfg.n_kv_head, head_dim)
        si = str(i)
        if si in model.ve_gate:
            ve = F.embedding(input_ids, model.value_embeds[si]).view(B, 1, cfg.n_kv_head, head_dim).to(x.dtype)
            g = 3 * torch.sigmoid(linear(xn[..., :model.ve_gate_channels], model.ve_gate[si]))
            v = v + g.unsqueeze(-1) * ve
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        wl, wr = model.window_sizes[i]
        y = fa_kvcache_paged(q, k_pool[i], v_pool[i], k, v, cache_seqlens, block_table, wl, wr)
        x = x + linear(y.view(B, 1, -1), model.attn_proj[i])
        x = x + linear(F.relu(linear(norm(x), model.mlp_fc[i])).square(), model.mlp_proj[i])
        if i == backout_layer:
            x_backout = x
    x = x - model.backout_lambda.to(x.dtype) * x_backout
    x = norm(x)

    softcap = 15
    logits = linear(x[:, -1, :], model.lm_head)
    logits = logits[..., :cfg.vocab_size].float()
    logits = softcap * torch.tanh(logits / softcap)
    return logits, x_pre


def prefill_body(model, ids, pos, cu_seqlens, slot_map, notstart, gather_idx, k_flat, v_flat,
                 prefill_t):
    """Packed varlen prefill of node contexts: writes KV into the paged pool via
    kv_scatter and returns each sequence's last-position post-norm/PRE-smear
    embedding (the smear seed for the forced first decode token).

    ids/pos/slot_map (T,) long | notstart (T,1) bf16 in {0,1}, 0 at each doc's
    first position (smear must not cross documents) | gather_idx (S,) long.
    """
    cfg = model.config
    T = ids.shape[0]
    head_dim = cfg.n_embd // cfg.n_head

    x = F.embedding(ids, model.wte)               # (T, C)
    x = norm(x)
    x_pre = x
    x_prev = torch.cat([x[:1], x[:-1]], dim=0)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[:, :24], model.smear_gate))
    x = x + (gate * notstart) * x_prev

    cos = model.cos[0, :, 0][pos].unsqueeze(0).unsqueeze(2)   # (1,T,1,D/2)
    sin = model.sin[0, :, 0][pos].unsqueeze(0).unsqueeze(2)

    x0 = x
    for i in range(cfg.n_layer):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        xn = norm(x)
        q = linear(xn, model.c_q[i]).view(T, cfg.n_head, head_dim)
        k = linear(xn, model.c_k[i]).view(T, cfg.n_kv_head, head_dim)
        v = linear(xn, model.c_v[i]).view(T, cfg.n_kv_head, head_dim)
        si = str(i)
        if si in model.ve_gate:
            ve = F.embedding(ids, model.value_embeds[si]).view(T, cfg.n_kv_head, head_dim).to(x.dtype)
            g = 3 * torch.sigmoid(linear(xn[:, :model.ve_gate_channels], model.ve_gate[si]))
            v = v + g.unsqueeze(-1) * ve
        q = apply_rotary_emb(q.unsqueeze(0), cos, sin)[0]
        k = apply_rotary_emb(k.unsqueeze(0), cos, sin)[0]
        q, k = norm(q) * 1.2, norm(k) * 1.2
        kv_scatter(k_flat[i], v_flat[i], slot_map, k, v)
        wl, wr = model.window_sizes[i]
        y = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                                   max_seqlen_q=prefill_t, max_seqlen_k=prefill_t,
                                   causal=True, window_size=(wl, wr))
        x = x + linear(y.reshape(T, -1), model.attn_proj[i])
        x = x + linear(F.relu(linear(norm(x), model.mlp_fc[i])).square(), model.mlp_proj[i])
    # No backout / final norm / lm_head: prefill only produces KV + smear seeds.
    return x_pre[gather_idx]


# -----------------------------------------------------------------------------
# §3. CUDA-VMM buffer — reserve VA once, map/unmap physical (vmm_pool.py M2,
#     agent-ops/rollout-miner/2026-06-19_0927pm_persistent-graph-vmm, verbatim)
# -----------------------------------------------------------------------------
from cuda.bindings import driver as _drv


def _ck(ret):
    """Check a cuda-python driver return; raise on non-success, else unwrap."""
    if isinstance(ret, tuple):
        err, outs = ret[0], ret[1:]
    else:
        err, outs = ret, ()
    if err != _drv.CUresult.CUDA_SUCCESS:
        _, name = _drv.cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver error: {err} ({name})")
    return outs[0] if len(outs) == 1 else (outs if outs else None)


_WRAP = None


def _get_wrap():
    """Host-only C++ ext: bf16 tensor VIEW over an externally-owned device VA
    (torch::from_blob, no-op deleter). g++ only — no nvcc."""
    global _WRAP
    if _WRAP is None:
        from torch.utils.cpp_extension import load_inline
        src = r"""
        #include <torch/extension.h>
        #include <vector>
        torch::Tensor wrap_bf16(uint64_t ptr, std::vector<int64_t> sizes, int device) {
            auto opts = torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(torch::kCUDA, device);
            return torch::from_blob(reinterpret_cast<void*>(ptr), sizes,
                                    [](void*){}, opts);
        }
        """
        _WRAP = load_inline(name="vmm_wrap", cpp_sources=[src], functions=["wrap_bf16"],
                            with_cuda=False, verbose=False)
    return _WRAP


class VmmBuffer:
    """A fixed virtual address range whose physical backing can be released
    (``unmap``) and re-acquired (``map``). ``ptr`` is constant for the run, so
    captured CUDA graphs binding it stay valid across cycles."""

    def __init__(self, nbytes: int, dev: int = 0):
        _ck(_drv.cuInit(0))
        prop = _drv.CUmemAllocationProp()
        prop.type = _drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev
        gran = _ck(_drv.cuMemGetAllocationGranularity(
            prop, _drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        self.gran = int(gran)
        self.size = (nbytes + self.gran - 1) // self.gran * self.gran
        self.prop = prop
        self.dev = dev
        self.access = _drv.CUmemAccessDesc()
        self.access.location = prop.location
        self.access.flags = _drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.ptr = int(_ck(_drv.cuMemAddressReserve(self.size, 0, 0, 0)))
        self.handle = None
        self.mapped = False
        self.map()

    def map(self) -> None:
        if self.mapped:
            return
        self.handle = _ck(_drv.cuMemCreate(self.size, self.prop, 0))
        _ck(_drv.cuMemMap(self.ptr, self.size, 0, self.handle, 0))
        _ck(_drv.cuMemSetAccess(self.ptr, self.size, [self.access], 1))
        self.mapped = True

    def unmap(self) -> None:
        """Release physical to the driver's free pool; VA stays reserved."""
        if not self.mapped:
            return
        _ck(_drv.cuMemUnmap(self.ptr, self.size))
        _ck(_drv.cuMemRelease(self.handle))
        self.handle = None
        self.mapped = False

    def tensor(self, shape) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        assert n * 2 <= self.size, f"shape {shape} exceeds VA {self.size}B"
        return _get_wrap().wrap_bf16(self.ptr, list(shape), self.dev)


# -----------------------------------------------------------------------------
# §4. Refcounted paged KV pool on VMM reservations
# -----------------------------------------------------------------------------
def nblocks(tokens: int) -> int:
    return (tokens + PAGE - 1) // PAGE


class KVPool:
    """All decode-time KV memory as two big tensors — REFCOUNTED free-list pool.
    Block 0 = null block. K/V live on VMM reservations, mapped once at __init__
    and never unmapped: gen and train are coresident, so the pool is sized to the
    measured round peak instead of being lent back and forth (that handoff cost
    ~0.8 s of remap plus a ~0.9 s first-pack tax per round — see the retired
    lend()/reclaim() protocol in git history)."""

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
    indexed by slot id. Rows hold a slot from mint to retire; seed_window
    gathers slots -> the dense graph buffer ONCE at round start, after which
    prev_emb lives in place (positions are stable between bucket drops)."""

    def __init__(self, n_slots: int, n_embd: int, device):
        self.data = torch.zeros(n_slots, n_embd, dtype=torch.bfloat16, device=device)
        self.free = list(range(n_slots - 1, -1, -1))

    def alloc(self) -> int:
        assert self.free, "SmearStore exhausted — raise MAX_SEQS slack"
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
                 "n_full", "partial", "rows_left", "seed_emb")

    def __init__(self, context_ids: list[int], cand_jobs: list[dict]):
        self.fed_ids = context_ids
        self.plen = len(context_ids)
        self.cand_jobs = cand_jobs
        self.blocks: list[int] = []
        self.seq_len = 0
        self.n_full = self.plen // PAGE
        self.partial = (self.plen % PAGE) != 0
        self.rows_left = len(cand_jobs)
        self.seed_emb: torch.Tensor | None = None  # (C,) bf16, set after prefill

    def new_pages(self) -> int:
        return nblocks(self.plen) + (max(len(self.cand_jobs) - 1, 0) if self.partial else 0)


class Seq:
    """One rollout row. Context KV is SHARED (aliased prefix pages); ``forced``
    (= prompt[-1]) is the first decode input; ``gen`` collects sampled tokens."""
    __slots__ = ("job", "node", "forced", "plen", "allow", "blocks", "n_shared",
                 "seq_len", "gen", "next_tok", "need", "done", "slot")

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
        # pinned host mirror of token_record: the window's ONE D2H lands here
        # and is scanned as numpy — no python list materialization.
        self.tok_host = torch.empty(max_seqs, macro_n, dtype=torch.long, pin_memory=True)
        self.tok_host_np = self.tok_host.numpy()
        self.inv_temp = torch.tensor(1.0 / temperature, dtype=torch.float32, device=dev)
        self.top_p_t = torch.tensor(top_p, dtype=torch.float32, device=dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mempool = None
        self._scratch = pool.alloc(1)[0]

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

    # -- SoA window protocol: the graph ITSELF carries inputs / cache_seqlens /
    # prev_emb between windows (each replay writes the sampled token into
    # input_ids and advances cache_seqlens), so as long as a row keeps its
    # batch position, a steady-state window uploads NOTHING and downloads one
    # pinned (bucket, N) token block. Rows move only at bucket drops (compact)
    # and dead rows are parked in place (park) — the capture-time padded-row
    # state — until the next drop reclaims their position.

    def seed_window(self, forced_np, cache_lens_np, block_tables_np, slots_np,
                    bucket: int) -> None:
        """Round-start upload (window 0): forced first inputs, context lens,
        initial block rows, smear seeds gathered from the rows' store slots.
        The only full-state upload of the round."""
        B = len(forced_np)
        dev = "cuda"
        self.input_ids[:B, 0] = torch.from_numpy(forced_np).to(dev, non_blocking=True)
        self.cache_seqlens[:B] = torch.from_numpy(cache_lens_np).to(dev, non_blocking=True)
        self.block_table[:B] = torch.from_numpy(block_tables_np).to(dev, non_blocking=True)
        slots_t = torch.from_numpy(slots_np).to(dev, non_blocking=True)
        self.prev_emb[:B] = self.store.data[slots_t]
        self.park(B, bucket)

    def park(self, positions, bucket: int = 0) -> None:
        """cache 0 + null block table for dead/padded rows (int start => park
        [start:bucket); ndarray => park those positions). A parked row keeps
        replaying at bucket cost but writes only the null block and attends
        over ~nothing; its cache drifts +N/window until re-parked or dropped."""
        if isinstance(positions, np.ndarray):
            if positions.size == 0:
                return
            idx = torch.from_numpy(positions).to("cuda", non_blocking=True)
            self.cache_seqlens.index_fill_(0, idx, 0)
            self.block_table.index_fill_(0, idx, 0)
        elif bucket > positions:
            self.cache_seqlens[positions:bucket] = 0
            self.block_table[positions:bucket] = 0

    def compact(self, keep_np) -> None:
        """Bucket drop: gather survivors to the head positions, carrying their
        device-resident inputs/cache/blocks/smear state (one tiny index H2D;
        index_select materializes before the head copy, so overlap is safe)."""
        k = len(keep_np)
        idx = torch.from_numpy(keep_np).to("cuda", non_blocking=True)
        for buf in (self.input_ids, self.cache_seqlens, self.block_table, self.prev_emb):
            buf[:k].copy_(buf.index_select(0, idx))

    def replay_window(self, bucket: int) -> None:
        g = self.graphs[bucket]
        for i in range(self.macro_n):
            g.replay()
            self.token_record[:bucket, i] = self.tok_buf[:bucket]

    def collect_np(self, bucket: int):
        """The window's ONE host sync: pinned D2H of the (bucket, N) record,
        returned as a numpy view (valid until the next window)."""
        self.tok_host[:bucket].copy_(self.token_record[:bucket], non_blocking=True)
        torch.cuda.synchronize()
        return self.tok_host_np[:bucket]


# -----------------------------------------------------------------------------
# §8. FastEngine — the fused generation driver (speedrun §6 run_gen)
# -----------------------------------------------------------------------------
def default_buckets(ceiling_rows: int, max_seqs: int, step: int = 16) -> tuple:
    roundup = lambda n: min(max_seqs, -(-n // step) * step)
    top = roundup(max(ceiling_rows, 1))
    coarse = {b for b in (64, 128) if b < top}
    return tuple(sorted(coarse | {top}))


class PrefillAllEngine:
    """Prefill every context once, then decode the whole round to completion.

    The RL-train regime is a fixed handful of contexts (PPR problems x K rows)
    that, at the train budget, fit the pool all at once: one prefill replay mints
    and admits every row, then the round decodes them down — no mid-round refill,
    no adoption. Both invariants (one replay, whole round resident) are asserted,
    NOT recovered from: a trip means the config left the tuned path and we want to
    know. An eval round is far too many rows to admit this way, so eval runs
    offline (agent-ops `eval_trajectory.py`) rather than in the RL loop.

    Life cycle: __init__ builds the pool + graphs (untimed setup), then
    `run_round(nodes)` per round with the KV pool permanently mapped — gen and
    train are coresident, so nothing is handed back between phases.
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
        self.model, self.tok = model, tokenizer
        self.max_tokens = max_tokens
        self.macro_n, self.prefill_t, self.prefill_seqs = macro_n, prefill_t, prefill_seqs
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
        # Admission ceiling: whole pool over the worst-case per-row reserve —
        # nothing is held back, the round owns the pool.
        min_priv = nblocks(2 + max_tokens + macro_n)  # smallest plausible context
        self.ceiling_rows = max(1, len(self.pool.free) // min_priv)
        if buckets is None:
            buckets = default_buckets(min(self.ceiling_rows, max_seqs), max_seqs)
        assert max(buckets) <= max_seqs
        self.buckets = tuple(sorted(buckets))
        self.max_seqs = max_seqs

        n_slots = max_seqs + 64  # one smear slot per resident row + headroom
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
        # mid-decode in a fused gen+train process; prefill stays on the decode stream.
        self.prefill_stream = torch.cuda.current_stream()

    @torch.no_grad()
    def capture(self, warm_context: list[int]) -> None:
        """Capture decode buckets + the prefill graph; warm one prefill replay."""
        temp, top_p, top_k = self.sampler_cfg
        print(f"  engine config: kv_pool {self.kv_pool_gb:g} GB "
              f"({self.pool.num_blocks} blocks x {PAGE} tok) | max_seqs {self.max_seqs} | "
              f"buckets {self.buckets} | macro_n {self.macro_n} | "
              f"prefill T={self.prefill_t} x{self.prefill_seqs} seqs | "
              f"max_tokens {self.max_tokens} | "
              f"temp {temp:g} top_p {top_p:g} top_k {top_k} | "
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
            jobs = [dict(meta=meta, prompt_ids=list(prompt_ids), forced=prompt_ids[-1],
                         allow=allow, budget=allow, final=True)
                    for _ in range(k)]
            nodes.append(Node(list(prompt_ids[:-1]), jobs))
        return nodes

    # -- the round driver ------------------------------------------------------
    @torch.no_grad()
    def run_round(self, nodes_all: list["Node"], rnd: int = 0,
                  on_retire=None) -> tuple[list[dict], dict]:
        """One round, static-prefill: prefill every context in a SINGLE replay,
        mint + admit all rows, then decode to completion — no mid-round refill.
        Returns (rows, stats); each row: dict(meta, completion_token_ids,
        completion_text, terminal, stop_reason, finish_reason). `on_retire(row)`
        fires as rows finish (e.g. inline grading)."""
        pfg, gd, pool, store = self.pfg, self.gd, self.pool, self.store
        MACRO_N = self.macro_n
        pfg.replays = pfg.real_tok = 0
        rows: list[dict] = []
        rolls_done = tok_total = 0
        stop_fires = bnd_copies = 0
        n_target = sum(len(n.cand_jobs) for n in nodes_all)
        print_every = self.print_every or max(1, n_target // 8)

        # -- preconditions: one prefill replay, whole round resident (no fallback)
        row_cap = min(self.max_seqs, max(self.buckets))
        assert n_target <= row_cap, \
            f"{n_target} rows > row_cap {row_cap} (raise MAX_SEQS / top bucket)"
        assert len(nodes_all) <= pfg.prefill_seqs, \
            f"{len(nodes_all)} contexts > PREFILL_SEQS={pfg.prefill_seqs}"
        ctx_tok = sum(nd.plen for nd in nodes_all)
        assert ctx_tok <= pfg.prefill_t, \
            f"round context {ctx_tok} tok > PREFILL_T={pfg.prefill_t} " \
            f"(assemble balanced rounds, or raise PREFILL_T)"

        def mint_all(nodes: list[Node]) -> list[Seq]:
            """Mint every node's K sibling rows: full context pages are shared
            (addref); a partial last page is aliased by row 0 and copy-on-written
            for the rest. GPU work is BATCHED: the K-1 boundary-page clones of a
            node share one source page, so they go as ONE broadcast scatter per
            node (RHS is a zero-copy view) over just the OCCUPIED prefix rows —
            the stale tail is always decode-written before any read. The naive
            per-sibling strided copy_ pair was ~8k tiny kernels / ~100 ms a
            round, and moved the full page. Smear seeds land in one index_put."""
            nonlocal bnd_copies
            seqs = []
            for nd in nodes:
                full_pages = nd.blocks[:nd.n_full]
                src_bnd = nd.blocks[nd.n_full] if nd.partial else None
                dsts = []
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
                            dsts.append(dst)
                            s.blocks.append(dst)
                            bnd_copies += 1
                    s.seq_len = nd.plen
                    s.slot = store.alloc()
                    s.next_tok = s.forced
                    seqs.append(s)
                if dsts:
                    used = nd.plen - nd.n_full * PAGE   # occupied boundary rows
                    dst_t = torch.tensor(dsts, dtype=torch.long, device="cuda")
                    pool.k[:, dst_t, :used] = pool.k[:, src_bnd, :used].unsqueeze(1)
                    pool.v[:, dst_t, :used] = pool.v[:, src_bnd, :used].unsqueeze(1)
                pool.release(full_pages)  # node's own ref drops; K row refs remain
            # Seed every sibling's smear state with its node's last-context
            # pre-smear embedding (computed by the prefill graph) — one scatter.
            slots_t = torch.tensor([s.slot for s in seqs], dtype=torch.long, device="cuda")
            seeds = torch.stack([nd.seed_emb for nd in nodes])
            reps = torch.tensor([len(nd.cand_jobs) for nd in nodes], device="cuda")
            store.data[slots_t] = torch.repeat_interleave(seeds, reps, dim=0)
            return seqs

        def retire(s: Seq, full_gen: list[int], eos: bool, stop: str | None = None) -> None:
            nonlocal rolls_done
            s.done = True
            pool.release(s.blocks)
            s.blocks = []
            store.release(s.slot)
            j = s.job
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

        def _stop_hit_ids(ids: list[int]) -> str | None:
            tail = self.tok.decode(ids)
            hits = [ss for ss in self.stop_strings if ss in tail]
            if not hits:
                return None
            return max(hits, key=lambda ss: tail.rfind(ss))

        t0 = time.perf_counter()
        # -- prefill EVERY context in one replay, mint + admit ALL rows ---------
        pfg.run(nodes_all)
        running: list[Seq] = mint_all(nodes_all)
        assert pfg.replays == 1, \
            f"round used {pfg.replays} prefill replays, not 1 — raise PREFILL_T/PREFILL_SEQS"
        # Every context is resident; private decode pages grow lazily below. The
        # round is NOT guaranteed to fit at worst case (sum(s.need) can exceed the
        # pool when PPR*K rows each reserve their full budget) — it fits because
        # most rows complete well short of the budget and free their pages first.
        # We admit all rows and report the headroom; the decode-loop guard fails
        # LOUD if a round ever actually exhausts the pool (no fallback — that means
        # the config is too large for the pool).
        wc_reserve = sum(s.need for s in running)
        if rnd == 0:
            over = wc_reserve > self.pool.num_blocks
            print(f"    [r{rnd}] admitted {len(running)} rows in one wave | worst-case "
                  f"reserve {wc_reserve} vs pool {self.pool.num_blocks} blocks "
                  f"({'OVER-COMMIT, relies on early completion' if over else 'fits worst-case'})",
                  flush=True)
        min_free = len(pool.free)

        # -- SoA scheduler state: batch position is STABLE between bucket drops,
        # so the graph carries inputs/cache/prev_emb across windows and a quiet
        # window's host work is one pinned D2H + numpy scans. All rows minted in
        # one wave => every live row's gen length is exactly w*MACRO_N.
        seqs = running
        B0 = len(seqs)
        orig = np.arange(B0)                  # position -> gen_buf row (stable id)
        live = np.ones(B0, dtype=bool)
        plens = np.array([s.plen for s in seqs], dtype=np.int64)
        allows = np.array([s.allow for s in seqs], dtype=np.int64)
        n_pages = np.array([len(s.blocks) for s in seqs], dtype=np.int64)
        gen_buf = np.empty((B0, (-(-int(allows.max()) // MACRO_N)) * MACRO_N),
                           dtype=np.int64)
        TERM = np.fromiter(self.terminal_ids, dtype=np.int64)
        GATE = (np.fromiter(self.gate_ids, dtype=np.int64)
                if self.stop_detect and self.gate_ids else None)

        bucket = next(x for x in self.buckets if x >= B0)
        bt0 = np.zeros((B0, self.max_blocks), dtype=np.int32)
        for p, s in enumerate(seqs):
            bt0[p, :len(s.blocks)] = s.blocks
        gd.seed_window(np.array([s.next_tok for s in seqs], dtype=np.int64),
                       plens.astype(np.int32), bt0,
                       np.array([s.slot for s in seqs], dtype=np.int64), bucket)

        last_roll_print = 0
        w = 0
        park_dirty = False                    # dead rows drift until re-parked
        while True:

            # ### Clearer rename suggestions

            # | Current | Likely meaning | Clearer name |
            # |--------|----------------|--------------|
            # | `lp`      | live batch indices     | `live_batch_indices` or `live_row_indices` |
            # | `fast_lp` | live rows, fast path   | `fast_live_indices`                    |
            # | `nb`      | next bucket size       | `next_bucket_size`                     |
            # | `gidx`    | rows needing KV pages  | `rows_needing_pages`                   |
            # | `toks_np` | sampled tokens (numpy) | `sampled_tokens` or `window_tokens_np` |
            # | `p`       | index into batch       | `batch_idx` or `row_idx`               |

            # Example of the compaction block with clearer names:

            # ```python
            # live_row_indices = np.flatnonzero(live)
            # if live_row_indices.size == 0:
            #     break
            # next_bucket_size = next(x for x in self.buckets if x >= live_row_indices.size)
            # if next_bucket_size < bucket:
            #     gd.compact(live_row_indices)
            #     ...
            #     live_row_indices = np.arange(len(seqs))  # survivors now at 0..k-1
            # ```

            lp = np.flatnonzero(live)
            if lp.size == 0:
                break
            nb = next(x for x in self.buckets if x >= lp.size)
            if nb < bucket:                   # bucket drop: compact survivors
                gd.compact(lp)
                seqs = [seqs[p] for p in lp]
                orig, plens, allows, n_pages = orig[lp], plens[lp], allows[lp], n_pages[lp]
                live = np.ones(lp.size, dtype=bool)
                bucket = nb
                gd.park(lp.size, bucket)
                park_dirty = False
                lp = np.arange(lp.size)
            elif park_dirty:
                gd.park(np.flatnonzero(~live))
                park_dirty = False
            # vectorized page growth; ONE consolidated block-table scatter
            need = (plens + (w + 1) * MACRO_N + PAGE - 1) // PAGE - n_pages
            need[~live] = 0
            gidx = np.flatnonzero(need > 0)
            if gidx.size:
                total = int(need[gidx].sum())
                if total > len(pool.free):
                    raise RuntimeError(
                        f"KV pool exhausted mid-round r{rnd}: need {total} pages, "
                        f"{len(pool.free)} free, {lp.size} rows live — round too "
                        f"large for the pool (lower MAX_TOKENS / PPR / K, or raise "
                        f"KV_POOL_GB)")
                fresh = pool.alloc(total)     # one batch pop, row-major chunks
                flat_pos, o = [], 0
                for p in gidx:
                    s, k = seqs[p], int(need[p])
                    flat_pos.extend(int(p) * gd.max_blocks + c
                                    for c in range(len(s.blocks), len(s.blocks) + k))
                    s.blocks.extend(fresh[o:o + k])
                    o += k
                n_pages[gidx] += need[gidx]
                upd = torch.tensor([flat_pos, fresh],
                                   dtype=torch.int64).to("cuda", non_blocking=True)
                gd.block_table.view(-1)[upd[0]] = upd[1].to(torch.int32)
            min_free = min(min_free, len(pool.free))
            gd.replay_window(bucket)
            toks_np = gd.collect_np(bucket)   # the window's single host sync
            # event scan: rows with a terminal / stop-gate / budget crossing go
            # to the exact per-token path; everyone else bulk-appends.
            t_live = toks_np[lp]
            slow = np.isin(t_live, TERM).any(axis=1)
            if GATE is not None:
                slow |= np.isin(t_live, GATE).any(axis=1)
            slow |= (w + 1) * MACRO_N >= allows[lp]
            base = w * MACRO_N
            fast_lp = lp[~slow]
            gen_buf[orig[fast_lp], base:base + MACRO_N] = toks_np[fast_lp]
            tok_total += int(fast_lp.size) * MACRO_N
            any_done = False
            for p in lp[slow]:
                s = seqs[p]
                cur = toks_np[p]
                gen_row = gen_buf[orig[p]]
                retired = False
                for j in range(MACRO_N):
                    t = int(cur[j])
                    tok_total += 1
                    if t in self.terminal_ids:
                        retire(s, gen_row[:base].tolist() + cur[:j + 1].tolist(), True)
                        retired = True
                        break
                    if base + j + 1 >= s.allow:  # budget reached = truncated
                        retire(s, gen_row[:base].tolist() + cur[:j + 1].tolist(), False)
                        retired = True
                        break
                    if self.stop_detect and t in self.gate_ids:
                        glen = base + j + 1
                        take = min(self.window_tokens, glen)
                        from_cur = min(take, j + 1)
                        from_gen = take - from_cur
                        ids = (gen_row[base - from_gen:base].tolist()
                               + cur[j + 1 - from_cur:j + 1].tolist())
                        if take < self.window_tokens:  # pad from the pre-gen stream
                            pre = list(s.job["prompt_ids"])
                            ids = pre[len(ids) - self.window_tokens:] + ids
                        hit = _stop_hit_ids(ids)
                        if hit is not None:
                            stop_fires += 1
                            retire(s, gen_row[:base].tolist() + cur[:j + 1].tolist(),
                                   False, stop=hit)
                            retired = True
                            break
                if retired:
                    live[p] = False
                    any_done = True
                else:                          # gate false alarm: full window kept
                    gen_row[base:base + MACRO_N] = cur
            if any_done:
                park_dirty = True
            w += 1
            if rolls_done - last_roll_print >= print_every:
                el = time.perf_counter() - t0
                # all Python-side counters — no GPU sync, no throughput cost
                print(f"    [r{rnd}] roll {rolls_done:4d}/{n_target} | tok {tok_total:>10,} | "
                      f"{tok_total / max(el, 1e-9):7,.0f} tok/s | rows {int(live.sum()):3d} | "
                      f"free {len(pool.free):4d} | {el:6.1f}s", flush=True)
                last_roll_print = rolls_done

        gen_s = time.perf_counter() - t0
        leaked = self.pool.num_blocks - 1 - len(self.pool.free) - 2  # null + gd/pfg scratch
        assert leaked == 0, f"{leaked} KV blocks leaked"
        assert len(store.free) == len(store.data), "smear slots leaked"
        return rows, dict(
            gen_s=gen_s, gen_tok=tok_total, replays=pfg.replays,
            prefill_tok=pfg.real_tok, bnd_copies=bnd_copies, stop_fires=stop_fires,
            peak_blocks=self.pool.num_blocks - 1 - min_free)

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
