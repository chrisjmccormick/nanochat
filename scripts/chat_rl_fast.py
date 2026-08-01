"""chat_rl_fast.py — fused same-process RL speedrun on GSM8K.

Generation and training are CORESIDENT: the KV pool is sized to the measured
round peak (default 28 GB ~= 796 blocks vs an observed worst round of 599) and
mapped once for the life of the process. Earlier revisions time-shared the card
by lending the pool's physical pages back to training between phases, which cost
~0.8 s of remap plus a ~0.9 s first-pack tax every round (training rebuilt its
flushed activation segments inside pack 1). Dropping the handoff took the round
wall from ~10.5 s to ~8.5 s at identical rollout statistics; the pool-exhaustion
guard stays fail-loud, and a trip means policy collapse.

The prefill pack is auto-sized from the pre-designed round schedule (§1), and
the KV pool is the one number to revisit if the regime changes: capacity is
pool 28 + train peak ~44.5 reserved + context ~3 = ~75.5 of 80 GB.

The fast counterpart of scripts/chat_rl.py: generation runs through the paged,
prefix-shared, CUDA-graph engine (nanochat/fast_engine.py) fused into the
training process — one model instance, in-place full-param bf16 updates with
fp32-state Muon/AdamW, no reload / re-capture between rounds.

Per ROUND: (1) decode K fresh natural rollouts for each of PROBLEMS_PER_ROUND
GSM8K train problems through the captured graphs, (2) grade with the `#### <n>`
regex reward, (3) take ONE REINFORCE (DAPO global token-mean) optimizer step on
ALL parameters.

ALGORITHMIC PARITY with scripts/chat_rl.py — the baseline is stock nanochat
(minus tool use, removed repo-wide) and is NOT modified; this script reproduces
its learning behavior so a fast-vs-baseline A/B measures the engine, not a
different algorithm. Reproduced exactly: sampler temp 1.0 / top-k 50, advantage
r - mean per problem group, loss on every completion token except the terminal,
chat_rl's loss normalizer (each group split into passes of DEVICE_BATCH_SIZE
rollouts, pass loss summed and divided by that pass's own completion-token
count x num_passes x examples_per_rank — emulated host-side as a per-doc
advantage scale, see train_step), chat_rl's LR defaults with linear rampdown to
zero, unconditional optimizer step, no grad clip. Zero-signal groups produce
exactly-zero gradients in the baseline; we skip their docs outright (identical
gradient, less compute). The remaining differences are engine-only: paged
prefix-shared KV, captured decode graphs, in-graph Gumbel-max sampling,
compiled packed-varlen training forward (stock runs it eager), bf16 params
with fp32 masters.

Train-only: no in-loop eval (a pass@k round is more concurrent rows than the
pool admits in one wave) — score the SAVE_EVERY checkpoints offline.

1 GPU:   python -m scripts.chat_rl_fast <tag>
8 GPUs:  torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_fast <tag>

Config via env (speedrun runner convention), see §0 below.
"""

# -----------------------------------------------------------------------------
# §0. Config
# -----------------------------------------------------------------------------
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile as torch_profile, record_function

from nanochat.flash_attention import USE_FA
# Packed RL training REQUIRES real FA varlen kernels: the shim's SDPA fallback
# has no per-document isolation and would silently train on wrong attention.
assert USE_FA, "flash_attention resolved to the SDPA fallback — unsupported for packed RL"

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.dataloader import build_reinforce_packs, assemble_balanced_rounds
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.gpt import cast_model_bf16, setup_fp32_optimizer
from nanochat.fast_engine import PrefillAllEngine

from tasks.gsm8k import GSM8K

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_flag(name, default):
    return os.environ.get(name, str(default)) == "1"


# Rollouts / rounds
K_DRAWS      = _env_int("K", 16)                  # rollouts per problem per round
PPR          = _env_int("PROBLEMS_PER_ROUND", 16) # global, across ranks
EPOCHS       = _env_int("EPOCHS", 1)
ROUNDS_CAP   = _env_int("ROUNDS", 0)              # 0 = full EPOCHS horizon
MAX_TOKENS   = _env_int("MAX_TOKENS", 256)
# Sampler (chat_rl's: temp 1.0, top-k 50, no nucleus)
TEMPERATURE  = _env_float("TEMPERATURE", 1.0)
TOP_K        = _env_int("TOP_K", 50)              # baked into the captured graph
# chat_rl's --device-batch-size: in the baseline this is a memory knob, but it
# also shapes the LOSS — each group of K rollouts trains in K/B passes, each
# normalized by its own token count. We reproduce that pass partition
# host-side, so this must match the baseline run being compared against.
DEVICE_BATCH_SIZE = _env_int("DEVICE_BATCH_SIZE", 8)
assert K_DRAWS % DEVICE_BATCH_SIZE == 0, "K must divide into chat_rl-style passes"
# LRs: chat_rl's own CLI defaults, so both sides train identically by default.
# Env-overridable the same way chat_rl's args are — override BOTH or NEITHER.
UNEMBEDDING_LR = _env_float("UNEMBEDDING_LR", 0.004)
EMBEDDING_LR   = _env_float("EMBEDDING_LR", 0.2)
MATRIX_LR      = _env_float("MATRIX_LR", 0.02)
SCALAR_LR      = _env_float("SCALAR_LR", 0.5)     # setup_optimizer's default (chat_rl leaves it)
# FREEZE_SCALARS=1: zero the LR on ALL per-layer scalar groups (resid_lambdas,
# x0_lambdas, smear_gate/smear_lambda/backout_lambda). Their pretraining
# trajectories are smooth and deliberate — RL should not touch them (Chris).
FREEZE_SCALARS = _env_flag("FREEZE_SCALARS", 0)
WEIGHT_DECAY   = _env_float("WEIGHT_DECAY", 0.0)
# chat_rl's --init-lr-frac; the LR then ramps down linearly to zero over the
# run, exactly as the baseline schedules it.
INIT_LR_FRAC   = _env_float("INIT_LR_FRAC", 0.05)
_TB_ENV = os.environ.get("TRAIN_BUCKETS")
TRAIN_BUCKETS = tuple(int(x) for x in _TB_ENV.split(",")) if _TB_ENV else (16384,)
MAX_NUM_DOCS  = _env_int("MAX_NUM_DOCS", 64)
COMPILE_TRAIN = _env_flag("COMPILE_TRAIN", 1)
# Engine knobs
KV_POOL_GB   = _env_float("KV_POOL_GB", 28)  # sized to the measured round peak; stays mapped
MAX_SEQS     = _env_int("MAX_SEQS", 256)
MACRO_N      = _env_int("MACRO_N", 8)
_BK_ENV = os.environ.get("BUCKETS")
BUCKETS      = tuple(int(x) for x in _BK_ENV.split(",")) if _BK_ENV else None
PREFILL_T    = _env_int("PREFILL_T", 0)           # 0/unset -> auto-size (see §1)
PREFILL_SEQS = _env_int("PREFILL_SEQS", 12)
COMPILE      = _env_flag("COMPILE", 1)
PREFILL_COMPILE = _env_flag("PREFILL_COMPILE", 1)
PREFILL_FULLGRAPH = _env_flag("PREFILL_FULLGRAPH", 1)
# Checkpoint
SAVE_EVERY    = _env_int("SAVE_EVERY", 60)        # 0 = only at end
SAVE_OPT      = _env_flag("SAVE_OPT", 0)          # fp32 optimizer state is resumable
SAVE_ROLLOUTS = _env_flag("SAVE_ROLLOUTS", 0)
# PROFILE=1: chrome trace for ui.perfetto.dev (speedrun_v7-profiled pattern).
# Steps are ROUNDS: PROF_WAIT warm rounds skipped, 1 profiler-warmup round, then
# PROF_ACTIVE rounds recorded (CPU+CUDA, with_stack for the python trace); the
# run is capped to exactly that horizon and trace_<tag>.json.gz lands in cwd.
# record_function labels mark gen / grade / train phases — the compiled decode
# and train bodies swallow interior labels, so labels sit OUTSIDE them.
PROFILE     = _env_flag("PROFILE", 0)
PROF_WAIT   = _env_int("PROF_WAIT", 3)
PROF_ACTIVE = _env_int("PROF_ACTIVE", 1)
OUT_TAG       = os.environ.get("OUT_TAG", "d24-fastrl")
SOURCE        = os.environ.get("SOURCE", "sft")
# FIXED_PROBLEMS: csv of train indices — every round trains on exactly these
# problems (single/multi-problem overfit smoke, like the speedrun's sp1).
FIXED_PROBLEMS = [int(x) for x in os.environ.get("FIXED_PROBLEMS", "").split(",") if x] or None
# POOL_PROBLEMS: csv of train indices to RESTRICT the on-policy round sampler to
# (unlike FIXED, still draws PPR problems/round from this pool, rotating). Use to
# RL only on a difficulty-selected subset (e.g. the 0<rate<1 signal band) while
# keeping the normal 32-problem mini-batch structure.
POOL_PROBLEMS = [int(x) for x in os.environ.get("POOL_PROBLEMS", "").split(",") if x] or None
MODEL_TAG     = os.environ.get("MODEL_TAG") or None
MODEL_STEP    = int(os.environ["MODEL_STEP"]) if os.environ.get("MODEL_STEP") else None
PUSH          = _env_flag("PUSH", 0)
MODEL_REPO    = os.environ.get("MODEL_REPO", "ChrisMcCormick/nanochat-varlen-d24-2026-03-22")

HERE = Path.cwd()

# -----------------------------------------------------------------------------
# §1. Init compute, model, tasks, prompts
# -----------------------------------------------------------------------------
ddp, rank, local_rank, world_size, device = compute_init("cuda")
master = rank == 0
assert PPR % world_size == 0, "PROBLEMS_PER_ROUND must divide by world size"
ppr_rank = PPR // world_size

t = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
model, tokenizer, meta = load_model(SOURCE, device, phase="eval",
                                    model_tag=MODEL_TAG, step=MODEL_STEP)
ASSISTANT_END = tokenizer.encode_special("<|assistant_end|>")
BOS = tokenizer.get_bos_token_id()
PAD_ID = BOS
SEQ_CAP = model.config.sequence_len

train_task = GSM8K(subset="main", split="train")

print0("rendering prompts ...", flush=True)
train_convs = [train_task[i] for i in range(len(train_task))]
train_prompts = [tokenizer.render_for_completion(c) for c in train_convs]
max_prompt = max(len(p) for p in train_prompts)
assert max_prompt + 1 + MAX_TOKENS <= SEQ_CAP, "prompt+budget exceeds model context"
pool = POOL_PROBLEMS if POOL_PROBLEMS is not None else list(range(len(train_task)))
shard = pool[rank::world_size]
if FIXED_PROBLEMS is not None:
    round_schedule = None                         # same fixed problems every round
    num_rounds = (len(pool) // PPR) * EPOCHS
    round_max = sum(len(train_prompts[i]) - 1 for i in FIXED_PROBLEMS)
else:
    # Balanced round assembly (dataloader): each round's contexts must fit ONE
    # static-prefill replay (Sigma context <= PREFILL_T). Stratified partition
    # keeps per-round context sums near the mean, so the longest round is only
    # marginally above the mean and the prefill pack can be sized right to it.
    round_schedule, _sched = assemble_balanced_rounds(
        [(i, len(train_prompts[i]) - 1) for i in shard], ppr_rank, epochs=EPOCHS)
    num_rounds = len(round_schedule)
    round_max = _sched["max"]
    print0(f"[{TAG}] balanced rounds: per-round context tokens "
           f"min/mean/max {_sched['min']}/{_sched['mean']:.0f}/{_sched['max']}", flush=True)

# Prefill pack length. Every round is pre-designed above, so the longest prefill
# batch of the whole epoch is known here — size the static graph to it (rounded up
# to a multiple of 64) rather than over-provisioning: the graph pushes PREFILL_T
# positions through the dense layers every replay whether or not they hold real
# tokens, and pads the tail into one attended segment. Setting PREFILL_T in the env
# pins it instead (to reproduce an earlier run's shape).
prefill_need = max(round_max, max_prompt)
if PREFILL_T:
    assert PREFILL_T >= prefill_need, (
        f"PREFILL_T={PREFILL_T} < {prefill_need} tok needed (longest prefill batch "
        f"{round_max}, longest prompt {max_prompt})")
    _how = f"pinned by PREFILL_T={PREFILL_T}"
else:
    PREFILL_T = -(-prefill_need // 64) * 64
    _how = f"auto-sized to %64: {PREFILL_T:,}"
print0(f"[{TAG}] Longest prefill batch: {round_max:,} tokens, prefill pack is {_how} "
       f"tokens ({100 * round_max / PREFILL_T:.0f}% packed at the longest round)",
       flush=True)
# Both schedule modes lay rounds out epoch-by-epoch (a full pass over the pool
# each), so this division is exact and epoch boundaries are real pass boundaries.
rounds_per_epoch = max(1, num_rounds // EPOCHS)
if ROUNDS_CAP:
    num_rounds = min(num_rounds, ROUNDS_CAP)
if PROFILE:
    num_rounds = min(num_rounds, PROF_WAIT + 1 + PROF_ACTIVE)
    print0(f"[{TAG}] PROFILE: {num_rounds} rounds "
           f"(wait {PROF_WAIT} + warmup 1 + active {PROF_ACTIVE})")
print0(f"[{TAG}] {PPR} problems x K={K_DRAWS} = {PPR * K_DRAWS} rollouts/round "
       f"x {num_rounds} rounds @ budget {MAX_TOKENS} | max prompt {max_prompt} tok "
       f"| train buckets {TRAIN_BUCKETS}", flush=True)

# -----------------------------------------------------------------------------
# §2. Optimizer (fp32 master/state) -> bf16 cast -> engine + graph capture
# -----------------------------------------------------------------------------
print0(f"[{TAG}] raw LRs (chat_rl defaults unless env-pinned): "
       f"unembedding {UNEMBEDDING_LR:g} | embedding {EMBEDDING_LR:g} | "
       f"matrix {MATRIX_LR:g} | scalar {SCALAR_LR:g}")

# Snapshot fp32 masters from the checkpoint weights BEFORE the bf16 cast.
optimizer = setup_fp32_optimizer(model, unembedding_lr=UNEMBEDDING_LR,
                                 embedding_lr=EMBEDDING_LR, matrix_lr=MATRIX_LR,
                                 weight_decay=WEIGHT_DECAY, scalar_lr=SCALAR_LR)
if FREEZE_SCALARS:
    _scalar_ids = {id(model.resid_lambdas), id(model.x0_lambdas),
                   id(model.smear_gate.weight), id(model.smear_lambda),
                   id(model.backout_lambda)}
    for group in optimizer.param_groups:
        if any(id(p) in _scalar_ids for p in group["params"]):
            group["lr"] = 0.0
    print0("FREEZE_SCALARS: resid/x0/smear/backout groups at lr 0")
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * INIT_LR_FRAC
    group["initial_lr"] = group["lr"]
# setup_fp32_optimizer's fixed group order: the 6 AdamW groups, then Muon
# shape-groups (all at matrix_lr).
_GROUP_NAMES = ["unembed", "embed", "value_emb", "resid", "x0", "smear"]
_eff = {(_GROUP_NAMES[i] if i < 6 else "muon"): g["lr"]
        for i, g in enumerate(optimizer.param_groups)}
print0(f"[{TAG}] initial LRs (x{INIT_LR_FRAC:g}, linear rampdown to 0 like chat_rl): "
       + " | ".join(f"{k} {v:.3g}" for k, v in _eff.items()))

cast_model_bf16(model)
model.eval()

engine = PrefillAllEngine(
    model, tokenizer,
    kv_pool_gb=KV_POOL_GB, max_seqs=MAX_SEQS, max_tokens=MAX_TOKENS,
    max_prompt_len=max_prompt, macro_n=MACRO_N, buckets=BUCKETS,
    prefill_t=PREFILL_T, prefill_seqs=PREFILL_SEQS,
    temperature=TEMPERATURE, top_p=1.0, top_k=TOP_K,
    stop_detect=False, stop_strings=(),
    compile_decode=COMPILE, compile_prefill=PREFILL_COMPILE,
    prefill_fullgraph=PREFILL_FULLGRAPH,
    extra_compile_slots=len(TRAIN_BUCKETS) + 8,
    device_index=local_rank if ddp else 0)
build_s = time.perf_counter() - t
print0(f"  build {build_s:.0f}s | pool {engine.pool.num_blocks} blocks "
       f"(~{engine.ceiling_rows} concurrent rollouts, need {ppr_rank * K_DRAWS}/round) | "
       f"decode buckets {engine.buckets}", flush=True)

t = time.perf_counter()
engine.capture(warm_context=train_prompts[shard[0]][:-1])

# The pool coexists with the train peak by construction, so warmup doubles as
# the capacity gate (an OOM here = pool too big for this card, shrink KV_POOL_GB).
_pool_gb = (engine.pool.k_buf.size + engine.pool.v_buf.size) / 2 ** 30
print0(f"  [vmm] pool permanent ({_pool_gb:.1f} GB mapped through warmup + rounds)",
       flush=True)

# -----------------------------------------------------------------------------
# §3. Trainer — group advantage + REINFORCE over compiled fixed-shape packs
#      (forward = nanochat's own GPT.forward)
# -----------------------------------------------------------------------------
ADV_EPS = 1e-6

def group_advantages(rewards: np.ndarray) -> np.ndarray | None:
    """chat_rl's advantage: plain (r - mean) over one problem's rewards; None
    for an all-equal group (all-correct / all-incorrect — zero advantage
    everywhere, hence zero gradient, so the docs are skipped outright)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2 or r.std() < ADV_EPS:
        return None
    return r - r.mean()


def reinforce_forward_loss(model, input_ids, cu_seqlens, targets, comp_mask, adv_tok):
    """Compile target (fullgraph, static shapes): nanochat packed forward ->
    Σ -A·logπ over completion tokens. Returns (loss_sum, n_loss_tokens); only
    loss_sum carries grad."""
    logits = model(input_ids, targets=None, cu_seqlens=cu_seqlens)  # (1, T, V) fp32 softcapped
    logp = -F.cross_entropy(logits[0], targets, reduction="none")
    comp = comp_mask.bool()
    loss_sum = (-(adv_tok * logp) * comp).sum()
    return loss_sum, comp.sum()


# TRAIN warmup: compile fwd+bwd per bucket on a dummy pack (weights untouched).
TRAIN_FN = (torch.compile(reinforce_forward_loss, fullgraph=True, dynamic=False)
            if COMPILE_TRAIN else reinforce_forward_loss)
for tb in TRAIN_BUCKETS:
    _t = time.perf_counter()
    seg = min(SEQ_CAP, tb)
    dummy = [([(i % 999) + 1 for i in range(32)], [(i % 999) + 1 for i in range(seg - 32)], 1.0)
             for _ in range(tb // seg)]
    packs, _ = build_reinforce_packs(dummy, buckets=list(TRAIN_BUCKETS),
                                     max_num_docs=MAX_NUM_DOCS, pad_id=PAD_ID,
                                     max_doc_len=SEQ_CAP)
    pk = packs[0]
    loss_sum, _ = TRAIN_FN(model, pk.input_ids, pk.cu_seqlens, pk.targets,
                           pk.comp_mask, pk.adv_tok)
    loss_sum.backward()
    model.zero_grad(set_to_none=True)
    del packs, pk, loss_sum
    torch.cuda.synchronize()
    print0(f"    train bucket {tb} ({'compiled' if COMPILE_TRAIN else 'eager'} fwd+bwd): "
           f"{time.perf_counter() - _t:5.1f}s", flush=True)
warm_s = time.perf_counter() - t
print0(f"  capture+compile+warmup {warm_s:.0f}s | peak mem "
       f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB", flush=True)


# -----------------------------------------------------------------------------
# §4. Grading + trainer step
# -----------------------------------------------------------------------------
def grade_rows(rows) -> list[float]:
    """GSM8K regex reward, inline (microseconds per row — no fork pool needed)."""
    return [train_task.reward(train_convs[r["meta"]], r["completion_text"]) for r in rows]


def train_step(groups: list[dict]) -> dict:
    """One REINFORCE optimizer step over the round's problem groups, computing
    the gradient chat_rl computes. Its normalizer is per-PASS: each group's K
    rollouts train in chunks of DEVICE_BATCH_SIZE, and every pass's loss is
    sum(-A*logp) / (that pass's completion-token count x num_passes x
    examples_per_rank). That is linear in the per-token advantage, so it folds
    into a per-doc advantage scale here — the packed forward then just sums."""
    _t0 = time.perf_counter()
    docs = []
    n_groups_used = n_excluded = 0
    n_sat = n_dead = 0
    for g in groups:
        adv = group_advantages(np.asarray(g["rewards"], dtype=np.float64))
        if adv is None:                               # zero-signal group: no gradient
            if np.mean(g["rewards"]) >= 1.0:
                n_sat += 1                            # every rollout correct
            elif np.mean(g["rewards"]) <= 0.0:
                n_dead += 1                           # every rollout wrong
            continue
        n_groups_used += 1
        bodies = []
        for comp in g["completions"]:
            comp = list(comp)
            if comp and comp[-1] in (ASSISTANT_END, BOS):
                comp = comp[:-1]  # terminal token is never trained (chat_rl strips it)
            bodies.append(comp)
        # chat_rl's pass partition, in rollout order. (Which rollouts share a
        # pass is arbitrary in both scripts — samples are exchangeable — but
        # the partition SHAPE and per-pass token counts are what set the scale.)
        n_pass = len(bodies) // DEVICE_BATCH_SIZE
        for p0 in range(0, len(bodies), DEVICE_BATCH_SIZE):
            chunk = range(p0, p0 + DEVICE_BATCH_SIZE)
            num_valid = max(1, sum(len(bodies[k]) for k in chunk))  # chat_rl clamps min=1
            scale = 1.0 / (num_valid * n_pass * ppr_rank)
            for k in chunk:
                if not bodies[k]:
                    n_excluded += 1
                    continue
                docs.append((g["prompt_ids"], bodies[k], float(adv[k]) * scale))
    total_tokens = 0
    total_loss = 0.0
    n_packs = 0
    pstats = None
    _t_build = _t_fwd = 0.0
    if docs:
        with record_function("train/build-packs"):
            packs, pstats = build_reinforce_packs(
                docs, buckets=list(TRAIN_BUCKETS), max_num_docs=MAX_NUM_DOCS,
                pad_id=PAD_ID, max_doc_len=SEQ_CAP)
        n_packs = pstats["n_packs"]
        _t_build = time.perf_counter() - _t0
        _pk_verbose = os.environ.get("TRAIN_TIMING_VERBOSE") == "1"
        _pk_t = time.perf_counter()
        with record_function("train/fwd+bwd"):
            for pk in packs:
                loss_sum, n_tok = TRAIN_FN(
                    model, pk.input_ids, pk.cu_seqlens, pk.targets, pk.comp_mask,
                    pk.adv_tok)
                nt = int(n_tok.item())
                if nt > 0:
                    loss_sum.backward()               # unnormalized; accumulates
                    total_loss += float(loss_sum.detach())
                    total_tokens += nt
                del loss_sum
                if _pk_verbose:
                    _now = time.perf_counter()
                    print0(f"      pack {n_packs}b{pk.input_ids.numel()}: {_now - _pk_t:.3f}s", flush=True)
                    _pk_t = _now
        _t_fwd = time.perf_counter() - _t0 - _t_build
    # The normalization already rode in on the per-doc advantage scale, and the
    # optimizer AVG-reduces per-rank grads exactly as the baseline relies on.
    # chat_rl steps UNconditionally every round (an all-zero-grad step still
    # moves Muon via momentum), so we do too — materialize zero grads for any
    # params that saw no docs so the (sharded) collectives stay well-formed.
    gnorm = 0.0
    with record_function("train/opt"):
        for prm in model.parameters():
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)
        if not ddp:  # grad-norm telemetry only — no clipping (chat_rl has none)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    # approximate wall split (host clocks; the per-pack .item() syncs make t_fwd
    # honest, and the trailing sync charges the drain + optimizer to t_opt)
    _t_opt = time.perf_counter() - _t0 - _t_build - _t_fwd
    return dict(t_build=_t_build, t_fwd=_t_fwd, t_opt=_t_opt,
                n_groups_used=n_groups_used, n_groups_total=len(groups),
                n_groups_sat=n_sat, n_groups_dead=n_dead,
                n_docs=len(docs), n_excluded=n_excluded, n_packs=n_packs,
                pstats=pstats, n_loss_tokens=total_tokens,
                loss_total=total_loss,  # == chat_rl's summed per-pass losses for the step
                grad_norm=gnorm)


# -----------------------------------------------------------------------------
# §5. Rounds
# -----------------------------------------------------------------------------
# Device-memory telemetry is SAMPLED, not per-round: cudaMemGetInfo measured
# ~125 ms a call against this process (trace_prof4) — ~1.4% of a 9 s round, for a
# number that barely moves, since the KV pool is VMM-mapped once and never
# unmapped and the torch allocator reaches steady state within a few rounds. It
# can NOT be swapped for the free torch.cuda.max_memory_reserved(): that sees
# only the caching allocator, and the 28 GB pool lives outside it. So sample
# every MEM_EVERY rounds and carry the last reading (0 = only round 0).
MEM_EVERY = _env_int("MEM_EVERY", 50)
_mem_last = 0.0


def _device_mem_gb(rnd: int) -> float:
    global _mem_last
    if rnd == 0 or (MEM_EVERY and rnd % MEM_EVERY == 0):
        free, total = torch.cuda.mem_get_info()
        _mem_last = round((total - free) / 2 ** 30, 1)
    return _mem_last


METRIC_COLS = ["round", "n_rollouts", "n_correct", "solve_rate", "n_truncated",
               "n_eos", "gen_s", "gen_tok", "gen_tok_per_s", "rolls_per_min",
               "peak_blocks", "train_s", "n_groups_used", "n_groups_sat",
               "n_groups_dead", "n_docs", "n_loss_tokens",
               "train_tok_per_s", "loss_total",
               "grad_norm", "lrm", "wnorm", "mem_gb", "round_s"]
metrics_path = HERE / f"metrics_{TAG}.csv"
mf = open(metrics_path, "w", newline="") if master else None
mw = None
if master:
    mw = csv.DictWriter(mf, fieldnames=METRIC_COLS)
    mw.writeheader()

RUN_DIR = Path(os.environ.get("RUN_DIR", str(Path.home() / ".cache" / "nanochat" / "fastrl_runs"))) / TAG
if master:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

curve: list[dict] = []
run_error = None
run_t0 = time.perf_counter()


def save_ckpt(step: int) -> None:
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "chatrl_checkpoints", OUT_TAG)
    opt_data = optimizer.state_dict() if SAVE_OPT else None
    # user_config records the LRs this run actually trained at, for provenance.
    meta_data = {"model_config": model.config.__dict__,
                 "user_config": dict(source=SOURCE, model_tag=MODEL_TAG,
                                     model_step=MODEL_STEP,
                                     unembedding_lr=UNEMBEDDING_LR,
                                     embedding_lr=EMBEDDING_LR,
                                     matrix_lr=MATRIX_LR, scalar_lr=SCALAR_LR,
                                     init_lr_frac=INIT_LR_FRAC)}
    if master or opt_data is not None:
        save_checkpoint(ckpt_dir, step, model.state_dict() if master else None,
                        opt_data, meta_data, rank=rank)
    if master:
        print(f"  saved checkpoint -> {ckpt_dir} (step {step})", flush=True)


profiler = None
if PROFILE and master:
    profiler = torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=True,
        schedule=torch.profiler.schedule(wait=PROF_WAIT, warmup=1,
                                         active=PROF_ACTIVE, repeat=1))
    profiler.__enter__()

try:
    for rnd in range(num_rounds):
        r_t0 = time.perf_counter()

        # -- generation ------------------------------------------------------
        idxs = FIXED_PROBLEMS if FIXED_PROBLEMS is not None else round_schedule[rnd]
        specs = [(i, train_prompts[i], K_DRAWS, MAX_TOKENS) for i in idxs]
        with record_function("round/gen"):
            rows, gstats = engine.run_round(engine.make_nodes(specs), rnd)

        # -- grade -----------------------------------------------------------
        with record_function("round/grade+group"):
            rewards = grade_rows(rows)
            by_pid: dict[int, list[int]] = {}
            for i, r in enumerate(rows):
                by_pid.setdefault(r["meta"], []).append(i)
            groups = [dict(
                prompt_ids=train_prompts[pid],
                completions=[rows[i]["completion_token_ids"] for i in idl],
                rewards=[rewards[i] for i in idl],
            ) for pid, idl in by_pid.items()]

        # -- train -----------------------------------------------------------
        lrm = 1.0 - rnd / num_rounds  # chat_rl's linear rampdown to zero
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        _t = time.perf_counter()
        with record_function("round/train"):
            tstats = train_step(groups)
        train_s = time.perf_counter() - _t

        # -- telemetry -------------------------------------------------------
        n_correct = sum(r == 1.0 for r in rewards)
        agg = torch.tensor([n_correct, len(rows)], dtype=torch.float, device=device)
        if ddp:
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        solve_rate = float(agg[0] / agg[1])
        with torch.no_grad():
            wnorm = float(torch.sqrt(sum((p.float() ** 2).sum() for p in model.parameters())))
        row = dict(
            round=rnd, n_rollouts=int(agg[1]), n_correct=int(agg[0]),
            solve_rate=round(solve_rate, 4),
            n_truncated=sum(r["terminal"] == "truncated" for r in rows),
            n_eos=sum(r["terminal"] == "emitted_eos" for r in rows),
            gen_s=round(gstats["gen_s"], 1), gen_tok=gstats["gen_tok"],
            gen_tok_per_s=round(gstats["gen_tok"] / gstats["gen_s"], 1),
            rolls_per_min=round(len(rows) / gstats["gen_s"] * 60, 1),
            peak_blocks=gstats.get("peak_blocks", 0),
            train_s=round(train_s, 1),
            n_groups_used=tstats["n_groups_used"],
            n_groups_sat=tstats["n_groups_sat"], n_groups_dead=tstats["n_groups_dead"],
            n_docs=tstats["n_docs"],
            n_loss_tokens=tstats["n_loss_tokens"],
            train_tok_per_s=round(tstats["n_loss_tokens"] / train_s, 1) if train_s else 0.0,
            loss_total=round(tstats["loss_total"], 6),
            grad_norm=round(tstats["grad_norm"], 6), lrm=round(lrm, 4),
            wnorm=round(wnorm, 2),
            mem_gb=_device_mem_gb(rnd),
            round_s=round(time.perf_counter() - r_t0, 1))
        curve.append(row)
        if master:
            mw.writerow(row)
            mf.flush()
            if SAVE_ROLLOUTS:
                import pandas as pd
                pd.DataFrame(rows).assign(reward=rewards).to_parquet(
                    RUN_DIR / f"rollouts_round_{rnd:04d}.parquet", index=False)
        # padding-waste telemetry (all host-side ints — no GPU sync):
        # prefill packing = real packed tokens / (replays x PREFILL_T); train pad%
        # = pad tail / sealed pack capacity. Keys for tuning PREFILL_SEQS and
        # TRAIN_BUCKETS/MAX_NUM_DOCS respectively.
        pf_pack = (100.0 * gstats["prefill_tok"] / (gstats["replays"] * PREFILL_T)
                   if gstats.get("replays") else 0.0)
        tr_pad = (100.0 * tstats["pstats"]["pad_tokens"] / max(1, tstats["pstats"]["cap_tokens"])
                  if tstats.get("pstats") else 0.0)
        print0(f"  [round {rnd:3d}] gen   {gstats['gen_s']:5.1f}s ({row['gen_tok_per_s']:>7,.0f} tok/s) | "
               f"solve {int(agg[0]):3d}/{int(agg[1])} ({100*solve_rate:5.1f}%) | "
               f"eos {row['n_eos']:3d} trunc {row['n_truncated']:2d} | "
               f"prefill {gstats.get('replays', 0)}r {pf_pack:.0f}% | "
               f"kv peak {gstats.get('peak_blocks', 0)}/{engine.pool.num_blocks}", flush=True)
        print0(f"              train {train_s:5.1f}s ({row['train_tok_per_s']:>7,.0f} tok/s) | "
               f"{tstats['n_loss_tokens']:,} loss-tok | "
               f"{tstats.get('n_packs', 0)} packs pad {tr_pad:.0f}% | "
               f"grp {tstats['n_groups_used']}/{tstats['n_groups_total']} "
               f"(sat {tstats['n_groups_sat']} dead {tstats['n_groups_dead']}) | "
               f"gnorm {tstats['grad_norm']:.3f} | lrm {lrm:.3f} | "
               f"build+fwd+opt {tstats['t_build']:.2f}+{tstats['t_fwd']:.2f}+{tstats['t_opt']:.2f}",
               flush=True)

        # per-epoch rollup: solve over the whole pass + avg round wall. Fires at
        # each pass boundary and, if ROUNDS_CAP cut the run mid-pass, at the end
        # (the partial pass is labeled by its round count).
        if (rnd + 1) % rounds_per_epoch == 0 or rnd + 1 == num_rounds:
            ep = curve[-((rnd % rounds_per_epoch) + 1):]
            ep_cor = sum(c["n_correct"] for c in ep)
            ep_roll = sum(c["n_rollouts"] for c in ep)
            print0(f"  == epoch {rnd // rounds_per_epoch + 1:2d}/{EPOCHS} | "
                   f"solve {ep_cor:,}/{ep_roll:,} ({100 * ep_cor / ep_roll:5.2f}%) | "
                   f"avg round {sum(c['round_s'] for c in ep) / len(ep):.1f}s "
                   f"(gen {sum(c['gen_s'] for c in ep) / len(ep):.1f} + "
                   f"train {sum(c['train_s'] for c in ep) / len(ep):.1f}) | "
                   f"{len(ep)} rounds ==", flush=True)

        if SAVE_EVERY and rnd > 0 and rnd % SAVE_EVERY == 0:
            save_ckpt(rnd)
        if profiler is not None:
            profiler.step()
except BaseException as e:
    run_error = f"{type(e).__name__}: {e}"
    raise
finally:
    if profiler is not None:
        profiler.__exit__(None, None, None)
        trace_path = HERE / f"trace_{TAG}.json.gz"
        try:
            profiler.export_chrome_trace(str(trace_path))
            print(f"\n  chrome trace -> {trace_path} (load in ui.perfetto.dev; labels: "
                  f"round/gen, round/grade+group, round/train > build-packs / fwd+bwd / "
                  f"clip+opt)", flush=True)
            print(profiler.key_averages().table(sort_by="self_cuda_time_total",
                                                row_limit=25), flush=True)
        except Exception as pe:
            print(f"  !! trace export failed: {pe}", flush=True)
    if master and mf:
        mf.close()
    total_s = time.perf_counter() - run_t0
    if curve:
        save_ckpt(len(curve))

    # ------------------------------------------------------------------
    # §6. Results — summary + save (+ optional HF push)
    # ------------------------------------------------------------------
    checklist = {
        "decode_body_compiled": bool(COMPILE),
        "decode_cuda_graphs_captured": len(engine.gd.graphs),
        "prefill_varlen_packed_graph": engine.pfg is not None and engine.pfg.graph is not None,
        "prefill_body_compiled": bool(PREFILL_COMPILE),
        "prefix_sharing_active": True,
        "train_forward_compiled": bool(COMPILE_TRAIN),
        "train_static_buckets": list(TRAIN_BUCKETS),
        "fullparam_inplace_no_recapture": True,
        "same_process_gen_train": True,
        "bf16_params_fp32_opt_state": True,
        "kv_pool_permanent_gb": round((engine.pool.k_buf.size + engine.pool.v_buf.size) / 2 ** 30, 1),
        "chat_rl_aligned_algorithm": True,
        "tool_use": False,
    }
    result = dict(
        tag=TAG, k=K_DRAWS, problems_per_round=PPR, rounds_run=len(curve),
        budget=MAX_TOKENS, world_size=world_size, source=SOURCE,
        temperature=TEMPERATURE, top_k=TOP_K, init_lr_frac=INIT_LR_FRAC,
        device_batch_size=DEVICE_BATCH_SIZE,
        raw_lrs=dict(unembedding=UNEMBEDDING_LR, embedding=EMBEDDING_LR,
                     matrix=MATRIX_LR, scalar=SCALAR_LR),
        error=run_error,
        solve_rate_first=(curve[0]["solve_rate"] if curve else None),
        solve_rate_last=(curve[-1]["solve_rate"] if curve else None),
        solve_rate_max=(max(c["solve_rate"] for c in curve) if curve else None),
        gen_tok_per_s_med=(sorted(c["gen_tok_per_s"] for c in curve)[len(curve) // 2]
                           if curve else None),
        train_s_med=(sorted(c["train_s"] for c in curve)[len(curve) // 2] if curve else None),
        round_s_med=(sorted(c["round_s"] for c in curve)[len(curve) // 2] if curve else None),
        total_s=round(total_s, 1), build_s=round(build_s, 1), warm_s=round(warm_s, 1),
        peak_mem_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 1),
        kv_pool_gb=KV_POOL_GB, buckets=",".join(map(str, engine.buckets)),
        checklist=checklist, run_dir=str(RUN_DIR))
    if master:
        (HERE / f"result_{TAG}.json").write_text(json.dumps(result, indent=1))
        print(f"\n== chat_rl_fast [{TAG}] ==", flush=True)
        print(f"  rounds {len(curve)} | solve {result['solve_rate_first']} -> "
              f"{result['solve_rate_last']} (max {result['solve_rate_max']}) | "
              f"total {total_s / 60:.1f} min | peak mem {result['peak_mem_gb']} GB", flush=True)
        print("  checklist: " + json.dumps(checklist), flush=True)
        try:
            from tabulate import tabulate
            md = tabulate([[k, v] for k, v in result.items() if k != "checklist"],
                          headers=["metric", "value"], tablefmt="github")
            cmd = tabulate([[c["round"], c["n_correct"], c["solve_rate"], c["gen_s"],
                             c["train_s"], c["grad_norm"], c["wnorm"]] for c in curve],
                           headers=["round", "correct", "rate", "gen_s", "train_s",
                                    "gnorm", "|w|"], tablefmt="github")
            (HERE / f"result_{TAG}.md").write_text(
                f"# chat_rl_fast `{TAG}`\n\n{md}\n\nChecklist: `{json.dumps(checklist)}`\n\n"
                f"## Round curve\n\n{cmd}\n")
        except ImportError:
            pass
        import shutil
        shutil.copy2(metrics_path, RUN_DIR / "metrics.csv")
        (RUN_DIR / f"result_{TAG}.json").write_text(json.dumps(result, indent=1))
        if PUSH:
            try:
                from huggingface_hub import HfApi
                HfApi().upload_folder(folder_path=str(RUN_DIR), repo_id=MODEL_REPO,
                                      repo_type="model", path_in_repo=f"runs/fastrl/{TAG}",
                                      commit_message=f"chat_rl_fast {TAG}"
                                      + (" (errored)" if run_error else ""))
                print(f"  pushed: runs/fastrl/{TAG} (repo {MODEL_REPO})", flush=True)
            except Exception as e:
                print(f"  !! run push FAILED ({e}) — push {RUN_DIR} before releasing the box",
                      flush=True)
        print(f"  results -> result_{TAG}.json / result_{TAG}.md / {metrics_path.name}",
              flush=True)

compute_cleanup()
