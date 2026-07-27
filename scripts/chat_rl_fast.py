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
regex reward, (3) take ONE branch-masked REINFORCE (DAPO token-mean) optimizer
step on ALL parameters. Differences from stock chat_rl.py, per the locked plan:
  * masked-token branch REINFORCE: loss only on the policy's OWN T=1.0 nucleus>1
    positions; group-normalized advantage (r-mean)/std (ADV_STD=0 -> stock r-mean);
    truncated-incorrect excluded from the loss but kept in the baseline.
  * sampler: temp 0.6 / top-k 512 / top-p 0.95, in-graph Gumbel-max.
  * NO tool use: the calculator force-injection of the stock Engine is dropped —
    the policy must emit <|output_start|>...<|output_end|> contents itself.
  * training forward = nanochat's GPT.forward on packed varlen buckets,
    torch.compile'd fullgraph with static shapes (stock runs it eager).
  * terminal set {<|assistant_end|>, <|bos|>}; completions DO train on the
    terminal token (stock never reinforces emitting it).

1 GPU:   python -m scripts.chat_rl_fast <tag>
8 GPUs:  torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_fast <tag>

Config via env (speedrun runner convention), see §0 below.
"""

# -----------------------------------------------------------------------------
# §0. Config
# -----------------------------------------------------------------------------
import csv
import json
import math
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
from nanochat.checkpoint_manager import save_checkpoint, load_model, find_last_step
from nanochat.gpt import cast_model_bf16
from nanochat.optim import Fp32MuonAdamW, Fp32DistMuonAdamW
from nanochat.schedules import Ramp, AdamWGroup, MuonGroup, build_param_groups
from nanochat.fast_engine import PrefillAllEngine

from tasks.gsm8k import GSM8K, extract_answer, GSM_RE

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_opt_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _env_flag(name, default):
    return os.environ.get(name, str(default)) == "1"


# Rollouts / rounds
K_DRAWS      = _env_int("K", 16)                  # rollouts per problem per round
PPR          = _env_int("PROBLEMS_PER_ROUND", 16) # global, across ranks
EPOCHS       = _env_int("EPOCHS", 1)
ROUNDS_CAP   = _env_int("ROUNDS", 0)              # 0 = full EPOCHS horizon
MAX_TOKENS   = _env_int("MAX_TOKENS", 256)
# Sampler (speedrun house sampler)
TEMPERATURE  = _env_float("TEMPERATURE", 0.6)
TOP_P        = _env_float("TOP_P", 0.95)
TOP_K        = _env_int("TOP_K", 512)
# Trainer
ADV_STD      = _env_flag("ADV_STD", 1)            # 0 -> stock (r - mean)
# TRAIN_TERMINAL=1 (speedrun): the emitted <|assistant_end|> is a trained token.
# 0 (stock chat_rl): exclude it from the loss — with negative advantages on
# failing problems, training the terminal actively suppresses termination and
# produces a truncation/rambling spiral (observed in run ep1).
TRAIN_TERMINAL = _env_flag("TRAIN_TERMINAL", 1)
# CLIP_ANSWER=1: answer-then-ramble guard. A CORRECT completion that keeps
# generating past its `#### <answer>` is cut right after the answer and
# terminated with <|assistant_end|> BEFORE becoming a training doc, so positive
# advantage reinforces answer->stop rather than the ramble. Without it the loop
# amplifies: truncated-correct rollouts train the ramble at positive advantage
# (only truncated-INCORRECT are excluded), and their tails eventually burst the
# KV pool — how pool30_lr05 died at r109. 0 = pre-2026-07-26 behavior
# (div4/div5/pool30/pool30_lr05). Incorrect rollouts are never clipped.
CLIP_ANSWER    = _env_flag("CLIP_ANSWER", 1)
TRAIN_BRANCH_TEMP  = _env_float("TRAIN_BRANCH_TEMP", 1.0)
TRAIN_BRANCH_TOP_P = _env_float("TRAIN_BRANCH_TOP_P", 0.95)
GRAD_CLIP    = _env_float("GRAD_CLIP", 1.0)       # exact on 1 GPU; skipped under DDP
# LR args are RAW pretraining-style values (§2 applies the
# usual 1/sqrt(dmodel) scale to the AdamW groups). Unset (the default) = inherit
# the pretraining run's raw args through the checkpoint chain (§2); set to pin
# an absolute value (how the pre-2026-07-25 launchers fixed a flat 3e-5 — which
# for Muon is ~1000x colder than pretraining's effective matrix LR, and trained
# ~nothing: see agent-ops rl-engine-baseline pool30).
UNEMBEDDING_LR = _env_opt_float("UNEMBEDDING_LR")
EMBEDDING_LR   = _env_opt_float("EMBEDDING_LR")
MATRIX_LR      = _env_opt_float("MATRIX_LR")
SCALAR_LR      = _env_opt_float("SCALAR_LR")      # resid/x0/smear scalar groups
# FREEZE_SCALARS=1: zero the LR on ALL per-layer scalar groups (resid_lambdas,
# x0_lambdas, smear_gate/smear_lambda/backout_lambda). Their pretraining
# trajectories are smooth and deliberate — RL should not touch them (Chris).
FREEZE_SCALARS = _env_flag("FREEZE_SCALARS", 0)
WEIGHT_DECAY   = _env_float("WEIGHT_DECAY", 0.0)
# The one RL temperature knob: every group trains at INIT_LR_FRAC x its
# pretraining LR (chat_rl's design, keeping its 0.05). LR_SCHEDULE "flat" holds
# it there (the RL-paper norm); "linear" is chat_rl's rampdown to zero.
INIT_LR_FRAC   = _env_float("INIT_LR_FRAC", 0.05)
LR_SCHEDULE    = os.environ.get("LR_SCHEDULE", "flat")
assert LR_SCHEDULE in ("flat", "linear"), f"bad LR_SCHEDULE {LR_SCHEDULE!r}"
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
STOP_DETECT  = _env_flag("STOP_DETECT", 1)
STOP = (json.loads(os.environ["STOP"]) if os.environ.get("STOP")
        else ["\nQuestion:", " Question:", "\nProblem:"])
# Eval / checkpoint
EVAL_EVERY    = _env_int("EVAL_EVERY", 60)        # 0 = off
EVAL_EXAMPLES = _env_int("EVAL_EXAMPLES", 400)
EVAL_K        = _env_int("EVAL_K", 8)
EVAL_TEMP     = _env_float("EVAL_TEMP", 1.0)
EVAL_MAX_TOKENS = _env_int("EVAL_MAX_TOKENS", MAX_TOKENS)
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

# PrefillAllEngine is train-only: an eval round (EVAL_EXAMPLES x EVAL_K against the
# full test split) is far more concurrent rows than the pool can admit in one wave.
assert EVAL_EVERY == 0, (
    "chat_rl_fast is train-only — its single-wave round can't admit an eval "
    "round. Set EVAL_EVERY=0 and score checkpoints offline with agent-ops "
    "eval_trajectory.py.")

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
val_task = GSM8K(subset="main", split="test")

print0("rendering prompts ...", flush=True)
train_convs = [train_task[i] for i in range(len(train_task))]
train_prompts = [tokenizer.render_for_completion(c) for c in train_convs]
n_eval = min(EVAL_EXAMPLES, len(val_task)) if EVAL_EVERY else 0
val_convs = [val_task[i] for i in range(n_eval)]
val_prompts = [tokenizer.render_for_completion(c) for c in val_convs]
max_prompt = max(max(len(p) for p in train_prompts),
                 max((len(p) for p in val_prompts), default=0))
assert max_prompt + 1 + max(MAX_TOKENS, EVAL_MAX_TOKENS) <= SEQ_CAP, \
    "prompt+budget exceeds model context"
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
       f"| train buckets {TRAIN_BUCKETS} | stop-detect "
       f"{'ON ' + str(STOP) if STOP_DETECT else 'OFF'}", flush=True)

# -----------------------------------------------------------------------------
# §2. Optimizer (fp32 master/state) -> bf16 cast -> engine + graph capture
# -----------------------------------------------------------------------------
# RL trains at INIT_LR_FRAC x the pretraining LRs: raw args inherited through
# the checkpoint chain, with §2 applying the same
# 1/sqrt(dmodel) AdamW scale pretraining used. base_train's sqrt(B/B_ref)
# batch_lr_scale is NOT applied — that rule is for token batches; an RL round is
# ~25k branch tokens, and neither chat_sft nor chat_rl carries it either.
def _pretrain_user_config():
    """The user_config of the PRETRAINING run behind the loaded checkpoint.
    An SFT meta snapshots user_config before its inheritance step resolves the
    LR args (they stay null), so walk one hop back to the base checkpoint it
    trained from."""
    uc = meta.get("user_config", {})
    if uc.get("matrix_lr") is not None:
        return uc, "checkpoint user_config"
    base_tag = uc.get("model_tag")
    if base_tag:
        ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", base_tag)
        step = uc.get("model_step") or find_last_step(ckpt_dir)
        with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json")) as f:
            base_uc = json.load(f).get("user_config", {})
        if base_uc.get("matrix_lr") is not None:
            return base_uc, f"base checkpoint {base_tag} step {step}"
    return {}, "none"


_pt_uc, _lr_source = _pretrain_user_config()
_LR_FALLBACKS = dict(unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02, scalar_lr=0.5)


def _resolve_lr(env_val, key):
    if env_val is not None:
        return env_val  # pinned absolute by env (legacy launchers)
    if _pt_uc.get(key) is not None:
        return float(_pt_uc[key])
    print0(f"WARNING: {key} not recorded in checkpoint chain — "
           f"falling back to base_train default {_LR_FALLBACKS[key]}")
    return _LR_FALLBACKS[key]


unembedding_lr = _resolve_lr(UNEMBEDDING_LR, "unembedding_lr")
embedding_lr   = _resolve_lr(EMBEDDING_LR, "embedding_lr")
matrix_lr      = _resolve_lr(MATRIX_LR, "matrix_lr")
scalar_lr      = _resolve_lr(SCALAR_LR, "scalar_lr")
print0(f"[{TAG}] raw LRs (source: {_lr_source}; env pins: "
       f"{[k for k, v in dict(UNEMBEDDING_LR=UNEMBEDDING_LR, EMBEDDING_LR=EMBEDDING_LR, MATRIX_LR=MATRIX_LR, SCALAR_LR=SCALAR_LR).items() if v is not None] or 'none'}): "
       f"unembedding {unembedding_lr:g} | embedding {embedding_lr:g} | "
       f"matrix {matrix_lr:g} | scalar {scalar_lr:g}")

# Parameter groups + the whole run's hyperparameter schedules, pre-computed into
# per-step update coefficients (nanochat/schedules.py). The 1/sqrt(dmodel) AdamW
# scale and the INIT_LR_FRAC RL temperature both fold into the tables here, so
# nothing rescales the groups afterwards and the round loop sets nothing.
dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5
adamw_lr_scale = dmodel_lr_scale * INIT_LR_FRAC
# FREEZE_SCALARS: the per-layer scalar groups simply train at LR 0.
scalar_frac = 0.0 if FREEZE_SCALARS else INIT_LR_FRAC
# "flat" holds the LR where it starts (the RL-paper norm); "linear" is chat_rl's
# rampdown to zero.
lrm = Ramp(peak=1.0) if LR_SCHEDULE == "flat" else Ramp(peak=1.0, end=0.0, cooldown_frac=1.0)

_eff = dict(
    unembed=unembedding_lr * adamw_lr_scale,
    embed=embedding_lr * adamw_lr_scale,
    value_emb=embedding_lr * adamw_lr_scale * 0.5,
    resid=scalar_lr * scalar_frac * 0.01,
    x0=scalar_lr * scalar_frac,
    smear=0.2 * scalar_frac,
    muon=matrix_lr * INIT_LR_FRAC,
)
pl = model.named_parameter_lists()
specs = [
    AdamWGroup(pl['lm_head'],      lr=lrm * _eff['unembed'],   betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
    AdamWGroup(pl['wte'],          lr=lrm * _eff['embed'],     betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
    AdamWGroup(pl['value_embeds'], lr=lrm * _eff['value_emb'], betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
    AdamWGroup(pl['resid'],        lr=lrm * _eff['resid'],     betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.05),
    AdamWGroup(pl['x0'],           lr=lrm * _eff['x0'],        betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    AdamWGroup(pl['smear'],        lr=lrm * _eff['smear'],     betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.0),
]
for shape in sorted({p.shape for p in pl['matrix']}):
    specs.append(MuonGroup([p for p in pl['matrix'] if p.shape == shape],
                           lr=lrm * _eff['muon'], momentum=0.95, beta2=0.9,
                           weight_decay=WEIGHT_DECAY, ns_steps=5))

# Snapshot fp32 masters from the checkpoint weights BEFORE the bf16 cast.
Factory = Fp32DistMuonAdamW if ddp else Fp32MuonAdamW
optimizer = Factory(build_param_groups(specs, num_steps=num_rounds))
lrm_table = lrm.materialize(num_rounds) # host-side copy, for logging only (no device sync)
if FREEZE_SCALARS:
    print0("FREEZE_SCALARS: resid/x0/smear/backout groups at lr 0")
print0(f"[{TAG}] effective LRs (x{INIT_LR_FRAC:g} of pretrain, {LR_SCHEDULE}): "
       + " | ".join(f"{k} {v:.3g}" for k, v in _eff.items()))

cast_model_bf16(model)
model.eval()

engine = PrefillAllEngine(
    model, tokenizer,
    kv_pool_gb=KV_POOL_GB, max_seqs=MAX_SEQS, max_tokens=max(MAX_TOKENS, EVAL_MAX_TOKENS),
    max_prompt_len=max_prompt, macro_n=MACRO_N, buckets=BUCKETS,
    prefill_t=PREFILL_T, prefill_seqs=PREFILL_SEQS,
    temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
    stop_detect=STOP_DETECT, stop_strings=tuple(STOP),
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
# §3. Trainer — group advantage + branch-masked REINFORCE over compiled
#      fixed-shape packs (speedrun §3; forward = nanochat's own GPT.forward)
# -----------------------------------------------------------------------------
ADV_EPS = 1e-6

def group_advantages(rewards: np.ndarray, resolved: np.ndarray | None = None,
                     use_std: bool = True) -> np.ndarray | None:
    """(r - mean)/std over one problem's RESOLVED rewards; None for a std=0 group
    (all-correct / all-incorrect — no signal, skip). use_std=False gives stock
    nanochat's plain (r - mean) advantage (still skipping zero-signal groups)."""
    r = np.asarray(rewards, dtype=np.float64)
    res = (np.ones(len(r), dtype=bool) if resolved is None
           else np.asarray(resolved, dtype=bool))
    rr = r[res]
    if rr.size < 2:
        return None
    std = rr.std()
    if std < ADV_EPS:
        return None
    adv = np.zeros(len(r), dtype=np.float64)
    adv[res] = (rr - rr.mean()) / std if use_std else (rr - rr.mean())
    return adv


def reinforce_forward_loss(model, input_ids, cu_seqlens, targets, comp_mask, adv_tok,
                           branch_temperature: float, branch_top_p: float):
    """Compile target (fullgraph, static shapes): nanochat packed forward ->
    Σ -A·logπ over kept BRANCH tokens (the policy's own T=1.0 nucleus>1
    positions). Returns (loss_sum, n_loss_tokens, n_branch, n_comp); only
    loss_sum carries grad."""
    logits = model(input_ids, cu_seqlens)  # (T, V) fp32 softcapped
    logp = -F.cross_entropy(logits, targets, reduction="none")
    z = logits / branch_temperature
    log_pmax = z.amax(-1) - z.logsumexp(-1)
    branch = log_pmax <= math.log(branch_top_p)
    comp = comp_mask.bool()
    tok_mask = comp & branch
    loss_sum = (-(adv_tok * logp) * tok_mask).sum()
    return loss_sum, tok_mask.sum(), tok_mask.sum(), comp.sum()


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
    loss_sum, *_ = TRAIN_FN(model, pk.input_ids, pk.cu_seqlens, pk.targets,
                            pk.comp_mask, pk.adv_tok, TRAIN_BRANCH_TEMP, TRAIN_BRANCH_TOP_P)
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


def clip_post_answer(comp_ids: list[int], text: str) -> list[int]:
    """The CLIP_ANSWER surgery for one correct completion: token-granular cut at
    the first token whose decode covers the `#### <answer>` regex match end,
    plus <|assistant_end|>. `text` is the row's completion_text (trailing
    terminal token already excluded by the engine, so a clean answer-then-stop
    has nothing after the match). Returns comp_ids ITSELF when already clean —
    callers identity-check to see whether surgery happened. Token boundaries
    don't split, so the cut token may carry a merged trailing char."""
    m = GSM_RE.search(text)
    if m is None or not text[m.end():].strip():
        return comp_ids
    lo, hi = 1, len(comp_ids)      # smallest prefix whose decode covers the answer
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(comp_ids[:mid])) >= m.end():
            hi = mid
        else:
            lo = mid + 1
    return comp_ids[:lo] + [ASSISTANT_END]


def train_step(groups: list[dict]) -> dict:
    """One branch-masked REINFORCE optimizer step over the round's problem
    groups. Same advantages/exclusions/DAPO token-mean as the speedrun; no
    'unresolved' verdicts (regex reward always resolves)."""
    _t0 = time.perf_counter()
    docs = []
    n_groups_used = n_excluded = 0
    n_sat = n_dead = 0
    for g in groups:
        adv = group_advantages(np.asarray(g["rewards"], dtype=np.float64), use_std=ADV_STD)
        if adv is None:                               # zero-signal group: no gradient
            if np.mean(g["rewards"]) >= 1.0:
                n_sat += 1                            # every rollout correct
            elif np.mean(g["rewards"]) <= 0.0:
                n_dead += 1                           # every rollout wrong
            continue
        n_groups_used += 1
        for k, comp in enumerate(g["completions"]):
            # truncated-incorrect stays in the baseline but is excluded from loss
            if (g["truncated"][k] and g["rewards"][k] == 0) or not comp:
                n_excluded += 1
                continue
            comp = list(comp)
            if not TRAIN_TERMINAL and comp[-1] in (ASSISTANT_END, BOS):
                comp = comp[:-1]
                if not comp:
                    n_excluded += 1
                    continue
            docs.append((g["prompt_ids"], comp, float(adv[k])))
    total_tokens = total_branch = total_comp = 0
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
                loss_sum, n_tok, n_branch, n_comp = TRAIN_FN(
                    model, pk.input_ids, pk.cu_seqlens, pk.targets, pk.comp_mask,
                    pk.adv_tok, TRAIN_BRANCH_TEMP, TRAIN_BRANCH_TOP_P)
                total_comp += int(n_comp.item())
                total_branch += int(n_branch.item())
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
    # DAPO token-level mean across ALL ranks' loss tokens
    tok_t = torch.tensor(float(total_tokens), device=device)
    if ddp:
        dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
    global_tokens = float(tok_t.item())
    gnorm = 0.0
    stepped = False
    with record_function("train/clip+opt"):
        if global_tokens > 0:
            # A rank can have zero local docs while others train: materialize zero
            # grads so the (sharded) optimizer's collectives stay well-formed.
            for prm in model.parameters():
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
            inv = world_size / global_tokens  # optimizer AVG-reduces grads across ranks
            params = [p for p in model.parameters() if p.grad is not None]
            for prm in params:
                prm.grad.mul_(inv)
            if GRAD_CLIP > 0 and not ddp:  # exact clip; skipped under DDP (sharded reduce)
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP))
            optimizer.step()
            stepped = True
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    # approximate wall split (host clocks; the per-pack .item() syncs make t_fwd
    # honest, and the trailing sync charges the drain + optimizer to t_opt)
    _t_opt = time.perf_counter() - _t0 - _t_build - _t_fwd
    return dict(t_build=_t_build, t_fwd=_t_fwd, t_opt=_t_opt,
                n_groups_used=n_groups_used, n_groups_total=len(groups),
                n_groups_sat=n_sat, n_groups_dead=n_dead,
                n_docs=len(docs), n_excluded=n_excluded, n_packs=n_packs,
                pstats=pstats,
                n_loss_tokens=total_tokens, n_comp_tokens=total_comp,
                branch_frac=(total_branch / total_comp) if total_comp else 0.0,
                loss_token_mean=(total_loss / total_tokens) if total_tokens else 0.0,
                grad_norm=gnorm, stepped=stepped)


# -----------------------------------------------------------------------------
# §5. Eval — pass@k on the test split through the fast engine
# -----------------------------------------------------------------------------
def run_eval(rnd: int) -> dict:
    engine.set_sampling(temperature=EVAL_TEMP, top_p=TOP_P)
    specs = [(("eval", i), val_prompts[i], EVAL_K, EVAL_MAX_TOKENS)
             for i in range(rank, n_eval, world_size)]
    nodes = engine.make_nodes(specs)
    rows, gstats = engine.run_round(nodes, rnd)
    engine.set_sampling(temperature=TEMPERATURE, top_p=TOP_P)
    by_idx: dict[int, list[int]] = {}
    for r in rows:
        i = r["meta"][1]
        ok = val_task.evaluate(val_convs[i], r["completion_text"])
        by_idx.setdefault(i, []).append(ok)
    passk = torch.zeros(EVAL_K, device=device)
    for outcomes in by_idx.values():
        for k in range(1, EVAL_K + 1):
            passk[k - 1] += float(any(outcomes[:k]))
    n_rec = torch.tensor(len(by_idx), dtype=torch.long, device=device)
    # rollout-level eval diagnostics: truncation rate (hit the token budget) and
    # correct-formatting rate (emitted an extractable `#### n`, right or wrong) —
    # so a flat pass@k can be attributed to format/truncation vs actual wrongness.
    n_roll = torch.tensor(float(len(rows)), device=device)
    n_trunc = torch.tensor(float(sum(r["terminal"] == "truncated" for r in rows)), device=device)
    n_fmt = torch.tensor(float(sum(extract_answer(r["completion_text"]) is not None for r in rows)), device=device)
    if ddp:
        for _t in (n_rec, passk, n_roll, n_trunc, n_fmt):
            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
    passk = (passk / n_rec.item()).tolist()
    trunc_rate = (n_trunc / n_roll).item()
    fmt_rate = (n_fmt / n_roll).item()
    print0(f"  [eval r{rnd}] " + ", ".join(f"pass@{k+1}: {v:.4f}" for k, v in enumerate(passk))
           + f" | fmt {100*fmt_rate:.1f}% | trunc {100*trunc_rate:.1f}%"
           + f" | {gstats['gen_tok']:,} tok in {gstats['gen_s']:.1f}s "
           f"({gstats['gen_tok']/gstats['gen_s']:,.0f} tok/s)", flush=True)
    return {f"pass@{k+1}": round(v, 4) for k, v in enumerate(passk)} | {
        "round": rnd, "n": int(n_rec.item()),
        "fmt_rate": round(fmt_rate, 4), "trunc_rate": round(trunc_rate, 4),
        "gen_s": round(gstats["gen_s"], 1), "gen_tok": gstats["gen_tok"]}


# -----------------------------------------------------------------------------
# §6. Rounds
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
               "n_clipped",
               "n_stop", "n_eos", "gen_s", "gen_tok", "gen_tok_per_s", "rolls_per_min",
               "peak_blocks", "train_s", "n_groups_used", "n_groups_sat",
               "n_groups_dead", "n_docs", "n_loss_tokens",
               "n_comp_tok", "train_tok_per_s", "branch_frac", "loss_token_mean",
               "grad_norm", "lrm", "wnorm", "mem_gb", "round_s"]
metrics_path = HERE / f"metrics_{TAG}.csv"
passk_path = HERE / f"passk_{TAG}.csv"
mf = open(metrics_path, "w", newline="") if master else None
mw = None
if master:
    mw = csv.DictWriter(mf, fieldnames=METRIC_COLS)
    mw.writeheader()
pf = pw = None

RUN_DIR = Path(os.environ.get("RUN_DIR", str(Path.home() / ".cache" / "nanochat" / "fastrl_runs"))) / TAG
if master:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

curve: list[dict] = []
eval_curve: list[dict] = []
run_error = None
run_t0 = time.perf_counter()


def save_ckpt(step: int) -> None:
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "chatrl_checkpoints", OUT_TAG)
    opt_data = optimizer.state_dict() if SAVE_OPT else None
    # user_config carries the RESOLVED raw LRs so a chain off this checkpoint
    # inherits them directly (an SFT meta's snapshot has them null instead).
    meta_data = {"model_config": model.config.__dict__,
                 "user_config": dict(source=SOURCE, model_tag=MODEL_TAG,
                                     model_step=MODEL_STEP,
                                     unembedding_lr=unembedding_lr,
                                     embedding_lr=embedding_lr,
                                     matrix_lr=matrix_lr, scalar_lr=scalar_lr,
                                     init_lr_frac=INIT_LR_FRAC,
                                     lr_schedule=LR_SCHEDULE)}
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

        if EVAL_EVERY and rnd % EVAL_EVERY == 0:
            ev = run_eval(rnd)
            eval_curve.append(ev)
            if master:
                if pw is None:
                    pf = open(passk_path, "w", newline="")
                    pw = csv.DictWriter(pf, fieldnames=list(ev.keys()))
                    pw.writeheader()
                pw.writerow(ev)
                pf.flush()

        # -- generation ------------------------------------------------------
        idxs = FIXED_PROBLEMS if FIXED_PROBLEMS is not None else round_schedule[rnd]
        specs = [(i, train_prompts[i], K_DRAWS, MAX_TOKENS) for i in idxs]
        with record_function("round/gen"):
            rows, gstats = engine.run_round(engine.make_nodes(specs), rnd)

        # -- grade -----------------------------------------------------------
        with record_function("round/grade+group"):
            rewards = grade_rows(rows)
            n_clipped = 0
            comps, truncs = [], []
            for r, rw in zip(rows, rewards):
                comp = r["completion_token_ids"]
                trunc = r["terminal"] == "truncated"
                if CLIP_ANSWER and rw == 1.0:
                    # A stop-string retire cuts completion_text BEFORE the
                    # marker but comp keeps the ids through it — decide on the
                    # ids' own decode so the marker lead-in gets clipped too.
                    text = (tokenizer.decode(comp) if r["terminal"] == "stop_string"
                            else r["completion_text"])
                    clipped = clip_post_answer(comp, text)
                    if clipped is not comp:      # surgery happened: now ends in EOS
                        comp, trunc = clipped, False
                        n_clipped += 1
                comps.append(comp)
                truncs.append(trunc)
            by_pid: dict[int, list[int]] = {}
            for i, r in enumerate(rows):
                by_pid.setdefault(r["meta"], []).append(i)
            groups = [dict(
                prompt_ids=train_prompts[pid],
                completions=[comps[i] for i in idl],
                rewards=[rewards[i] for i in idl],
                truncated=[truncs[i] for i in idl],
            ) for pid, idl in by_pid.items()]

        # -- train (the optimizer's schedules are pre-computed) ---------------
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
            n_clipped=n_clipped,
            n_stop=gstats["stop_fires"],
            n_eos=sum(r["terminal"] == "emitted_eos" for r in rows),
            gen_s=round(gstats["gen_s"], 1), gen_tok=gstats["gen_tok"],
            gen_tok_per_s=round(gstats["gen_tok"] / gstats["gen_s"], 1),
            rolls_per_min=round(len(rows) / gstats["gen_s"] * 60, 1),
            peak_blocks=gstats.get("peak_blocks", 0),
            train_s=round(train_s, 1),
            n_groups_used=tstats["n_groups_used"],
            n_groups_sat=tstats["n_groups_sat"], n_groups_dead=tstats["n_groups_dead"],
            n_docs=tstats["n_docs"],
            n_loss_tokens=tstats["n_loss_tokens"], n_comp_tok=tstats["n_comp_tokens"],
            train_tok_per_s=round(tstats["n_comp_tokens"] / train_s, 1) if train_s else 0.0,
            branch_frac=round(tstats["branch_frac"], 4),
            loss_token_mean=round(tstats["loss_token_mean"], 6),
            grad_norm=round(tstats["grad_norm"], 6), lrm=round(float(lrm_table[rnd]), 4),
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
               f"eos {row['n_eos']:3d} stop {gstats['stop_fires']:2d} trunc {row['n_truncated']:2d} "
               f"clip {n_clipped:2d} | "
               f"prefill {gstats.get('replays', 0)}r {pf_pack:.0f}% | "
               f"kv peak {gstats.get('peak_blocks', 0)}/{engine.pool.num_blocks}", flush=True)
        print0(f"              train {train_s:5.1f}s ({row['train_tok_per_s']:>7,.0f} tok/s) | "
               f"{tstats['n_loss_tokens']:,} br-tok | "
               f"{tstats.get('n_packs', 0)} packs pad {tr_pad:.0f}% | "
               f"grp {tstats['n_groups_used']}/{tstats['n_groups_total']} "
               f"(sat {tstats['n_groups_sat']} dead {tstats['n_groups_dead']}) | "
               f"gnorm {tstats['grad_norm']:.3f} | lrm {lrm_table[rnd]:.3f} | "
               f"build+fwd+opt {tstats['t_build']:.2f}+{tstats['t_fwd']:.2f}+{tstats['t_opt']:.2f}"
               + ("" if tstats["stepped"] else " [SKIPPED no signal]"), flush=True)

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
    if master and pf:
        pf.close()
    total_s = time.perf_counter() - run_t0
    if curve:
        save_ckpt(len(curve))

    # ------------------------------------------------------------------
    # §7. Results — summary + save (+ optional HF push)
    # ------------------------------------------------------------------
    checklist = {
        "decode_body_compiled": bool(COMPILE),
        "decode_cuda_graphs_captured": len(engine.gd.graphs),
        "prefill_varlen_packed_graph": engine.pfg is not None and engine.pfg.graph is not None,
        "prefill_body_compiled": bool(PREFILL_COMPILE),
        "stop_string_detection": bool(STOP_DETECT),
        "prefix_sharing_active": True,
        "train_forward_compiled": bool(COMPILE_TRAIN),
        "train_static_buckets": list(TRAIN_BUCKETS),
        "fullparam_inplace_no_recapture": True,
        "same_process_gen_train": True,
        "bf16_params_fp32_opt_state": True,
        "kv_pool_permanent_gb": round((engine.pool.k_buf.size + engine.pool.v_buf.size) / 2 ** 30, 1),
        "tool_use": False,
    }
    result = dict(
        tag=TAG, k=K_DRAWS, problems_per_round=PPR, rounds_run=len(curve),
        budget=MAX_TOKENS, world_size=world_size, source=SOURCE,
        temperature=TEMPERATURE, adv_std=ADV_STD, init_lr_frac=INIT_LR_FRAC,
        lr_schedule=LR_SCHEDULE, lr_source=_lr_source,
        raw_lrs=dict(unembedding=unembedding_lr, embedding=embedding_lr,
                     matrix=matrix_lr, scalar=scalar_lr),
        error=run_error,
        solve_rate_first=(curve[0]["solve_rate"] if curve else None),
        solve_rate_last=(curve[-1]["solve_rate"] if curve else None),
        solve_rate_max=(max(c["solve_rate"] for c in curve) if curve else None),
        gen_tok_per_s_med=(sorted(c["gen_tok_per_s"] for c in curve)[len(curve) // 2]
                           if curve else None),
        train_s_med=(sorted(c["train_s"] for c in curve)[len(curve) // 2] if curve else None),
        round_s_med=(sorted(c["round_s"] for c in curve)[len(curve) // 2] if curve else None),
        passk_curve=eval_curve,
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
            md = tabulate([[k, v] for k, v in result.items() if k not in ("checklist", "passk_curve")],
                          headers=["metric", "value"], tablefmt="github")
            cmd = tabulate([[c["round"], c["n_correct"], c["solve_rate"], c["gen_s"],
                             c["train_s"], c["grad_norm"], c["wnorm"]] for c in curve],
                           headers=["round", "correct", "rate", "gen_s", "train_s",
                                    "gnorm", "|w|"], tablefmt="github")
            pk = (tabulate([[e["round"]] + [e[f"pass@{k+1}"] for k in range(EVAL_K)]
                            for e in eval_curve],
                           headers=["round"] + [f"pass@{k+1}" for k in range(EVAL_K)],
                           tablefmt="github") if eval_curve else "(no evals)")
            (HERE / f"result_{TAG}.md").write_text(
                f"# chat_rl_fast `{TAG}`\n\n{md}\n\nChecklist: `{json.dumps(checklist)}`\n\n"
                f"## Round curve\n\n{cmd}\n\n## pass@k\n\n{pk}\n")
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
