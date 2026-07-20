"""chat_rl_fast.py — fused same-process RL speedrun on GSM8K.

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
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from nanochat.fast_engine import install_dao_flash_attention
install_dao_flash_attention()  # before anything touches nanochat.flash_attention

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.fast_engine import (
    FastEngine, cast_model_bf16, setup_fp32_optimizer, group_advantages,
    build_reinforce_packs, reinforce_forward_loss)
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
PASS1        = _env_int("PASS1", 0)               # two-pass off by default (wash at small budgets)
# Sampler (speedrun house sampler)
TEMPERATURE  = _env_float("TEMPERATURE", 0.6)
TOP_P        = _env_float("TOP_P", 0.95)
TOP_K        = _env_int("TOP_K", 512)
# Trainer
ADV_STD      = _env_flag("ADV_STD", 1)            # 0 -> stock (r - mean)
TRAIN_BRANCH_TEMP  = _env_float("TRAIN_BRANCH_TEMP", 1.0)
TRAIN_BRANCH_TOP_P = _env_float("TRAIN_BRANCH_TOP_P", 0.95)
GRAD_CLIP    = _env_float("GRAD_CLIP", 1.0)       # exact on 1 GPU; skipped under DDP
UNEMBEDDING_LR = _env_float("UNEMBEDDING_LR", 0.004)
EMBEDDING_LR   = _env_float("EMBEDDING_LR", 0.2)
MATRIX_LR      = _env_float("MATRIX_LR", 0.02)
WEIGHT_DECAY   = _env_float("WEIGHT_DECAY", 0.0)
INIT_LR_FRAC   = _env_float("INIT_LR_FRAC", 0.05)
_TB_ENV = os.environ.get("TRAIN_BUCKETS")
TRAIN_BUCKETS = tuple(int(x) for x in _TB_ENV.split(",")) if _TB_ENV else (16384,)
MAX_NUM_DOCS  = _env_int("MAX_NUM_DOCS", 64)
COMPILE_TRAIN = _env_flag("COMPILE_TRAIN", 1)
# Engine knobs
KV_POOL_GB   = _env_float("KV_POOL_GB", 24)
MAX_SEQS     = _env_int("MAX_SEQS", 256)
MACRO_N      = _env_int("MACRO_N", 8)
_BK_ENV = os.environ.get("BUCKETS")
BUCKETS      = tuple(int(x) for x in _BK_ENV.split(",")) if _BK_ENV else None
PREFILL_T    = _env_int("PREFILL_T", 2048)
PREFILL_SEQS = _env_int("PREFILL_SEQS", 12)
PANTRY_BLOCKS = _env_int("PANTRY_BLOCKS", 96)
STARVE_JOBS  = _env_int("STARVE_JOBS", 12)
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
OUT_TAG       = os.environ.get("OUT_TAG", "d24-fastrl")
SOURCE        = os.environ.get("SOURCE", "sft")
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
assert max_prompt <= PREFILL_T, f"longest prompt {max_prompt} > PREFILL_T={PREFILL_T}"
if PASS1:
    assert max_prompt + PASS1 <= PREFILL_T, "prompt+PASS1 exceeds PREFILL_T"

shard = list(range(rank, len(train_task), world_size))
num_rounds = (len(train_task) // PPR) * EPOCHS
if ROUNDS_CAP:
    num_rounds = min(num_rounds, ROUNDS_CAP)
print0(f"[{TAG}] {PPR} problems x K={K_DRAWS} = {PPR * K_DRAWS} rollouts/round "
       f"x {num_rounds} rounds @ budget {MAX_TOKENS} | max prompt {max_prompt} tok "
       f"| train buckets {TRAIN_BUCKETS} | stop-detect "
       f"{'ON ' + str(STOP) if STOP_DETECT else 'OFF'}", flush=True)

# -----------------------------------------------------------------------------
# §2. Optimizer (fp32 master/state) -> bf16 cast -> engine + graph capture
# -----------------------------------------------------------------------------
# Snapshot fp32 masters from the checkpoint weights BEFORE the bf16 cast.
optimizer = setup_fp32_optimizer(model, unembedding_lr=UNEMBEDDING_LR,
                                 embedding_lr=EMBEDDING_LR, matrix_lr=MATRIX_LR,
                                 weight_decay=WEIGHT_DECAY)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * INIT_LR_FRAC
    group["initial_lr"] = group["lr"]

cast_model_bf16(model)
model.eval()

engine = FastEngine(
    model, tokenizer,
    kv_pool_gb=KV_POOL_GB, max_seqs=MAX_SEQS, max_tokens=max(MAX_TOKENS, EVAL_MAX_TOKENS),
    max_prompt_len=max_prompt, macro_n=MACRO_N, buckets=BUCKETS,
    prefill_t=PREFILL_T, prefill_seqs=PREFILL_SEQS, pantry_blocks=PANTRY_BLOCKS,
    starve_jobs=STARVE_JOBS, pass1=PASS1, temperature=TEMPERATURE, top_p=TOP_P,
    top_k=TOP_K, stop_detect=STOP_DETECT, stop_strings=tuple(STOP),
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

# Lend the pool's physical back BEFORE the train warmup — full-size pool and
# full-size train buckets never need to coexist.
_lent_gb = (engine.pool.k_buf.size + engine.pool.v_buf.size) / 2 ** 30
print0(f"  [vmm] pool physical lent back ({_lent_gb:.1f} GB, {engine.pool.lend():.2f}s) "
       f"for train warmup", flush=True)

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
# §3. Grading + trainer step
# -----------------------------------------------------------------------------
def grade_rows(rows) -> list[float]:
    """GSM8K regex reward, inline (microseconds per row — no fork pool needed)."""
    return [train_task.reward(train_convs[r["meta"]], r["completion_text"]) for r in rows]


def train_step(groups: list[dict]) -> dict:
    """One branch-masked REINFORCE optimizer step over the round's problem
    groups. Same advantages/exclusions/DAPO token-mean as the speedrun; no
    'unresolved' verdicts (regex reward always resolves)."""
    docs = []
    n_groups_used = n_excluded = 0
    for g in groups:
        adv = group_advantages(np.asarray(g["rewards"], dtype=np.float64), use_std=ADV_STD)
        if adv is None:
            continue                                  # zero-signal group
        n_groups_used += 1
        for k, comp in enumerate(g["completions"]):
            # truncated-incorrect stays in the baseline but is excluded from loss
            if (g["truncated"][k] and g["rewards"][k] == 0) or not comp:
                n_excluded += 1
                continue
            docs.append((g["prompt_ids"], list(comp), float(adv[k])))
    total_tokens = total_branch = total_comp = 0
    total_loss = 0.0
    n_packs = 0
    if docs:
        packs, pstats = build_reinforce_packs(
            docs, buckets=list(TRAIN_BUCKETS), max_num_docs=MAX_NUM_DOCS,
            pad_id=PAD_ID, max_doc_len=SEQ_CAP)
        n_packs = pstats["n_packs"]
        for pk in packs:
            loss_sum, n_tok, n_branch, n_comp = TRAIN_FN(
                model, pk.input_ids, pk.cu_seqlens, pk.targets, pk.comp_mask,
                pk.adv_tok, TRAIN_BRANCH_TEMP, TRAIN_BRANCH_TOP_P)
            total_comp += int(n_comp.item())
            total_branch += int(n_branch.item())
            nt = int(n_tok.item())
            if nt > 0:
                loss_sum.backward()                   # unnormalized; accumulates
                total_loss += float(loss_sum.detach())
                total_tokens += nt
            del loss_sum
    # DAPO token-level mean across ALL ranks' loss tokens
    tok_t = torch.tensor(float(total_tokens), device=device)
    if ddp:
        dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
    global_tokens = float(tok_t.item())
    gnorm = 0.0
    stepped = False
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
    return dict(n_groups_used=n_groups_used, n_groups_total=len(groups),
                n_docs=len(docs), n_excluded=n_excluded, n_packs=n_packs,
                n_loss_tokens=total_tokens, n_comp_tokens=total_comp,
                branch_frac=(total_branch / total_comp) if total_comp else 0.0,
                loss_token_mean=(total_loss / total_tokens) if total_tokens else 0.0,
                grad_norm=gnorm, stepped=stepped)


# -----------------------------------------------------------------------------
# §4. Eval — pass@k on the test split through the fast engine
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
    if ddp:
        dist.all_reduce(n_rec, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)
    passk = (passk / n_rec.item()).tolist()
    print0(f"  [eval r{rnd}] " + ", ".join(f"pass@{k+1}: {v:.4f}" for k, v in enumerate(passk))
           + f" | {gstats['gen_tok']:,} tok in {gstats['gen_s']:.1f}s "
           f"({gstats['gen_tok']/gstats['gen_s']:,.0f} tok/s)", flush=True)
    return {f"pass@{k+1}": round(v, 4) for k, v in enumerate(passk)} | {
        "round": rnd, "n": int(n_rec.item()),
        "gen_s": round(gstats["gen_s"], 1), "gen_tok": gstats["gen_tok"]}


# -----------------------------------------------------------------------------
# §5. Rounds
# -----------------------------------------------------------------------------
METRIC_COLS = ["round", "n_rollouts", "n_correct", "solve_rate", "n_truncated",
               "n_stop", "n_eos", "gen_s", "gen_tok", "gen_tok_per_s", "rolls_per_min",
               "train_s", "vmm_s", "n_ext", "n_groups_used", "n_docs", "n_loss_tokens",
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
    if master or opt_data is not None:
        save_checkpoint(ckpt_dir, step, model.state_dict() if master else None,
                        opt_data, {"model_config": model.config.__dict__}, rank=rank)
    if master:
        print(f"  saved checkpoint -> {ckpt_dir} (step {step})", flush=True)


try:
    for rnd in range(num_rounds):
        r_t0 = time.perf_counter()
        vmm_map_s = engine.pool.reclaim()

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
        idxs = [shard[(rnd * ppr_rank + j) % len(shard)] for j in range(ppr_rank)]
        specs = [(i, train_prompts[i], K_DRAWS, MAX_TOKENS) for i in idxs]
        rows, gstats = engine.run_round(engine.make_nodes(specs), rnd)
        vmm_unmap_s = engine.pool.lend()

        # -- grade -----------------------------------------------------------
        rewards = grade_rows(rows)
        by_pid: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_pid.setdefault(r["meta"], []).append(i)
        groups = [dict(
            prompt_ids=train_prompts[pid],
            completions=[rows[i]["completion_token_ids"] for i in idl],
            rewards=[rewards[i] for i in idl],
            truncated=[rows[i]["terminal"] == "truncated" for i in idl],
        ) for pid, idl in by_pid.items()]

        # -- train -----------------------------------------------------------
        lrm = 1.0 - rnd / num_rounds
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        _t = time.perf_counter()
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
            n_stop=gstats["stop_fires"],
            n_eos=sum(r["terminal"] == "emitted_eos" for r in rows),
            gen_s=round(gstats["gen_s"], 1), gen_tok=gstats["gen_tok"],
            gen_tok_per_s=round(gstats["gen_tok"] / gstats["gen_s"], 1),
            rolls_per_min=round(len(rows) / gstats["gen_s"] * 60, 1),
            train_s=round(train_s, 1), vmm_s=round(vmm_map_s + vmm_unmap_s, 2),
            n_ext=gstats["n_extended"],
            n_groups_used=tstats["n_groups_used"], n_docs=tstats["n_docs"],
            n_loss_tokens=tstats["n_loss_tokens"], n_comp_tok=tstats["n_comp_tokens"],
            train_tok_per_s=round(tstats["n_comp_tokens"] / train_s, 1) if train_s else 0.0,
            branch_frac=round(tstats["branch_frac"], 4),
            loss_token_mean=round(tstats["loss_token_mean"], 6),
            grad_norm=round(tstats["grad_norm"], 6), lrm=round(lrm, 4),
            wnorm=round(wnorm, 2),
            mem_gb=round((lambda f_t: (f_t[1] - f_t[0]) / 2 ** 30)(torch.cuda.mem_get_info()), 1),
            round_s=round(time.perf_counter() - r_t0, 1))
        curve.append(row)
        if master:
            mw.writerow(row)
            mf.flush()
            if SAVE_ROLLOUTS:
                import pandas as pd
                pd.DataFrame(rows).assign(reward=rewards).to_parquet(
                    RUN_DIR / f"rollouts_round_{rnd:04d}.parquet", index=False)
        print0(f"  [round {rnd:3d}] solve {int(agg[0]):3d}/{int(agg[1])} ({100*solve_rate:5.1f}%) | "
               f"gen {gstats['gen_s']:5.1f}s ({row['gen_tok_per_s']:>7,.0f} tok/s) | "
               f"train {train_s:4.1f}s vmm {vmm_map_s + vmm_unmap_s:.1f}s "
               f"({tstats['n_loss_tokens']} br-tok, gnorm {tstats['grad_norm']:.3f})"
               + ("" if tstats["stepped"] else " [SKIPPED no signal]"), flush=True)

        if SAVE_EVERY and rnd > 0 and rnd % SAVE_EVERY == 0:
            save_ckpt(rnd)
except BaseException as e:
    run_error = f"{type(e).__name__}: {e}"
    raise
finally:
    if master and mf:
        mf.close()
    if master and pf:
        pf.close()
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
        "stop_string_detection": bool(STOP_DETECT),
        "prefix_sharing_active": True,
        "train_forward_compiled": bool(COMPILE_TRAIN),
        "train_static_buckets": list(TRAIN_BUCKETS),
        "fullparam_inplace_no_recapture": True,
        "same_process_gen_train": True,
        "bf16_params_fp32_opt_state": True,
        "kv_vmm_lend_back_gb": round((engine.pool.k_buf.size + engine.pool.v_buf.size) / 2 ** 30, 1),
        "two_pass_pass1": PASS1,
        "tool_use": False,
    }
    result = dict(
        tag=TAG, k=K_DRAWS, problems_per_round=PPR, rounds_run=len(curve),
        budget=MAX_TOKENS, world_size=world_size, source=SOURCE,
        temperature=TEMPERATURE, adv_std=ADV_STD, init_lr_frac=INIT_LR_FRAC,
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
