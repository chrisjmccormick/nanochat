---
name: CORE eval A/B comparison
overview: Run both baseline and optimized CORE evaluation at end of pre-training, logging only baseline to wandb while saving a side-by-side comparison JSON to disk for cross-run analysis.
todos:
  - id: parameterize-evaluate-core
    content: Add `evaluate_task_fn` parameter to `evaluate_core` in `base_eval.py` (line 109), default to existing import, thread through to task evaluation call (line 170)
    status: completed
  - id: add-optim-eval-to-train
    content: "In `base_train.py`, after baseline CORE eval (~line 646): import `evaluate_task` from `core_eval_optim`, run second `evaluate_core` with it, time it, print results"
    status: completed
  - id: save-comparison-json
    content: On rank 0, write `logs/core_eval_comparison/{run_name}.json` with both baseline and optimized results plus run metadata
    status: completed
  - id: wandb-save-optim
    content: Add `wandb.save('nanochat/core_eval_optim.py')` near existing wandb.save calls (~line 694)
    status: completed
isProject: false
---

# CORE Eval A/B Comparison

## Context

- [base_train.py](nanochat/scripts/base_train.py) runs `evaluate_core` (from [base_eval.py](nanochat/scripts/base_eval.py)) at end of training (line 636-646), logging results to wandb
- `evaluate_core` iterates over tasks, calls `evaluate_task` for each, computes centered accuracy, and returns a dict with `results`, `centered_results`, `task_times`, `core_metric`
- [core_eval.py](nanochat/nanochat/core_eval.py) has the baseline `evaluate_task` — processes examples one at a time with per-example forward passes
- [core_eval_optim.py](nanochat/nanochat/core_eval_optim.py) has the optimized `evaluate_task` — batches sequences across examples, sorts by length to minimize padding, uses a B*T token budget

## Changes

### 1. Parameterize `evaluate_core` in `base_eval.py`

Add an optional `evaluate_task_fn` parameter to `evaluate_core` so callers can inject either implementation:

```python
def evaluate_core(model, tokenizer, device, max_per_task=-1, evaluate_task_fn=None):
    if evaluate_task_fn is None:
        evaluate_task_fn = evaluate_task  # module-level import from core_eval.py
    ...
    accuracy = evaluate_task_fn(model, tokenizer, data, device, task_meta)
```

This works because both signatures are compatible — `core_eval_optim.evaluate_task` adds `batch_size=64` as an optional kwarg.

### 2. Add optimized CORE eval run in `base_train.py`

After the existing baseline CORE eval block (lines 636-646):

- Import: `from nanochat.core_eval_optim import evaluate_task as evaluate_task_optim`
- Run a second `evaluate_core` call passing `evaluate_task_fn=evaluate_task_optim`
- Time the optimized run separately
- Print optimized results to stdout (for log visibility)
- Do NOT log optimized results to wandb (existing wandb logging remains baseline-only)

### 3. Save comparison data to disk as JSON

On rank 0, save a JSON file per run to `logs/core_eval_comparison/{run_name}.json` containing:

- **Run metadata**: run name, timestamp, training step, total training flops
- **Baseline results**: `core_metric`, per-task `results`, `centered_results`, `task_times`, total eval wall time
- **Optimized results**: same structure as baseline

This allows loading all JSON files into a DataFrame for cross-run comparison of accuracy deltas and speedup.

### 4. Also save `core_eval_optim.py` to wandb artifacts

Add `wandb.save("nanochat/core_eval_optim.py")` alongside the existing `wandb.save("nanochat/core_eval.py")` for reproducibility.
