---
name: Varlen Flash Attention CORE
overview: Add variable-length flash attention support to the model and create a third CORE eval variant (`core_eval_varlen.py`) that packs variable-length sequences via `cu_seqlens`, pads to a fixed token budget for static `torch.compile`, and eliminates per-sequence padding waste.
todos:
  - id: flash-attn-varlen
    content: Add `flash_attn_varlen_func` to `flash_attention.py` with FA3 native path + SDPA fallback, export in SimpleNamespace
    status: completed
  - id: gpt-varlen-forward
    content: Add optional `cu_seqlens` parameter to `CausalSelfAttention.forward()`, `Block.forward()`, and `GPT.forward()` with varlen rotary embedding position computation
    status: completed
  - id: core-eval-varlen
    content: "Create `nanochat/core_eval_varlen.py` with varlen-packed batching at a fixed token budget: concatenate sequences, pad to budget, build cu_seqlens, forward with varlen, extract per-sequence results"
    status: completed
  - id: base-train-wire
    content: "Add third varlen eval pass to `base_train.py`: torch.compile(orig_model) with static shapes before the varlen pass, update comparison JSON and wandb artifacts"
    status: completed
isProject: false
---

# Varlen Flash Attention for CORE Eval

## Context

Currently, CORE eval pads variable-length sequences to the longest in each batch, wasting compute on pad tokens. Flash Attention's varlen API (`flash_attn_varlen_func`) accepts concatenated sequences with cumulative sequence length metadata (`cu_seqlens`), eliminating per-sequence padding. Combined with padding to a **fixed token budget**, this enables `torch.compile` with fully static shapes (no recompilation).

```python
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                       causal=False, window_size=(-1,-1), ...)
# q, k, v: (total_tokens, nheads, headdim) -- 3D, all sequences concatenated
# cu_seqlens: (num_seqs+1,) int32, cumulative sequence boundaries
```

## File Changes

### 1. Export varlen function in `flash_attention.py`

Add `flash_attn_varlen_func` to the module. The FA3 path wraps the native `_fa3.flash_attn_varlen_func`. The SDPA fallback reconstructs padded batches from the varlen format, runs existing `_sdpa_attention`, then gathers results back. Add to the `flash_attn` SimpleNamespace export.

### 2. Add varlen forward path to `gpt.py`

Add an optional `cu_seqlens` parameter to `CausalSelfAttention.forward()`, `Block.forward()`, and `GPT.forward()`:

- `**GPT.forward()**`: When `cu_seqlens` is provided, input is `(1, total_tokens)`. Compute per-token positions from `cu_seqlens` (positions reset to 0 at each sequence boundary) and index into precomputed `cos`/`sin` buffers for correct rotary embeddings. Pass `cu_seqlens` down through each block. Embedding lookup, norms, MLP, and lm_head work unchanged since they operate on the last dimension.
- `**CausalSelfAttention.forward()**`: When `cu_seqlens` is provided:
  - Squeeze batch dim: `(1, T, H, D)` -> `(T, H, D)` for q, k, v
  - Call `flash_attn.flash_attn_varlen_func` instead of `flash_attn_func`
  - Unsqueeze back to `(1, T, H, D)`
- **Rotary position computation**: Build a positions tensor from `cu_seqlens` using vectorized ops:
  ```python
  # e.g. cu_seqlens=[0,3,7,10] -> positions=[0,1,2, 0,1,2,3, 0,1,2]
  positions = torch.arange(total_tokens, device=device)
  offsets = torch.zeros(total_tokens, dtype=torch.long, device=device)
  offsets[cu_seqlens[1:-1]] = cu_seqlens[1:-1] - cu_seqlens[:-2]  # jump-back amounts
  positions -= offsets.cumsum(0)  # subtract cumulative offsets -> positions reset per seq
  cos = self.cos[0, positions]  # (total_tokens, 1, D/2)
  sin = self.sin[0, positions]
  ```

### 3. Create `nanochat/core_eval_varlen.py`

New file modeled on [core_eval_optim.py](nanochat/core_eval_optim.py) but using varlen packing + fixed budget:

- `**forward_model_varlen()**`: Takes concatenated token IDs `(1, budget)` and `cu_seqlens`, calls `model(input_ids, cu_seqlens=cu_seqlens)`, returns logits `(1, budget, vocab)`. Uses `cu_seqlens` to slice only the real-token positions for per-sequence losses and predictions.
- **Fixed-budget batching strategy**: Every forward call uses the same tensor shape `(1, budget)`:
  1. Sort sequences by length (same as `core_eval_optim.py`)
  2. Greedily pack sequences until total tokens approach `budget`
  3. Concatenate packed sequences, then **right-pad to exactly `budget**` with a pad token
  4. Build `cu_seqlens` marking only the real sequence boundaries
  5. Flash attention only attends within `cu_seqlens` ranges -- padded tail tokens are never attended to
  This guarantees a constant input shape, enabling `torch.compile` with static shapes (no `dynamic=True` needed).
- `**evaluate_task()**`: Same 4-phase structure as `core_eval_optim.py`:
  1. Pre-process examples (render prompts, tokenize) -- reuse helpers from [core_eval.py](nanochat/core_eval.py)
  2. Pack sequences into fixed-budget varlen batches
  3. Forward through model, extract per-sequence results using `cu_seqlens`
  4. Aggregate correctness and sync across ranks

### 4. Wire into `base_train.py`

Add a third CORE eval pass after the existing two (lines 649-691). Key difference: compile the model with **static shapes** before this pass:

```python
from nanochat.core_eval_varlen import evaluate_task as evaluate_task_varlen

# Compile model with static shapes for varlen eval (fixed budget = constant input shape)
compiled_model = torch.compile(orig_model)

varlen_core_results = evaluate_core(compiled_model, tokenizer, device, max_per_task=-1,
                                    evaluate_task_fn=evaluate_task_varlen)
```

- `torch.compile(orig_model)` without `dynamic=True` uses static shapes -- every call to `compiled_model(input_ids, cu_seqlens=...)` sees the same `(1, budget)` shape, so the graph is traced once and reused.
- Add `varlen` entry to the `comparison_data` dict alongside `baseline` and `optimized`.
- Save `nanochat/core_eval_varlen.py` to wandb artifacts.

```mermaid
flowchart TD
    subgraph current [Current: Padded Batching]
        A1["Seqs: 3, 7, 5 tokens"] --> B1["Pad each to 7: 21 total tokens"]
        B1 --> C1["forward(3, 7) with dynamic compile"]
    end

    subgraph varlen [New: Varlen + Fixed Budget]
        A2["Seqs: 3, 7, 5 tokens"] --> B2["Concat: 15 real tokens"]
        B2 --> C2["Pad to budget=32768: always (1, 32768)"]
        C2 --> D2["cu_seqlens=[0,3,10,15]"]
        D2 --> E2["forward_varlen with static compile"]
    end
```



