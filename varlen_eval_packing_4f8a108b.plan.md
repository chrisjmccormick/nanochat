---
name: Varlen Eval Packing
overview: Building on Plan 1 (varlen full cleanup), convert all evaluation paths to explicit varlen packing -- eliminating padding waste. ModelWrapper gains cu_seqlens support so eval code is model-agnostic. Auto-construct cu_seqlens removed from GPT.forward.
todos:
  - id: model-wrapper-cuseqlens
    content: Add cu_seqlens parameter to ModelWrapper.__call__ with _forward_varlen helper that unpacks 1D to padded (B,T), forwards through HF model, and repacks to 1D.
    status: pending
  - id: pack-sequences-util
    content: Extract pack_sequences(tokens_list, device) to shared utility (nanochat/common.py or new file). Used by core_eval and chat_eval.
    status: pending
  - id: core-eval-varlen
    content: Rewrite forward_model to accept token lists, pack via pack_sequences, return per-document losses/predictions lists. Update evaluate_example callers.
    status: pending
  - id: chat-eval-varlen
    content: Convert run_categorical_eval to pack prompts via pack_sequences, index logits using cu_seqlens offsets.
    status: pending
  - id: gpt-generate-explicit
    content: Convert GPT.generate to explicitly construct cu_seqlens=[0, T] each step instead of relying on auto-construct.
    status: pending
  - id: remove-auto-construct
    content: Remove auto-construct cu_seqlens from GPT.forward. Add ValueError when neither cu_seqlens nor kv_cache provided.
    status: pending
  - id: hf-bpb-unify
    content: "Optional: switch HF model BPB to varlen loader too (ModelWrapper handles unpacking). Eliminates is_hf_model branch."
    status: pending
isProject: false
---

# Varlen Eval Packing: Eliminate Padding Waste

**Prerequisite:** Plan 1 (Varlen Full Cleanup) must be completed first. This plan is implemented as a new branch off Plan 1.

## Motivation

Plan 1 removes `flash_attn_func` and uses auto-construct cu_seqlens in `GPT.forward` to handle `(B, T)` callers. This works, but preserves the same padding waste as the old batched code: sequences of different lengths are padded to the longest, and all padding tokens are fully processed by the model.

This plan eliminates that waste by converting eval code to explicit varlen packing (concatenate sequences without padding, pass cu_seqlens). The `ModelWrapper` gains cu_seqlens support so that eval code doesn't need to know whether it's talking to a nanochat model or a HuggingFace model.

## What changes

### 1. ModelWrapper gains cu_seqlens support ([base_eval.py](nanochat/scripts/base_eval.py))

Add a `cu_seqlens` parameter to `ModelWrapper.__call`__ and a `_forward_varlen` helper:

```python
class ModelWrapper:
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, cu_seqlens=None, loss_reduction='mean'):
        if cu_seqlens is not None:
            return self._forward_varlen(input_ids, targets, cu_seqlens, loss_reduction)
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=-1, reduction=loss_reduction
        )
        return loss

    def _forward_varlen(self, input_ids, targets, cu_seqlens, loss_reduction):
        """Unpack 1D varlen to padded (B, T), forward through HF model, repack."""
        device = input_ids.device
        doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_docs = (doc_lens > 0).sum().item()
        max_doc_len = doc_lens.max().item()

        # Unpack: 1D --> padded (num_docs, max_doc_len)
        batched = torch.zeros(num_docs, max_doc_len, dtype=input_ids.dtype, device=device)
        doc_idx = 0
        for i in range(len(doc_lens)):
            length = doc_lens[i].item()
            if length == 0:
                continue
            start = cu_seqlens[i].item()
            batched[doc_idx, :length] = input_ids[start:start + length]
            doc_idx += 1

        logits = self.model(batched).logits  # (num_docs, max_doc_len, V)

        # Repack: padded (num_docs, max_doc_len, V) --> (1, total_T, V)
        total_T = input_ids.size(0)
        packed = torch.empty(1, total_T, logits.size(-1), dtype=logits.dtype, device=device)
        doc_idx = 0
        for i in range(len(doc_lens)):
            length = doc_lens[i].item()
            if length == 0:
                continue
            start = cu_seqlens[i].item()
            packed[0, start:start + length] = logits[doc_idx, :length]
            doc_idx += 1

        if targets is None:
            return packed
        loss = F.cross_entropy(
            packed.view(-1, packed.size(-1)), targets.view(-1),
            ignore_index=-1, reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device
```

Key contract: when `cu_seqlens` is provided, `input_ids` is 1D `(total_T,)` and the return is `(1, total_T, V)` -- matching `GPT.forward`'s varlen output shape.

Future optimization opportunity: HF models with Flash Attention 2 backends could accept packed inputs directly via `position_ids` and `attention_mask`, skipping the unpack/repack. Not needed now.

### 2. CORE evaluation explicit packing ([core_eval.py](nanochat/nanochat/core_eval.py))

#### Replace `stack_sequences` + `forward_model` with varlen equivalents

Delete `stack_sequences` (no longer needed). Replace `forward_model` with a varlen version:

```python
def pack_sequences(tokens_list, device):
    """Pack variable-length token sequences into 1D buffer + cu_seqlens."""
    packed = torch.cat([torch.tensor(t, dtype=torch.long, device=device) for t in tokens_list])
    cu_seqlens = torch.zeros(len(tokens_list) + 1, dtype=torch.int32, device=device)
    for i, t in enumerate(tokens_list):
        cu_seqlens[i + 1] = cu_seqlens[i] + len(t)
    return packed, cu_seqlens


@torch.no_grad()
def forward_model(model, tokens_list, device):
    """
    Pack token sequences into varlen, forward, extract per-sequence losses and predictions.
    Returns (losses_list, preds_list) where each element corresponds to one input sequence.
    """
    packed, cu_seqlens = pack_sequences(tokens_list, device)
    outputs = model(packed, cu_seqlens=cu_seqlens).squeeze(0)  # (total_T, V)

    losses_list = []
    preds_list = []
    for i in range(len(tokens_list)):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        doc_logits = outputs[s:e]
        doc_tokens = packed[s:e]
        doc_targets = torch.roll(doc_tokens, -1)
        doc_losses = F.cross_entropy(doc_logits[:-1], doc_targets[:-1], reduction='none')
        doc_losses = torch.cat([doc_losses, torch.tensor([float('nan')], device=device)])
        losses_list.append(doc_losses)
        preds_list.append(doc_logits.argmax(dim=-1))
    return losses_list, preds_list
```

The return type changes from `(B, T)` tensors to lists of variable-length tensors. This is more natural since the sequences actually have different lengths.

#### Update `evaluate_example` callers

The signature of `forward_model` changes: it now takes `tokens_list` (list of lists) and `device` instead of `input_ids` (padded tensor).

In `evaluate_example` ([core_eval.py](nanochat/nanochat/core_eval.py) line ~168), the change is:

**Before (Plan 1):**

```python
input_ids = stack_sequences(tokens, pad_token_id)
input_ids = input_ids.to(device)
losses, predictions = forward_model(model, input_ids)
# ... losses[i, si-1:ei-1] ...
```

**After (Plan 2):**

```python
losses_list, preds_list = forward_model(model, tokens, device)
# ... losses_list[i][si-1:ei-1] ...
```

The max_seq_len truncation logic (lines 198-213) also changes: instead of truncating tokens in-place and adjusting indices, truncate each token list directly. The logic is simpler since we're working with plain lists rather than padded tensors.

The MC/schema/LM scoring logic (lines 224-238) changes indexing from `losses[i, si-1:ei-1]` to `losses_list[i][si-1:ei-1]` and `predictions[0, si-1:ei-1]` to `preds_list[0][si-1:ei-1]`.

### 3. Chat categorical evaluation explicit packing ([chat_eval.py](nanochat/scripts/chat_eval.py))

In `run_categorical_eval` (line ~88), replace the pad-and-stack with varlen packing:

**Before (Plan 1):**

```python
padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)
logits = model(prompt_ids)  # (B, T, V)
# ... logits[idx, answer_pos, letter_ids] ...
```

**After (Plan 2):**

```python
packed, cu_seqlens = pack_sequences(prompt_ids, device)  # reuse from core_eval or extract to common util
logits = model(packed, cu_seqlens=cu_seqlens).squeeze(0)  # (total_T, V)
# ... logits[cu_seqlens[idx].item() + answer_pos, letter_ids] ...
```

The `answer_time_positions` list stays (it's per-sequence), but indexing into logits uses `cu_seqlens[idx] + answer_pos` instead of `(idx, answer_pos)`.

`pack_sequences` should be extracted to a shared utility (e.g., `nanochat/common.py` or a new `nanochat/packing.py`) since both `core_eval.py` and `chat_eval.py` need it.

### 4. GPT.generate explicit cu_seqlens ([gpt.py](nanochat/nanochat/gpt.py))

Convert `GPT.generate` (line ~496) to explicitly construct cu_seqlens instead of relying on auto-construct:

```python
ids = torch.tensor(tokens, dtype=torch.long, device=device)  # 1D, no batch dim
for _ in range(max_tokens):
    cu_seqlens = torch.tensor([0, ids.size(0)], dtype=torch.int32, device=device)
    logits = self.forward(ids, cu_seqlens=cu_seqlens)  # (1, T, V)
    logits = logits[:, -1, :]  # (1, V)
    # ... sampling unchanged ...
    ids = torch.cat((ids, next_ids.squeeze(0)))  # stay 1D
```

### 5. Remove auto-construct from GPT.forward ([gpt.py](nanochat/nanochat/gpt.py))

Delete the `elif kv_cache is None` auto-construct block added in Plan 1. `GPT.forward` now strictly requires either `cu_seqlens` or `kv_cache`:

```python
def forward(self, idx, targets=None, cu_seqlens=None, kv_cache=None, loss_reduction='mean'):
    if cu_seqlens is not None:
        assert idx.ndim == 1
        idx = idx.unsqueeze(0)
        if targets is not None:
            targets = targets.unsqueeze(0)
        max_seq_len = self.config.sequence_len
    elif kv_cache is not None:
        max_seq_len = None
    else:
        raise ValueError("GPT.forward requires either cu_seqlens or kv_cache")

    B, T = idx.size()
    # ... rest unchanged ...
```

This makes the two-path architecture explicit and enforced. Any caller that forgets cu_seqlens gets a clear error instead of silent auto-construction.

### 6. BPB evaluation for HuggingFace models ([base_eval.py](nanochat/scripts/base_eval.py))

With ModelWrapper supporting cu_seqlens, HF model BPB can optionally switch to varlen too (using `tokenizing_distributed_data_loader_varlen`). The ModelWrapper will unpack internally. This eliminates the `is_hf_model` branch in the BPB section:

```python
# Both model types use varlen loader now
loader = tokenizing_distributed_data_loader_varlen(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
bpb = evaluate_bpb(model, loader, steps, token_bytes)
```

This is optional -- the HF BPB path already works via auto-construct from Plan 1. But it's cleaner to have one code path.

---

## What does NOT change (from Plan 1)

- `flash_attention.py`: already cleaned up in Plan 1
- `CausalSelfAttention.forward`: already two-branch (varlen + kv_cache) from Plan 1
- `chat_sft.py`: already varlen from Plan 1
- `chat_rl.py`: already varlen from Plan 1
- `loss_eval.py`: no changes needed
- `Engine.generate`: uses KV cache, unaffected
- `dataloader.py`: no changes needed

## Summary of interface changes


| Component               | Plan 1                              | Plan 2                                                        |
| ----------------------- | ----------------------------------- | ------------------------------------------------------------- |
| GPT.forward             | Auto-construct cu_seqlens for (B,T) | Requires cu_seqlens or kv_cache (raises ValueError otherwise) |
| ModelWrapper            | No cu_seqlens support               | Accepts cu_seqlens, unpacks to (B,T) for HF model             |
| core_eval.forward_model | Takes (B,T) tensor                  | Takes list of token lists, returns per-doc lists              |
| chat_eval categorical   | Pads to (B,T)                       | Packs to 1D + cu_seqlens                                      |
| GPT.generate            | Auto-construct                      | Explicit cu_seqlens=[0, T]                                    |


