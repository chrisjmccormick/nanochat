"""
Varlen CORE evaluation: packs variable-length sequences using flash_attn_varlen_func
with a fixed token budget for static torch.compile (no recompilation across batches).

Instead of padding each sequence to the longest in a batch (wasting compute on pad tokens),
sequences are concatenated and cu_seqlens marks the boundaries. Flash attention only
attends within these boundaries. The concatenated tensor is then right-padded to a fixed
token budget so that every forward call has identical tensor shapes, enabling torch.compile
with static shapes.
"""
import random

import torch
import torch.distributed as dist

# Reuse prompt rendering and tokenization helpers from the baseline implementation
from nanochat.core_eval import (
    render_prompts_mc, render_prompts_schema, render_prompts_lm,
    batch_sequences_mc, batch_sequences_schema, batch_sequences_lm,
)


@torch.no_grad()
def forward_model_varlen(model, input_ids, cu_seqlens):
    """
    Forward a varlen-packed batch through the model.

    Precomputes position-aware rotary embeddings and max_seqlen OUTSIDE the
    compiled model to avoid graph breaks from data-dependent ops (.item(),
    variable-length indexing, branching on cu_seqlens).

    Args:
        model: GPT model (or torch.compiled wrapper)
        input_ids: (1, budget) token ids -- concatenated sequences padded to fixed budget
        cu_seqlens: (num_seqs+1,) int32 -- cumulative sequence lengths marking boundaries

    Returns:
        logits: (1, budget, vocab_size) tensor
    """
    # Access the unwrapped model to call compute_varlen_cos_sin (not part of compiled graph)
    orig = getattr(model, '_orig_mod', model)
    T = input_ids.size(1)
    cos_sin, _ = orig.compute_varlen_cos_sin(T, cu_seqlens, input_ids.device)
    # Use config.sequence_len as a fixed max_seqlen upper bound so torch.compile
    # sees a constant (no recompilation). All sequences are already truncated to
    # this length upstream, so it's always valid. Flash attention uses it as a
    # performance hint -- a slightly larger value is fine for correctness.
    max_seqlen = orig.config.sequence_len
    return model(input_ids, cu_seqlens=cu_seqlens, cos_sin=cos_sin, max_seqlen=max_seqlen)


@torch.no_grad()
def evaluate_task(model, tokenizer, data, device, task_meta, token_budget=32768):
    """
    Evaluate one task using varlen-packed forward passes with a fixed token budget.

    Sequences are concatenated (eliminating per-sequence padding waste), then the
    concatenated tensor is right-padded to a fixed budget. cu_seqlens marks real
    sequence boundaries so flash attention never attends to or across padding.

    Handles dispatch to all processes if the script is run with torchrun.
    """
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # For compiled models, access the original model's attributes
    orig = getattr(model, '_orig_mod', model)
    max_seq_len = getattr(orig, 'max_seq_len', None)

    # ---- Phase 1: Pre-process all examples assigned to this rank ----
    # Build a flat list of sequences to forward and metadata to map them back to examples
    sequences = []    # (tokens_list, start_idx, end_idx) per sequence
    example_info = [] # (global_idx, gold, num_seqs, seq_offset) per example
    my_indices = list(range(rank, len(data), world_size))

    for global_idx in my_indices:
        item = data[global_idx]

        # Sample few-shot examples (excluding current item)
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + global_idx)
            available_indices = [i for i in range(len(data)) if i != global_idx]
            fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
            fewshot_examples = [data[i] for i in fewshot_indices]

        # Render prompts and tokenize based on task type
        if task_type == 'multiple_choice':
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        elif task_type == 'schema':
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        elif task_type == 'language_modeling':
            prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Truncate sequences that exceed max_seq_len (e.g. GPT-2 style models)
        if max_seq_len is not None:
            new_tokens, new_start_idxs, new_end_idxs = [], [], []
            for t, s, e in zip(tokens, start_idxs, end_idxs):
                if len(t) > max_seq_len:
                    crop = len(t) - max_seq_len
                    new_tokens.append(t[-max_seq_len:])
                    new_start_idxs.append(s - crop)
                    new_end_idxs.append(e - crop)
                    assert s - crop >= 0
                    assert e - crop >= 0
                else:
                    new_tokens.append(t)
                    new_start_idxs.append(s)
                    new_end_idxs.append(e)
            tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

        # Record this example's sequences
        seq_offset = len(sequences)
        for t, si, ei in zip(tokens, start_idxs, end_idxs):
            sequences.append((t, si, ei))
        example_info.append((global_idx, item.get('gold', None), len(tokens), seq_offset))

    # ---- Phase 2: Forward all sequences using varlen packing ----
    # Pack sequences greedily until total tokens approach the budget, then pad to
    # exactly budget tokens. Every forward call sees (1, token_budget) -- static shape.
    pad_token_id = tokenizer.get_bos_token_id()
    seq_results = [None] * len(sequences)
    sorted_order = sorted(range(len(sequences)), key=lambda i: len(sequences[i][0]))

    # Greedily build varlen batches respecting the total token budget
    batches = []
    cur_batch, cur_total_tokens = [], 0
    for idx in sorted_order:
        seq_len = len(sequences[idx][0])
        if cur_batch and cur_total_tokens + seq_len > token_budget:
            batches.append(cur_batch)
            cur_batch, cur_total_tokens = [idx], seq_len
        else:
            cur_batch.append(idx)
            cur_total_tokens += seq_len
    if cur_batch:
        batches.append(cur_batch)

    # For static torch.compile (dynamic=False): cu_seqlens must have the same size
    # across ALL forward calls (not just within a task, but across tasks too).
    # We pad to a fixed size by repeating the last value, creating zero-length ghost
    # sequences that flash_attn treats as noops. The size is based on token_budget
    # divided by a conservative minimum sequence length estimate.
    max_seqs_per_batch = token_budget // 8 + 1  # conservative: assumes min ~8 tokens/seq
    cu_seqlens_size = max_seqs_per_batch + 1  # cu_seqlens has num_seqs + 1 entries

    for batch_indices in batches:
        # Concatenate all sequences in this batch
        all_tokens = []
        cu_seqlens_list = [0]
        for idx in batch_indices:
            toks = sequences[idx][0]
            all_tokens.extend(toks)
            cu_seqlens_list.append(cu_seqlens_list[-1] + len(toks))

        total_real = len(all_tokens)
        # Pad cu_seqlens to fixed size by repeating the last value (zero-length sequences)
        while len(cu_seqlens_list) < cu_seqlens_size:
            cu_seqlens_list.append(cu_seqlens_list[-1])

        # Pad tokens to fixed budget for static torch.compile shapes
        # If total_real > token_budget (rare edge case), use the actual size
        effective_budget = max(token_budget, total_real)
        pad_count = effective_budget - total_real
        all_tokens.extend([pad_token_id] * pad_count)

        input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)  # (1, budget)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

        # Forward with varlen attention
        outputs = forward_model_varlen(model, input_ids, cu_seqlens)  # (1, budget, vocab)
        logits = outputs[0]  # (budget, vocab)

        # Extract per-sequence losses and predictions
        running_offset = 0
        for j, idx in enumerate(batch_indices):
            toks, si, ei = sequences[idx]
            seq_len = len(toks)

            # For this sequence: logits[running_offset + t] predicts token at position t+1
            # We compute losses and predictions in the shifted autoregressive space
            seq_logits = logits[running_offset:running_offset + seq_len - 1]  # predict tokens[1:]
            seq_targets = input_ids[0, running_offset + 1:running_offset + seq_len]  # tokens[1:]

            losses = torch.nn.functional.cross_entropy(
                seq_logits, seq_targets, reduction='none'
            )
            predictions = seq_logits.argmax(dim=-1)

            # si and ei are indices into the original token sequence
            # In the shifted space (0-indexed): position t predicts token[t+1]
            # losses[si-1] = loss of predicting token[si], same as original forward_model
            if task_type == 'language_modeling':
                predicted = predictions[si-1:ei-1]
                actual = seq_targets[si-1:ei-1]
                seq_results[idx] = torch.all(predicted == actual).item()
            else:  # multiple_choice or schema
                seq_results[idx] = losses[si-1:ei-1].mean().item()

            running_offset += seq_len

    # ---- Phase 3: Aggregate per-example correctness ----
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for global_idx, gold, num_seqs, offset in example_info:
        if task_type == 'language_modeling':
            is_correct = seq_results[offset]
        else:
            choice_losses = [seq_results[offset + c] for c in range(num_seqs)]
            pred_idx = choice_losses.index(min(choice_losses))
            is_correct = (pred_idx == gold)
        correct[global_idx] = float(is_correct)

    # ---- Phase 4: Sync results across ranks if distributed ----
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    mean_correct = correct.mean().item()
    return mean_correct
