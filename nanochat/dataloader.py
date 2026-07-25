"""
Distributed dataloader for pretraining.

Varlen 1D packing:
   - Packs documents into 1D buffer with cu_seqlens for per-document attention isolation
   - No cropping, no padding: every token is used exactly once
   - Yields (inputs_1d, targets_1d, cu_seqlens) for flash_attn_varlen_func

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1



# =============================================================================
# 1D packed varlen dataloader
# =============================================================================
# Packs documents into a single flat buffer of B*T tokens with cu_seqlens marking
# document boundaries for flash_attn_varlen_func. Each document gets its own
# attention context. Greedy packing: documents are added sequentially until the
# buffer is full. Only the last document in each micro-batch gets cropped.
#
# Requires specificying a fixed maximum number of docs supported per batch. 
# The dataloader will append additional documents to the final segment if needed,
# resulting in cross-document attention bleeding, but that hasn't been a problem
# in practice. 
# It's recommended to keep max_num_docs tight rather than padding it conservatively
# because an oversized `cu_seqlens` tensor will hurt FlashAttention performance 
# somewhat.

def tokenizing_distributed_data_loader_varlen(
    tokenizer, B, T, split, max_num_docs,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
):
    """
    1D packed varlen dataloader for use with flash_attn_varlen_func.

    Yields (inputs, targets, cu_seqlens, state_dict) where:
    - inputs: 1D long tensor of shape (B*T,)
    - targets: 1D long tensor of shape (B*T,), shifted by 1
    - cu_seqlens: int32 tensor of shape (max_num_docs,), cumulative doc lengths
      padded with total_tokens for unused slots (ghost segments of length 0)
    - state_dict: {"pq_idx", "rg_idx", "epoch"} for checkpoint resume
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    total_tokens = B * T
    buffer_capacity = total_tokens + 1  # +1 so the last input position has a target

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        doc_buffer.extend(token_lists)

    # Pre-allocate all buffers once
    use_cuda = device == "cuda"
    pack_buffer = torch.empty(buffer_capacity, dtype=torch.long)        # 1D packing workspace
    cpu_buffer = torch.empty(2 * total_tokens, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * total_tokens, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:total_tokens]
    cpu_targets = cpu_buffer[total_tokens:]
    inputs = gpu_buffer[:total_tokens]
    targets = gpu_buffer[total_tokens:]
    cu_seqlens_cpu = torch.empty(max_num_docs, dtype=torch.int32)
    cu_seqlens_gpu = torch.empty(max_num_docs, dtype=torch.int32, device=device)

    warned = False
    warned_seqlen = False
    while True:
        # Greedily pack documents into a single 1D buffer
        pos = 0
        doc_count = 0
        cu_seqlens_cpu[0] = 0

        while pos < buffer_capacity:
            while len(doc_buffer) == 0:
                refill_buffer()

            doc = doc_buffer.pop(0)
            doc_len = min(len(doc), T)             # truncate to max_seq_len
            remaining = buffer_capacity - pos
            use_len = min(doc_len, remaining)      # crop last doc to fill exactly

            pack_buffer[pos:pos + use_len] = torch.tensor(doc[:use_len], dtype=torch.long)
            pos += use_len
            if doc_count < max_num_docs - 1:
                doc_count += 1
                cu_seqlens_cpu[doc_count] = min(pos, total_tokens)
            else:
                if not warned:
                    print(f"Warning: too many documents for cu_seqlens size ({max_num_docs}), "
                          f"merging remaining docs (cross-document attention bleeding)")
                    warned = True
                merged_len = min(pos, total_tokens) - cu_seqlens_cpu[doc_count].item()
                if merged_len > T and not warned_seqlen:
                    print(f"Warning: merged segment length ({merged_len}) exceeds max_seq_len ({T}). "
                          f"Increase max_num_docs to avoid silent attention truncation.")
                    warned_seqlen = True

        # Ensure the final document boundary always points to the end of the batch
        cu_seqlens_cpu[doc_count] = total_tokens

        # Pad remaining cu_seqlens slots (ghost segments of length 0)
        cu_seqlens_cpu[doc_count + 1:] = total_tokens

        # Split into inputs/targets (standard next-token prediction shift)
        cpu_inputs.copy_(pack_buffer[:total_tokens])
        cpu_targets.copy_(pack_buffer[1:total_tokens + 1])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # H2D transfer: single copy for tokens, small copy for cu_seqlens
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        cu_seqlens_gpu.copy_(cu_seqlens_cpu, non_blocking=use_cuda)
        yield inputs, targets, cu_seqlens_gpu, state_dict


# =============================================================================
# SFT varlen dataloader (replay from pre-packed batch plans)
# =============================================================================

def sft_data_loader_varlen(
    conversations, batch_plan, B, T, max_num_docs, bos_token,
    device="cuda", cycle=False,
):
    """
    Replay dataloader for SFT: constructs 1D-packed varlen batches from
    pre-computed batch plans (see tokenize_and_pack_sft in chat_sft.py).

    Args:
        conversations: list of (ids, mask) tuples (pre-tokenized)
        batch_plan: list of lists of conversation indices
        B, T: batch dimensions (total_tokens = B * T)
        max_num_docs: cu_seqlens tensor size (exact max from pre-packing)
        bos_token: BOS token id for padding
        device: target device
        cycle: if True, repeat the batch plan indefinitely (for val eval)
    """
    total_tokens = B * T
    buffer_capacity = total_tokens + 1
    use_cuda = torch.device(device).type == "cuda"

    pack_buffer = torch.empty(buffer_capacity, dtype=torch.long)
    mask_buffer = torch.empty(buffer_capacity, dtype=torch.int8)
    cpu_buffer = torch.empty(2 * total_tokens, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * total_tokens, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:total_tokens]
    cpu_targets = cpu_buffer[total_tokens:]
    inputs = gpu_buffer[:total_tokens]
    targets = gpu_buffer[total_tokens:]
    cu_seqlens_cpu = torch.empty(max_num_docs, dtype=torch.int32)
    cu_seqlens_gpu = torch.empty(max_num_docs, dtype=torch.int32, device=device)

    while True:
        for conv_indices in batch_plan:
            pos = 0
            doc_count = 0
            cu_seqlens_cpu[0] = 0

            for conv_idx in conv_indices:
                ids, mask = conversations[conv_idx]
                conv_len = len(ids)
                pack_buffer[pos:pos + conv_len] = torch.tensor(ids, dtype=torch.long)
                mask_buffer[pos:pos + conv_len] = torch.tensor(mask, dtype=torch.int8)
                pos += conv_len
                doc_count += 1
                cu_seqlens_cpu[doc_count] = min(pos, total_tokens)

            if pos < buffer_capacity:
                remaining = buffer_capacity - pos
                pack_buffer[pos:pos + remaining] = bos_token
                mask_buffer[pos:pos + remaining] = 0
                doc_count += 1
                cu_seqlens_cpu[doc_count] = total_tokens

            cu_seqlens_cpu[doc_count + 1:] = total_tokens

            cpu_inputs.copy_(pack_buffer[:total_tokens])
            cpu_targets.copy_(pack_buffer[1:total_tokens + 1])
            target_mask = mask_buffer[1:total_tokens + 1]
            cpu_targets[target_mask == 0] = -1

            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            cu_seqlens_gpu.copy_(cu_seqlens_cpu, non_blocking=use_cuda)
            yield inputs, targets, cu_seqlens_gpu

        if not cycle:
            break

# =============================================================================
# Reinforce pack dataloader
# =============================================================================

class ReinforcePack:
    __slots__ = ("input_ids", "cu_seqlens", "targets", "comp_mask",
                 "adv_tok", "n_seqs", "n_comp_targets")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_STAGE_BUFS: dict[str, torch.Tensor] = {}


def _stage(name, n, dtype, pin):
    """Grow-only staging buffer (pinned when pin), returned as an n-length
    view. Capacity rounds up to the next 64k elements so round-to-round size
    jitter never reallocates."""
    buf = _STAGE_BUFS.get(name)
    if buf is None or buf.numel() < n or buf.is_pinned() != pin:
        cap_n = -(-n // 65536) * 65536
        buf = torch.empty(cap_n, dtype=dtype, pin_memory=pin)
        _STAGE_BUFS[name] = buf
    return buf[:n]


def build_reinforce_packs(docs, *, buckets, max_num_docs, pad_id, max_doc_len,
                          device="cuda"):
    """FFD bin-pack ``docs`` = (prompt_ids, completion_ids, advantage) into
    fixed-shape packs; each bin sealed at the smallest bucket >= its fill. The
    pad tail is emitted as benign varying-ids segments of <= max_doc_len each
    (FA-varlen NaN doctrine; nanochat's forward passes max_seqlen=sequence_len,
    so no packed segment may exceed it).

    Assembly is numpy over ONE pinned host buffer per field for all packs, then
    one async H2D copy per field; the returned packs are contiguous views into
    the device buffers (identical shape/stride/dtype to standalone tensors, so
    compiled-consumer guards are unaffected). The naive per-doc torch build
    cost ~30 ms/round in tiny host ops + 5 pageable copies per pack."""
    buckets = sorted(buckets)
    cap = buckets[-1]
    stats = {"n_docs": len(docs), "n_packs": 0, "pad_tokens": 0, "comp_targets": 0,
             "cap_tokens": 0}  # sum of sealed bucket sizes -> pad% = pad/cap
    if not docs:
        return [], stats
    lens = [(len(p), len(c)) for p, c, _ in docs]
    L_max = max(p + c for p, c in lens)
    assert L_max <= cap, f"doc {L_max} tok > max bucket {cap} — raise TRAIN_BUCKETS"
    assert L_max <= max_doc_len, f"doc {L_max} tok > max_seqlen {max_doc_len}"

    order = sorted(range(len(docs)), key=lambda i: lens[i][0] + lens[i][1],
                   reverse=True)
    bins: list[dict] = []
    for di in order:
        L = lens[di][0] + lens[di][1]
        for b in bins:
            if b["used"] + L <= cap and len(b["items"]) < max_num_docs:
                b["used"] += L; b["items"].append(di); break
        else:
            bins.append({"used": L, "items": [di]})

    max_pad_segs = -(-cap // max_doc_len) + 1
    W = max_num_docs + max_pad_segs + 2
    T_packs = [next(x for x in buckets if x >= b["used"]) for b in bins]
    bases = np.concatenate([[0], np.cumsum(T_packs)])
    total_T = int(bases[-1])

    # Persistent pinned staging, grown on demand and numpy-filled: a fresh
    # torch.full/zeros costs ~7 ms per ~1 MB host tensor (vs 0.03 ms for a
    # numpy fill of a reused buffer), and pinning is what lets the H2D copies
    # go async. Reuse is safe because the caller consumes each round's packs
    # under the same stream and syncs before the next build (train_step ends
    # with torch.cuda.synchronize()).
    pin = device != "cpu" and torch.cuda.is_available()
    ids_h = _stage("ids", total_T, torch.long, pin)
    tgt_h = _stage("tgt", total_T, torch.long, pin)
    comp_h = _stage("comp", total_T, torch.float32, pin)
    adv_h = _stage("adv", total_T, torch.float32, pin)
    cu_h = _stage("cu", len(bins) * W, torch.int32, pin).view(len(bins), W)
    ids_np, tgt_np = ids_h.numpy(), tgt_h.numpy()
    cu_np = cu_h.numpy()
    comp_h.numpy()[:] = 0.0
    adv_h.numpy()[:] = 0.0
    cu_np[:, 0] = 0
    # ids needs no pre-fill (docs + arange pad tail cover every position);
    # targets' pad tails are filled per pack below.

    # Ragged completion-region index build (comp/adv fancy-write), global
    # across packs: starts/lengths/advantages collected per doc below.
    c_starts, c_lens, c_advs = [], [], []

    n_comp_packs = []
    for p, (b, T_pack) in enumerate(zip(bins, T_packs)):
        B, used, items = int(bases[p]), b["used"], b["items"]
        stats["cap_tokens"] += T_pack
        flat: list[int] = []
        n_comp_in_pack = 0
        doc_Ls = np.empty(len(items), dtype=np.int64)
        for si, di in enumerate(items):
            p_ids, c_ids, a = docs[di]
            flat.extend(p_ids)
            flat.extend(c_ids)
            p_len, c_len = lens[di]
            doc_Ls[si] = p_len + c_len
            if c_len > 0:
                c_starts.append(B + len(flat) - c_len - 1)   # B + off + p_len - 1
                c_lens.append(c_len)
                c_advs.append(float(a))
                n_comp_in_pack += c_len
        ends = np.cumsum(doc_Ls)                    # pack-local doc end offsets
        ids_np[B:B + used] = flat                   # ONE conversion per pack
        # targets = next token: a global shift is correct inside each doc; the
        # boundary position (last token of each doc) then resets to pad, which
        # also erases the cross-doc leak the shift wrote there.
        tgt_np[B:B + used - 1] = ids_np[B + 1:B + used]
        tgt_np[B + ends - 1] = pad_id
        cu_np[p, 1:len(items) + 1] = ends
        if used < T_pack:                           # benign pad segments
            ids_np[B + used:B + T_pack] = np.arange(T_pack - used) % 4096 + 1
            tgt_np[B + used:B + T_pack] = pad_id    # staging reuse: clear stale tail
            n_segs = -(-(T_pack - used) // max_doc_len)
            seg_ends = np.minimum(used + max_doc_len * np.arange(1, n_segs + 1),
                                  T_pack)
            cu_np[p, len(items) + 1:len(items) + 1 + n_segs] = seg_ends
            cu_np[p, len(items) + 1 + n_segs:] = T_pack
        else:
            cu_np[p, len(items) + 1:] = T_pack
        stats["pad_tokens"] += T_pack - used
        n_comp_packs.append(n_comp_in_pack)

    if c_starts:
        cl = np.asarray(c_lens)
        idx = (np.repeat(np.asarray(c_starts), cl) + np.arange(cl.sum())
               - np.repeat(np.cumsum(cl) - cl, cl))
        comp_h.numpy()[idx] = 1.0
        adv_h.numpy()[idx] = np.repeat(np.asarray(c_advs, dtype=np.float32), cl)
        stats["comp_targets"] = int(cl.sum())

    # One async DMA per field for the whole round (same-stream ordering makes
    # the views safe to consume immediately). copy=True because .to("cpu") on
    # a CPU tensor would otherwise return the staging buffer itself, aliasing
    # the next call's writes into these packs.
    ids_d = ids_h.to(device, non_blocking=True, copy=True)
    tgt_d = tgt_h.to(device, non_blocking=True, copy=True)
    comp_d = comp_h.to(device, non_blocking=True, copy=True)
    adv_d = adv_h.to(device, non_blocking=True, copy=True)
    cu_d = cu_h.to(device, non_blocking=True, copy=True)

    out = [ReinforcePack(
        input_ids=ids_d[int(bases[p]):int(bases[p]) + T_packs[p]],
        cu_seqlens=cu_d[p],
        targets=tgt_d[int(bases[p]):int(bases[p]) + T_packs[p]],
        comp_mask=comp_d[int(bases[p]):int(bases[p]) + T_packs[p]],
        adv_tok=adv_d[int(bases[p]):int(bases[p]) + T_packs[p]],
        n_seqs=len(b["items"]), n_comp_targets=n_comp_packs[p])
        for p, b in enumerate(bins)]
    stats["n_packs"] = len(out)
    return out, stats


# =============================================================================
# Balanced round assembly (static-prefill RL)
# =============================================================================

def assemble_balanced_rounds(items, ppr, *, epochs=1):
    """Partition problem indices into rounds of exactly ``ppr`` each, balancing
    the per-round context-token sum so no round is pathologically long — a long
    round would force an oversized static-prefill graph, since PrefillAllEngine
    prefills the whole round in one replay (Sigma context <= PREFILL_T).

    ``items``: list of ``(problem_index, context_len)``, where ``context_len`` is
    the prefill length the problem contributes — the engine prefills
    ``prompt[:-1]``, so ``len(prompt) - 1``.

    Balancing is stratified: sort by ``context_len``, cut the sorted list into
    ``ppr`` contiguous strata, and give every round one problem from each
    stratum. Each round then spans the full length range, so per-round sums
    cluster tightly around the mean (max round ~= mean, not ~= ppr * longest).
    Deterministic; each problem appears once per epoch; the trailing ``< ppr``
    remainder is dropped (varlen doctrine: no short final batch).

    Returns ``(rounds, stats)`` — ``rounds`` is a list of length
    ``(len(items) // ppr) * epochs`` of ``ppr``-length index lists; ``stats`` has
    the ``min`` / ``mean`` / ``max`` per-round context-token sums. Report those so
    the caller can set PREFILL_T explicitly — this function never sizes it and
    the engine never auto-sizes."""
    r = len(items) // ppr                         # rounds per epoch (drop remainder)
    if r == 0:
        return [], {"min": 0, "mean": 0.0, "max": 0}
    order = sorted(items[:r * ppr], key=lambda t: t[1])
    epoch_rounds = [[] for _ in range(r)]
    for j, (pid, _clen) in enumerate(order):
        _stratum, within = divmod(j, r)           # one problem per stratum -> balanced
        epoch_rounds[within].append(pid)
    clen = dict(items)
    sums = [sum(clen[pid] for pid in rd) for rd in epoch_rounds]
    stats = {"min": min(sums), "mean": sum(sums) / len(sums), "max": max(sums)}
    rounds = [list(rd) for _ in range(epochs) for rd in epoch_rounds]
    return rounds, stats
