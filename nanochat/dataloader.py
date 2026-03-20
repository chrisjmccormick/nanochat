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

def tokenizing_distributed_data_loader_with_state_varlen(
    tokenizer, B, T, split,
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

    # Fixed cu_seqlens size for torch.compile(dynamic=False). Must be large enough for
    # the maximum number of documents that could fit in one micro-batch. 
    max_num_docs = ((total_tokens // 400) + 15) // 16 * 16  # --> Yields 96 docs at 32K tokens.

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
            doc_count += 1
            assert doc_count < max_num_docs, \
                f"Too many documents ({doc_count}) in micro-batch for cu_seqlens " \
                f"size ({max_num_docs}). Need a smaller divisor in max_num_docs."
            cu_seqlens_cpu[doc_count] = min(pos, total_tokens)

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


def tokenizing_distributed_data_loader_varlen(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, cu_seqlens, state_dict in tokenizing_distributed_data_loader_with_state_varlen(*args, **kwargs):
        yield inputs, targets, cu_seqlens
