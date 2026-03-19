"""
Profile the varlen dataloader to capture document count and max sequence length
statistics across many batches.

Usage: python -m scripts.profile_dataloader [--num-shards 10] [--device-batch-size 16]
"""

import argparse
import time
from collections import Counter, deque

import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from nanochat.dataset import list_parquet_files

parser = argparse.ArgumentParser()
parser.add_argument("--num-shards", type=int, default=10)
parser.add_argument("--device-batch-size", "-B", type=int, default=16)
parser.add_argument("--max-seq-len", "-T", type=int, default=2048)
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()

B, T = args.device_batch_size, args.max_seq_len
total_tokens = B * T
buffer_capacity = total_tokens + 1

tokenizer = get_tokenizer()
bos_token = tokenizer.get_bos_token_id()

parquet_paths = list_parquet_files(warn_on_legacy=False)[:-1]  # exclude val shard
num_shards = min(args.num_shards, len(parquet_paths))
print(f"Profiling {num_shards} shards | B={B} T={T} | total_tokens={total_tokens:,} | threads={args.threads}")

# Pre-tokenize all shards and collect just the doc lengths
t0 = time.time()
doc_lengths = deque()  # just integers, no tensor overhead

for shard_idx in range(num_shards):
    st = time.time()
    pf = pq.ParquetFile(parquet_paths[shard_idx])
    shard_docs = 0
    for rg_idx in range(pf.num_row_groups):
        texts = pf.read_row_group(rg_idx).column('text').to_pylist()
        token_lists = tokenizer.encode(texts, prepend=bos_token, num_threads=args.threads)
        for toks in token_lists:
            doc_lengths.append(len(toks))
        shard_docs += len(texts)
    elapsed = time.time() - st
    print(f"  shard {shard_idx+1}/{num_shards}: {shard_docs:,} docs tokenized in {elapsed:.1f}s ({len(doc_lengths):,} total)")

print(f"Tokenization done: {len(doc_lengths):,} documents in {time.time()-t0:.1f}s")

# Simulate greedy packing using just lengths
max_doc_count = 0
max_seq_in_batch = 0
batch_count = 0
doc_count_counter = Counter()
max_seq_counter = Counter()

t1 = time.time()
while doc_lengths:
    pos = 0
    doc_count = 0
    batch_max_seq = 0

    while pos < buffer_capacity:
        if not doc_lengths:
            break
        raw_len = doc_lengths.popleft()
        doc_len = min(raw_len, T)
        remaining = buffer_capacity - pos
        use_len = min(doc_len, remaining)

        if use_len > batch_max_seq:
            batch_max_seq = use_len
        pos += use_len
        doc_count += 1

    if pos < buffer_capacity:
        break  # ran out of documents mid-batch

    if doc_count > max_doc_count:
        max_doc_count = doc_count
    if batch_max_seq > max_seq_in_batch:
        max_seq_in_batch = batch_max_seq

    doc_count_counter[doc_count] += 1
    seq_bucket = ((batch_max_seq + 127) // 128) * 128
    max_seq_counter[seq_bucket] += 1
    batch_count += 1

    if batch_count % 50000 == 0:
        print(f"  batch {batch_count:,} | max_docs={max_doc_count} | max_seq={max_seq_in_batch} | {time.time()-t1:.1f}s")

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"Profiled {batch_count:,} batches over {num_shards} shards in {elapsed:.1f}s")
print(f"Config: B={B}, T={T}, total_tokens={total_tokens:,}")
print(f"\n--- Document count per micro-batch ---")
print(f"  Max doc count seen: {max_doc_count}")
print(f"  Distribution:")
for k in sorted(doc_count_counter.keys()):
    pct = 100 * doc_count_counter[k] / batch_count
    bar = "#" * max(1, int(pct / 2))
    print(f"    {k:4d} docs: {doc_count_counter[k]:6d} ({pct:5.1f}%) {bar}")

print(f"\n--- Max sequence length in micro-batch ---")
print(f"  Max seq length seen: {max_seq_in_batch}")
print(f"  Distribution (bucketed to 128):")
for k in sorted(max_seq_counter.keys()):
    pct = 100 * max_seq_counter[k] / batch_count
    bar = "#" * max(1, int(pct / 2))
    print(f"    <={k:4d}: {max_seq_counter[k]:6d} ({pct:5.1f}%) {bar}")

for threshold in [256, 512, 768, 1024, 1536]:
    count_below = sum(v for k, v in max_seq_counter.items() if k <= threshold)
    print(f"  Batches with max_seq <= {threshold:4d}: {count_below:6d} / {batch_count:,} ({100*count_below/batch_count:.1f}%)")
