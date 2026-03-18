"""
Document length analysis for varlen flash attention migration.

Tokenizes documents from the first ~8 shards of ClimbMix (~2B chars, same
subset used for tokenizer training) and reports length statistics needed to
choose max_num_docs for 1D packed varlen attention.

Usage:
    python -m scripts.doc_length_analysis
    python -m scripts.doc_length_analysis --max-chars 500_000_000  # smaller run
    python -m scripts.doc_length_analysis --num-shards 2           # download only 2 shards
"""

import os
import math
import time
import pickle
import argparse
import numpy as np
import pyarrow.parquet as pq
import tiktoken

# ---- Minimal reimplementations to avoid importing nanochat (which pulls torch) ----

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

def _get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        d = os.environ["NANOCHAT_BASE_DIR"]
    else:
        d = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(d, exist_ok=True)
    return d

def _data_dir():
    d = os.path.join(_get_base_dir(), "base_data_climbmix")
    os.makedirs(d, exist_ok=True)
    return d

def _download_shard(index, data_dir):
    """Download a single shard if not present."""
    import requests
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {filename}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    tmp = filepath + ".tmp"
    with open(tmp, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.rename(tmp, filepath)
    return filepath

def _ensure_shards(n_shards):
    """Make sure at least n_shards are downloaded. Returns sorted parquet paths."""
    data_dir = _data_dir()
    for i in range(n_shards):
        _download_shard(i, data_dir)
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith('.parquet') and not f.endswith('.tmp'))
    return [os.path.join(data_dir, f) for f in files]

def _load_tokenizer():
    """Try nanochat's trained tokenizer; fall back to cl100k_base."""
    tok_dir = os.path.join(_get_base_dir(), "tokenizer")
    pkl_path = os.path.join(tok_dir, "tokenizer.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        print(f"Using nanochat tokenizer from {pkl_path}")
        bos_id = enc.encode_single_token("<|bos|>")
        return enc, bos_id, "nanochat (32K vocab)"
    # Fallback: cl100k_base (~100K vocab, GPT-4). Doc length distribution
    # will be slightly compressed vs nanochat's 32K vocab, but the shape
    # (min/median/p95/p99 ratios) is representative.
    print("nanochat tokenizer not found, falling back to cl100k_base (GPT-4)")
    enc = tiktoken.get_encoding("cl100k_base")
    bos_id = None  # no BOS prepend for the fallback
    return enc, bos_id, "cl100k_base (100K vocab, fallback)"

# ---- Main ----

parser = argparse.ArgumentParser(description="Analyze document lengths for varlen packing")
parser.add_argument("--max-chars", type=int, default=2_000_000_000,
                    help="Character budget (default: 2B, same as tokenizer training)")
parser.add_argument("--num-shards", type=int, default=8,
                    help="Number of shards to ensure are downloaded (default: 8)")
parser.add_argument("--doc-cap", type=int, default=100_000,
                    help="Max characters per document before tokenizing (default: 100K)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="Documents per encode() call")
parser.add_argument("--threads", type=int, default=8,
                    help="Threads for tiktoken batch encoding")
args = parser.parse_args()

enc, bos_id, tokenizer_name = _load_tokenizer()
bos_extra = 1 if bos_id is not None else 0

print(f"Tokenizer: {tokenizer_name}")
print(f"Ensuring {args.num_shards} shards are downloaded...")
parquet_paths = _ensure_shards(args.num_shards)
print(f"Found {len(parquet_paths)} shards on disk")
print(f"Character budget: {args.max_chars:,}")
print(f"Document cap: {args.doc_cap:,} chars")
print()

# ---- Tokenize and collect lengths ----
doc_lengths = []
nchars = 0
ndocs = 0
t0 = time.time()

for shard_idx, filepath in enumerate(parquet_paths):
    if nchars >= args.max_chars:
        break
    pf = pq.ParquetFile(filepath)
    shard_name = os.path.basename(filepath)
    for rg_idx in range(pf.num_row_groups):
        if nchars >= args.max_chars:
            break
        rg = pf.read_row_group(rg_idx)
        texts = rg.column('text').to_pylist()

        batch = []
        for doc in texts:
            if len(doc) > args.doc_cap:
                doc = doc[:args.doc_cap]
            nchars += len(doc)
            batch.append(doc)
            if len(batch) >= args.batch_size:
                token_lists = enc.encode_ordinary_batch(batch, num_threads=args.threads)
                doc_lengths.extend(len(toks) + bos_extra for toks in token_lists)
                ndocs += len(batch)
                batch = []
                if nchars >= args.max_chars:
                    break

        if batch:
            token_lists = enc.encode_ordinary_batch(batch, num_threads=args.threads)
            doc_lengths.extend(len(toks) + bos_extra for toks in token_lists)
            ndocs += len(batch)

    elapsed = time.time() - t0
    print(f"  Shard {shard_idx:4d} ({shard_name})  |  "
          f"docs: {ndocs:>9,}  |  chars: {nchars:>13,}  |  "
          f"elapsed: {elapsed:.1f}s")

elapsed = time.time() - t0
print()
print(f"Tokenized {ndocs:,} documents ({nchars:,} chars) in {elapsed:.1f}s")
print()

# ---- Statistics ----
lengths = np.array(doc_lengths, dtype=np.int64)
total_tokens = int(lengths.sum())

print("=" * 70)
print(f"DOCUMENT LENGTH STATISTICS (in tokens, tokenizer: {tokenizer_name})")
print("=" * 70)
print(f"  Documents analyzed:  {len(lengths):>12,}")
print(f"  Total tokens:        {total_tokens:>12,}")
print(f"  Min:                 {int(lengths.min()):>12,}")
print(f"  Max:                 {int(lengths.max()):>12,}")
print(f"  Mean:                {lengths.mean():>12,.1f}")
print(f"  Median:              {int(np.median(lengths)):>12,}")
print(f"  Std:                 {lengths.std():>12,.1f}")
print(f"  p1:                  {int(np.percentile(lengths, 1)):>12,}")
print(f"  p5:                  {int(np.percentile(lengths, 5)):>12,}")
print(f"  p10:                 {int(np.percentile(lengths, 10)):>12,}")
print(f"  p25:                 {int(np.percentile(lengths, 25)):>12,}")
print(f"  p75:                 {int(np.percentile(lengths, 75)):>12,}")
print(f"  p90:                 {int(np.percentile(lengths, 90)):>12,}")
print(f"  p95:                 {int(np.percentile(lengths, 95)):>12,}")
print(f"  p99:                 {int(np.percentile(lengths, 99)):>12,}")
print(f"  p99.9:               {int(np.percentile(lengths, 99.9)):>12,}")
print()

# ---- Length distribution buckets ----
buckets = [1, 10, 50, 100, 200, 500, 1000, 2048, 4096, 8192, 16384, 32768, 65536]
print("LENGTH DISTRIBUTION (cumulative)")
print("-" * 50)
for b in buckets:
    count = int(np.sum(lengths <= b))
    pct = 100.0 * count / len(lengths)
    print(f"  <= {b:>6,} tokens:  {count:>10,}  ({pct:5.1f}%)")
print()

# ---- Buffer analysis for various (B, T) configs ----
def ceil_to_128(x):
    return int(math.ceil(x / 128)) * 128

print("=" * 70)
print("BUFFER PACKING ANALYSIS")
print("=" * 70)
print()

T = 2048
batch_sizes = [4, 8, 16, 32, 64]
median_len = int(np.median(lengths))
mean_len = lengths.mean()
p1_len = int(np.percentile(lengths, 1))
p5_len = int(np.percentile(lengths, 5))

print(f"  Sequence length T = {T}")
print(f"  Median doc length = {median_len}")
print(f"  Mean doc length   = {mean_len:.0f}")
print(f"  p5 doc length     = {p5_len}")
print(f"  p1 doc length     = {p1_len}")
print()

header = f"{'B':>4}  {'B*T':>8}  {'mean docs':>10}  {'max docs (median)':>18}  {'max_num_docs':>13}  {'decoderstack //300':>19}"
print(header)
print("-" * len(header))

for B in batch_sizes:
    buffer_size = B * T
    expected_docs_mean = buffer_size / mean_len
    max_docs_median = buffer_size / median_len
    conservative = max(p5_len, 1)
    recommended = ceil_to_128(buffer_size // conservative)
    decoderstack_style = ceil_to_128(buffer_size // 300)
    print(f"{B:>4}  {buffer_size:>8,}  {expected_docs_mean:>10.1f}  {max_docs_median:>18.1f}  "
          f"{recommended:>13,}  {decoderstack_style:>19,}")

print()
print("NOTE: max_num_docs should be >= the maximum number of documents that")
print("could ever fit in a buffer. Using p5 doc length as divisor is very")
print("conservative. If median is close to decoderstack's assumption of ~400,")
print("then buffer_size // 300 (rounded up to 128) should work.")
print()

# ---- Short doc analysis (relevant for max_num_docs overflow) ----
short_threshold = 50
n_short = int(np.sum(lengths <= short_threshold))
pct_short = 100.0 * n_short / len(lengths)
print(f"Documents <= {short_threshold} tokens: {n_short:,} ({pct_short:.2f}%)")
if p1_len > 0:
    print(f"  Worst-case buffer fill (p1={p1_len}): B*T/p1 = {batch_sizes[-1]*T}/{p1_len} = {batch_sizes[-1]*T//p1_len} docs")
print()

# ---- Recommendation ----
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
for B in batch_sizes:
    buffer_size = B * T
    decoderstack_val = ceil_to_128(buffer_size // 300)
    conservative_val = ceil_to_128(buffer_size // max(p5_len, 1))
    final = max(decoderstack_val, conservative_val)
    print(f"  B={B:>2}, buffer={buffer_size:>6,}:  max_num_docs = {final:>5,}  "
          f"(decoderstack={decoderstack_val}, conservative={conservative_val})")
print()
print("Use: max_num_docs = ceil_to_128(buffer_size // MIN_DOC_ESTIMATE)")
print(f"where MIN_DOC_ESTIMATE should be validated from the p5 value above ({p5_len}).")
