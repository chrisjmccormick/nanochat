#!/bin/bash

# Pretrain a d24 model, SFT it, and run chat evals (excluding GSM8K).
# Assumes the setup in ~/setup-env-nanochat.md has been completed:
#   - venv created & deps installed (uv sync --extra gpu)
#   - dataset downloaded (170 shards) and tokenizer trained
#   - nanochat.report reset already run
#
# Usage:
#   cd ~/nanochat && source .venv/bin/activate
#   bash runs/d24_speedrun.sh
#
# With wandb:
#   WANDB_RUN=d24-speedrun bash runs/d24_speedrun.sh

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="d24_speedrun"
LOGDIR="./logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/d24_speedrun_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== d24 speedrun started at $(date) ==="
echo "Log file: $LOGFILE"

# ---------------------------------------------------------------------------
# 1. Base model pretraining
echo "=== [1/4] base_train ==="
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8 \
    --fp8

# ---------------------------------------------------------------------------
# 2. Base model evaluation
echo "=== [2/4] base_eval ==="
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=16

# ---------------------------------------------------------------------------
# 3. SFT
echo "=== [3/4] chat_sft ==="
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run="$WANDB_RUN"

# ---------------------------------------------------------------------------
# 4. Chat evaluation (all tasks except GSM8K)
echo "=== [4/4] chat_eval ==="
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- \
    -i sft \
    -a "ARC-Easy|ARC-Challenge|MMLU|HumanEval|SpellingBee"

echo "=== d24 speedrun finished at $(date) ==="
echo "Full log saved to: $LOGFILE"
