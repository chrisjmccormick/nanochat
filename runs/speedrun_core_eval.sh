#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# add uv to PATH (the installer puts it in ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# System dependencies (Python dev headers needed for Triton/torch compilation)

if ! dpkg -s python3-dev &> /dev/null; then
    echo "Installing python3-dev (required for Python.h)..."
    sudo apt-get update && sudo apt-get install -y python3-dev
fi

# # -----------------------------------------------------------------------------
# # wandb setup
# # If you wish to use wandb for logging (it's nice!, recommended).
# # 1) Make sure to first log in to wandb, e.g. run:
# #    `wandb login`
# # 2) Set the WANDB_RUN environment variable when running this script, e.g.:
# #    `WANDB_RUN=d26 bash speedrun.sh`
# if [ -z "$WANDB_RUN" ]; then
#     # by default use "dummy" : it's handled as a special case, skips logging to wandb
#     WANDB_RUN=dummy
# fi


# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
# python -m nanochat.report reset

# # -----------------------------------------------------------------------------
# # Tokenizer

# # Download the first ~2B characters of pretraining dataset
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# python -m nanochat.dataset -n 8
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # Approximately 350 shards are needed for 10B tokens of data for pretraining.
# # The maximum total number of shards available in the entire dataset is 1822.
# python -m nanochat.dataset -n 370 &
# DATASET_DOWNLOAD_PID=$!
# # train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
# python -m scripts.tok_train
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# # -----------------------------------------------------------------------------
# # Base model (pretraining)
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# # d24 model (slightly overtrained is enough to beat GPT-2 => increase data:params ratio from compute optimal 10.5 (default) to 12)

export WANDB_API_KEY=d63437856880292b3bc7e96dd31816c4c48af4b9

for i in {9..11}; do
    WANDB_RUN=$(date +"%Y-%m-%d_%H%M%S")-8xA100-d12-baseline-$i
    python -m nanochat.report reset
    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --target-param-data-ratio=8.5 --device-batch-size=16 --run=$WANDB_RUN 2>&1 | tee -a logs/$WANDB_RUN.log
done

# # evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# # -----------------------------------------------------------------------------
# # SFT (teach the model conversation special tokens, tool use, multiple choice)

# # download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# # see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# # run SFT and eval the model
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# # chat with the model over CLI! Leave out the -p to chat interactively
# # python -m scripts.chat_cli -p "Why is the sky blue?"

# # even better, chat with your model over a pretty WebUI ChatGPT style
# # python -m scripts.chat_web

# # -----------------------------------------------------------------------------
# # Generate the full report by putting together all the sections
# # report.md is the output and will be copied to current directory for convenience
# python -m nanochat.report generate



# Generate timestamp (same format as Python: YYYY-MM-DD_HHMMSS)
# TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

# # Set up logging directory and file
# LOG_DIR="logs/${TIMESTAMP} - ${RUN_NAME}"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/${TIMESTAMP} - ${RUN_NAME}.log"

# # If run name is not "dummy", enable logging to file
# if [ "$RUN_NAME" != "dummy" ]; then
#     echo "Logging to: $LOG_FILE"
#     exec > >(tee -a "$LOG_FILE") 2>&1
# fi

# # Dump the model code at the top of the log for reproducibility
# echo "===== nanochat/gpt.py ====="
# cat ./nanochat/gpt.py
# echo -e "\n===== END nanochat/gpt.py =====\n"
# echo "===== nanochat/optim.py ====="
# cat ./nanochat/optim.py
# echo -e "\n===== END nanochat/optim.py =====\n"
# echo "===== scripts/base_train.py ====="
# cat ./scripts/base_train.py
# echo -e "\n===== END scripts/base_train.py =====\n"

# echo "=== Run: $RUN_NAME (profiled=$PROFILED) ==="
# echo "=== Started: $(date) ==="