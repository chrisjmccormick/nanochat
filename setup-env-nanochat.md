# Agent: Set up environment for nanochat

Follow these instructions to set up the environment for running nanochat on the current GPU instance.

## 1. Load API keys and secrets

```bash
source ~/env.sh
```

## 2. Configure Git and clone repos

Set Git identity (required for commits and for private clone via token):

```bash
git config --global user.name "$GH_YOUR_NAME"
git config --global user.email "$GH_EMAIL"
```

Clone my nanochat repo and set the remote URL so pushes work with the token:

```bash
cd ~/
git clone "https://${GITHUB_TOKEN}@github.com/chrisjmccormick/nanochat.git" && cd nanochat
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/chrisjmccormick/nanochat.git"
```

Clone **Karpathy's** nanochat as a baseline (read-only, no token needed):

```bash
cd ~/
git clone https://github.com/karpathy/nanochat.git nanochat-baseline
```

## 3. Install system dependencies

Install Python development headers (required for Triton/CUDA compilation):

```bash
# Detect Python version and install matching dev package
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
apt-get update
apt-get install -y python3-dev "python${PYTHON_VERSION}-dev"
```

This detects the installed Python version (e.g., 3.10, 3.11, 3.12) and installs the corresponding development headers package.

## 4. Python environment with uv (in your nanochat repo)

From `~/nanochat`, use the same steps as in `runs/speedrun.sh`:

```bash
cd ~/nanochat

# Install uv if not present
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure PATH includes uv (installer may add it to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment if it doesn't exist
[ -d ".venv" ] || uv venv

# Install project dependencies including GPU extras
uv sync --extra gpu
```

**Activate the environment** (do this in every new shell where you want to run nanochat):

```bash
cd ~/nanochat
source .venv/bin/activate
```

Then `python` and installed tools (e.g. `torchrun`, `wandb`) use the project venv.

## 5. Speedrun Setup

Perform some of the additional prep steps for executing ~/nanochat/runs/speedrun.sh

Environment variables:

```bash
# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
```

Dataset download and tokenizer:

```bash
# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID
```

## 6. Instruct Me

Tell me to do:

```bash
source ~/env.sh
cd ~/nanochat && source .venv/bin/activate
```
