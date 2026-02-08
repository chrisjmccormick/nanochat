"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
from nanochat.core_eval_optim import evaluate_task as evaluate_task_optim
from nanochat.core_eval_varlen import evaluate_task as evaluate_task_varlen
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
if not use_dummy_wandb:
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

# Will be populated after model is initialized and used to update wandb config
training_setup_stats = {}

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            # FP8 requires both in_features and out_features divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Determine the optimization horizon based on the model size
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params

# Get the parameter counts of the model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Scaling params: transformer matrices + lm_head (gives cleanest scaling laws, see dev/LOG.md Jan 27, 2026)
get_scaling_params = lambda m: m.num_scaling_params()['transformer_matrices'] + m.num_scaling_params()['lm_head']
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Auto-compute optimal batch size based on Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    d12_ref = build_model_meta(12) # d12 is where the optimal batch size was measured to be 2**19 tokens
    d12_num_scaling_params = get_scaling_params(d12_ref)
    D_REF = args.target_param_data_ratio * d12_num_scaling_params
    B_REF = 2**19
    batch_size_ratio = target_tokens / D_REF
    total_batch_size = 2 ** round(math.log2(B_REF * batch_size_ratio ** 0.383)) # also clamp to power of 2
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {reference_batch_size:,})")

# Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
weight_decay_scaled = args.weight_decay * (12 / args.depth)**2
if args.depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# Populate training setup stats for wandb config
training_setup_stats = {
    "Number of parameters": num_params,
    "Number of FLOPs per token": f"{num_flops_per_token:e}",
    "Calculated number of iterations": num_iterations,
    "Number of training tokens": total_tokens,
    "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
    "DDP world size": ddp_world_size,
    "warmup_ratio": args.warmup_ratio,
    "warmdown_ratio": args.warmdown_ratio,
    "final_lr_frac": args.final_lr_frac,
}

# Update wandb config with training setup stats
if not use_dummy_wandb:
    wandb_run.config.update(training_setup_stats)

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# GC event monitoring — track time spent in garbage collection per rank
gc_total_time = 0.0          # accumulated GC wall-clock time (seconds)
gc_event_count = 0           # number of GC collections observed
gc_tracking_enabled = False  # only start recording after the first step completes

def _gc_callback(phase, info):
    """gc.callbacks hook: measure wall-clock time of every GC collection."""
    global gc_total_time, gc_event_count
    if not gc_tracking_enabled:
        return
    if phase == "start":
        _gc_callback._t0 = time.perf_counter()
    elif phase == "stop":
        t0 = getattr(_gc_callback, "_t0", None)
        if t0 is not None:
            elapsed = time.perf_counter() - t0
            gc_total_time += elapsed
            gc_event_count += 1
            print(f"[rank {ddp_rank}] GC event #{gc_event_count}: gen {info['generation']}, "
                  f"collected {info['collected']} objects in {elapsed*1000:.2f}ms "
                  f"(cumulative gc: {gc_total_time*1000:.2f}ms)")
            _gc_callback._t0 = None

gc.callbacks.append(_gc_callback)

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
        gc_tracking_enabled = True  # start recording GC events from now on
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# -----------------------------------------------------------------------------
# GC stats: gather per-rank totals and log to wandb
gc.callbacks.remove(_gc_callback)
print(f"[rank {ddp_rank}] GC summary: {gc_total_time*1000:.2f}ms across {gc_event_count} events")
gc_metrics = {}
if ddp:
    import torch.distributed as dist
    gc_time_tensor = torch.tensor([gc_total_time], dtype=torch.float64, device=device)
    gc_count_tensor = torch.tensor([gc_event_count], dtype=torch.long, device=device)
    gc_all_times = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(ddp_world_size)]
    gc_all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(ddp_world_size)]
    dist.all_gather(gc_all_times, gc_time_tensor)
    dist.all_gather(gc_all_counts, gc_count_tensor)
    if master_process:
        print0("--- GC Statistics (per rank) ---")
        for r in range(ddp_world_size):
            t_ms = gc_all_times[r].item() * 1000
            cnt = int(gc_all_counts[r].item())
            print0(f"  rank {r}: {t_ms:.2f}ms across {cnt} GC events")
            gc_metrics[f"gc/rank{r}_total_time_ms"] = t_ms
            gc_metrics[f"gc/rank{r}_event_count"] = cnt
        gc_metrics["gc/max_rank_time_ms"] = max(t.item() for t in gc_all_times) * 1000
        gc_metrics["gc/total_time_ms"] = sum(t.item() for t in gc_all_times) * 1000
else:
    gc_metrics["gc/rank0_total_time_ms"] = gc_total_time * 1000
    gc_metrics["gc/rank0_event_count"] = gc_event_count
    gc_metrics["gc/max_rank_time_ms"] = gc_total_time * 1000
    gc_metrics["gc/total_time_ms"] = gc_total_time * 1000
wandb_run.log(gc_metrics)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Run final CORE evaluation on the trained model (full evaluation, not limited)
print0("\n" + "="*80)
print0("Running final CORE evaluation on trained model")
print0("="*80)
core_eval_start_time = time.time()
model.eval()
with disable_fp8(orig_model), autocast_ctx:
    final_core_results = evaluate_core(orig_model, tokenizer, device, max_per_task=-1)
core_eval_time = time.time() - core_eval_start_time
print0(f"Final CORE metric: {final_core_results['core_metric']:.4f}")
print0(f"CORE evaluation time: {core_eval_time:.2f}s ({core_eval_time/60:.2f}m)")

# Run optimized CORE evaluation for A/B comparison (not logged to wandb)
print0("\n" + "="*80)
print0("Running optimized CORE evaluation (A/B comparison)")
print0("="*80)
optim_eval_start_time = time.time()
with disable_fp8(orig_model), autocast_ctx:
    optim_core_results = evaluate_core(orig_model, tokenizer, device, max_per_task=-1,
                                       evaluate_task_fn=evaluate_task_optim)
optim_eval_time = time.time() - optim_eval_start_time
print0(f"Optimized CORE metric: {optim_core_results['core_metric']:.4f}")
print0(f"Optimized CORE evaluation time: {optim_eval_time:.2f}s ({optim_eval_time/60:.2f}m)")
print0(f"Speedup: {core_eval_time / optim_eval_time:.2f}x")
print0(f"Accuracy delta (optim - baseline): {optim_core_results['core_metric'] - final_core_results['core_metric']:.6f}")

# Run varlen CORE evaluation with static torch.compile (not logged to wandb)
print0("\n" + "="*80)
print0("Running varlen CORE evaluation (compiled, static shapes)")
print0("="*80)
# Compile with static shapes: varlen eval uses a fixed token budget so every forward
# call sees the same (1, budget) shape -- no dynamic=True needed, no recompilation
compiled_model = torch.compile(orig_model)
varlen_eval_start_time = time.time()
with disable_fp8(orig_model), autocast_ctx:
    varlen_core_results = evaluate_core(compiled_model, tokenizer, device, max_per_task=-1,
                                        evaluate_task_fn=evaluate_task_varlen)
varlen_eval_time = time.time() - varlen_eval_start_time
print0(f"Varlen CORE metric: {varlen_core_results['core_metric']:.4f}")
print0(f"Varlen CORE evaluation time: {varlen_eval_time:.2f}s ({varlen_eval_time/60:.2f}m)")
print0(f"Speedup vs baseline: {core_eval_time / varlen_eval_time:.2f}x")
print0(f"Speedup vs optimized: {optim_eval_time / varlen_eval_time:.2f}x")
print0(f"Accuracy delta (varlen - baseline): {varlen_core_results['core_metric'] - final_core_results['core_metric']:.6f}")

# Save A/B/C comparison data to disk as JSON for cross-run analysis
if ddp_rank == 0:
    import datetime
    comparison_dir = os.path.join("logs", "core_eval_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_data = {
        "run_name": args.run,
        "timestamp": datetime.datetime.now().isoformat(),
        "training_step": step,
        "total_training_flops": flops_so_far,
        "baseline": {
            "core_metric": final_core_results["core_metric"],
            "results": final_core_results["results"],
            "centered_results": final_core_results["centered_results"],
            "task_times": final_core_results["task_times"],
            "total_eval_time_seconds": core_eval_time,
        },
        "optimized": {
            "core_metric": optim_core_results["core_metric"],
            "results": optim_core_results["results"],
            "centered_results": optim_core_results["centered_results"],
            "task_times": optim_core_results["task_times"],
            "total_eval_time_seconds": optim_eval_time,
        },
        "varlen": {
            "core_metric": varlen_core_results["core_metric"],
            "results": varlen_core_results["results"],
            "centered_results": varlen_core_results["centered_results"],
            "task_times": varlen_core_results["task_times"],
            "total_eval_time_seconds": varlen_eval_time,
        },
    }
    comparison_path = os.path.join(comparison_dir, f"{args.run}.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    print0(f"A/B/C comparison saved to: {comparison_path}")

# Prepare training outcomes for logging
training_outcomes = {
    "final/min_val_bpb": min_val_bpb if val_bpb is not None else None,
    "final/val_bpb": val_bpb,
    "final/core_metric": final_core_results["core_metric"],
    "final/mfu_percent": mfu,
    "final/total_training_flops": flops_so_far,
    "final/total_training_time_minutes": total_training_time / 60,
    "final/peak_memory_mib": get_max_memory() / 1024 / 1024,
    "final/core_eval_time_seconds": core_eval_time,
}

# Log training outcomes to wandb as final metrics
if not use_dummy_wandb:
    wandb_run.log(training_outcomes)
    # Also log the detailed CORE results
    wandb_run.log({"final/core_centered_results": final_core_results["centered_results"]})
    
    # Log detailed per-task CORE results with "core/" prefix
    core_detailed_metrics = {}
    for task_label in final_core_results["results"]:
        core_detailed_metrics[f"core/{task_label}/accuracy"] = final_core_results["results"][task_label]
        core_detailed_metrics[f"core/{task_label}/centered_accuracy"] = final_core_results["centered_results"][task_label]
        core_detailed_metrics[f"core/{task_label}/time_seconds"] = final_core_results["task_times"][task_label]
    wandb_run.log(core_detailed_metrics)

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    training_setup_stats, # stats about the training setup
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric": final_core_results["core_metric"],
        "CORE eval time": f"{core_eval_time:.2f}s",
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

if not use_dummy_wandb:
    wandb.save("scripts/base_train.py")
    wandb.save("scripts/base_eval.py")
    wandb.save("nanochat/core_eval.py")
    wandb.save("nanochat/core_eval_optim.py")
    wandb.save("nanochat/core_eval_varlen.py")
    wandb.save("nanochat/dataloader.py")
    wandb.save(f"logs/{args.run}.log")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
