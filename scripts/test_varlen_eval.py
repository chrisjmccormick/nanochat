"""
Test script: load a checkpoint and run baseline / optimized / varlen CORE evaluations
to compare accuracy and speed.

Usage (via torchrun):
    torchrun --standalone --nproc_per_node=8 -m scripts.test_varlen_eval
    torchrun --standalone --nproc_per_node=8 -m scripts.test_varlen_eval -- --max-per-task 5
"""
import argparse
import time
import os
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.core_eval_optim import evaluate_task as evaluate_task_optim
from nanochat.core_eval_varlen import evaluate_task as evaluate_task_varlen
from scripts.base_eval import evaluate_core


def main():
    parser = argparse.ArgumentParser(description="A/B/C CORE evaluation comparison")
    parser.add_argument("--model-tag", type=str, default="d12", help="Model tag (default: d12)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--max-per-task", type=int, default=-1,
                        help="Max examples per task (-1 = all, use small number for quick smoke test)")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile for varlen (debugging)")
    args = parser.parse_args()

    # Multi-GPU init
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = device.type
    print0(f"Running on {ddp_world_size} GPU(s), device={device}")

    # Load checkpoint
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    if args.step is None:
        args.step = find_last_step(checkpoint_dir)
    print0(f"Loading checkpoint: {checkpoint_dir}/model_{args.step:06d}.pt")

    model, tokenizer, meta_data = build_model(checkpoint_dir, args.step, device, phase="eval")
    print0(f"Model config: {meta_data['model_config']}")
    print0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # ---- A) Baseline CORE evaluation ----
    print0("\n" + "=" * 80)
    print0("A) Running BASELINE CORE evaluation (uncompiled, original evaluate_task)")
    print0("=" * 80)
    t0 = time.time()
    with autocast_ctx:
        baseline_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
    baseline_time = time.time() - t0
    print0(f"Baseline CORE metric: {baseline_results['core_metric']:.4f}")
    print0(f"Baseline time: {baseline_time:.2f}s ({baseline_time/60:.2f}m)")

    # ---- B) Optimized CORE evaluation ----
    print0("\n" + "=" * 80)
    print0("B) Running OPTIMIZED CORE evaluation (uncompiled, batched evaluate_task_optim)")
    print0("=" * 80)
    t0 = time.time()
    with autocast_ctx:
        optim_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task,
                                      evaluate_task_fn=evaluate_task_optim)
    optim_time = time.time() - t0
    print0(f"Optimized CORE metric: {optim_results['core_metric']:.4f}")
    print0(f"Optimized time: {optim_time:.2f}s ({optim_time/60:.2f}m)")

    # ---- C) Varlen compiled CORE evaluation ----
    print0("\n" + "=" * 80)
    print0("C) Running VARLEN CORE evaluation (torch.compiled, static shapes)")
    print0("=" * 80)
    if args.no_compile:
        compiled_model = model
        print0("Skipping torch.compile (--no-compile)")
    else:
        print0("Compiling model with torch.compile(dynamic=False)...")
        compiled_model = torch.compile(model, dynamic=False)

    t0 = time.time()
    with autocast_ctx:
        varlen_results = evaluate_core(compiled_model, tokenizer, device, max_per_task=args.max_per_task,
                                       evaluate_task_fn=evaluate_task_varlen)
    varlen_time = time.time() - t0
    print0(f"Varlen CORE metric: {varlen_results['core_metric']:.4f}")
    print0(f"Varlen time: {varlen_time:.2f}s ({varlen_time/60:.2f}m)")

    # ---- Summary ----
    print0("\n" + "=" * 80)
    print0("SUMMARY")
    print0("=" * 80)
    print0(f"{'Method':<20s} {'CORE metric':>12s} {'Time (s)':>10s} {'Speedup':>10s}")
    print0(f"{'-'*20} {'-'*12} {'-'*10} {'-'*10}")
    print0(f"{'Baseline':<20s} {baseline_results['core_metric']:>12.4f} {baseline_time:>10.2f} {'1.00x':>10s}")
    print0(f"{'Optimized':<20s} {optim_results['core_metric']:>12.4f} {optim_time:>10.2f} {baseline_time/optim_time:>9.2f}x")
    print0(f"{'Varlen (compiled)':<20s} {varlen_results['core_metric']:>12.4f} {varlen_time:>10.2f} {baseline_time/varlen_time:>9.2f}x")
    print0()
    print0(f"Accuracy delta (optimized - baseline): {optim_results['core_metric'] - baseline_results['core_metric']:.6f}")
    print0(f"Accuracy delta (varlen    - baseline): {varlen_results['core_metric'] - baseline_results['core_metric']:.6f}")

    compute_cleanup()


if __name__ == "__main__":
    main()
