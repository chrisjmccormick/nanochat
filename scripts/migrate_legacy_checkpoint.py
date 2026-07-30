"""
One-time migration for checkpoints written before the current parameter layout.

These remaps used to run on EVERY load, inside checkpoint_manager.build_model,
so the hot path carried three generations of format history and re-derived the
same conversion each time a model was loaded. They are a migration, not a
loading concern, so they live here instead: run this once against an old
checkpoint directory, and the normal loader reads the result directly.

Two generations are handled, applied in this order:

1. PRE-FLATTENING (modular GPT with Block/Attention/MLP submodules)
   `transformer.h.3.attn.c_q.weight` -> `c_q.3`
   Also covers the HuggingFace-published base checkpoints.

2. PRE-BANKING (one Parameter per layer)
   `c_q.0` ... `c_q.11` -> a single stacked `c_q` of shape (n_layer, out, in)

Plus two content patches for configs/params that did not exist yet:
   - `window_pattern` defaults to "L" (old models trained at full context)
   - `resid_lambdas` defaults to 1.0, `x0_lambdas` to 0.0 (identity / disabled)

Usage
-----
    # convert in place next to the original, writing model_<step>.migrated.pt
    python -m scripts.migrate_legacy_checkpoint --checkpoint-dir ~/.cache/nanochat/base_checkpoints/d12 --step 21400

    # write the converted checkpoint into a new directory instead
    python -m scripts.migrate_legacy_checkpoint --checkpoint-dir <old> --step 21400 --out-dir <new>

    # check what would change without writing anything
    python -m scripts.migrate_legacy_checkpoint --checkpoint-dir <old> --step 21400 --dry-run

The optimizer half is deliberately NOT migrated: the explicit trainer's state
layout (bf16-live + uint16 mantissa masters, factored second moments) has no
counterpart in the old torch.optim states, so a migrated checkpoint can be used
for inference or to start a fresh optimizer, not to resume one mid-run.
"""

import os
import re
import json
import argparse

import torch


_BANKED_ROLES = ("c_q", "c_k", "c_v", "attn_proj", "mlp_fc", "mlp_proj", "value_embeds", "ve_gate")
_PER_LAYER_KEY = re.compile(rf"^({'|'.join(_BANKED_ROLES)})\.(\d+)$")


def remap_legacy_keys(model_data):
    """Map pre-flattening checkpoint keys (modular Block/Attention/MLP GPT) to the
    flattened GPT parameter names. New-format checkpoints pass through untouched."""
    if not any(k.startswith("transformer.") for k in model_data):
        return model_data, False
    legacy_patterns = [
        (re.compile(r"^transformer\.wte\.weight$"), "wte"),
        (re.compile(r"^lm_head\.weight$"), "lm_head"),
        (re.compile(r"^transformer\.h\.(\d+)\.attn\.c_q\.weight$"), r"c_q.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.attn\.c_k\.weight$"), r"c_k.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.attn\.c_v\.weight$"), r"c_v.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.attn\.c_proj\.weight$"), r"attn_proj.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.attn\.ve_gate\.weight$"), r"ve_gate.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_fc\.weight$"), r"mlp_fc.\1"),
        (re.compile(r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$"), r"mlp_proj.\1"),
        (re.compile(r"^value_embeds\.(\d+)\.weight$"), r"value_embeds.\1"),
        (re.compile(r"^smear_gate\.weight$"), "smear_gate"),
    ]
    remapped = {}
    for k, v in model_data.items():
        for pattern, repl in legacy_patterns:
            new_k, n = pattern.subn(repl, k)
            if n:
                remapped[new_k] = v
                break
        else:
            remapped[k] = v  # scalars etc. keep their names
    return remapped, True


def stack_legacy_banks(model_data):
    """Stack pre-bank per-layer keys (c_q.0 ... c_q.11, value_embeds.3, ...) into
    the banked single-tensor layout. For value_embeds/ve_gate the numeric suffix
    is the LAYER index; ascending layer order matches the model's ve_index slot
    order. New-format (banked) checkpoints pass through untouched."""
    if not any(_PER_LAYER_KEY.match(k) for k in model_data):
        return model_data, False
    stacked = {k: v for k, v in model_data.items() if not _PER_LAYER_KEY.match(k)}
    for role in _BANKED_ROLES:
        keys = sorted((k for k in model_data if re.match(rf"^{role}\.\d+$", k)),
                      key=lambda k: int(k.rsplit(".", 1)[1]))
        if keys:
            stacked[role] = torch.stack([model_data[k] for k in keys])
    return stacked, True


def patch_missing_config_keys(model_config_kwargs):
    """Add default values for config keys that did not exist in old checkpoints."""
    notes = []
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"  # old models trained at full context
        notes.append("window_pattern -> 'L'")
    return notes


def patch_missing_keys(model_data, n_layer):
    """Add default values for parameters that did not exist in old checkpoints."""
    notes = []
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)  # identity scaling
        notes.append("resid_lambdas -> 1.0")
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)  # disabled
        notes.append("x0_lambdas -> 0.0")
    return notes


def migrate(checkpoint_dir, step, out_dir=None, dry_run=False):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    model_data = torch.load(model_path, map_location="cpu")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    changes = []
    # torch.compile wrapping prepends _orig_mod. to every key
    if any(k.startswith("_orig_mod.") for k in model_data):
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        changes.append("stripped _orig_mod. prefix")

    model_data, did = remap_legacy_keys(model_data)
    if did:
        changes.append("remapped modular (pre-flattening) keys")
    model_data, did = stack_legacy_banks(model_data)
    if did:
        changes.append("stacked per-layer keys into banks")

    changes += patch_missing_config_keys(meta_data["model_config"])
    changes += patch_missing_keys(model_data, meta_data["model_config"]["n_layer"])

    if not changes:
        print(f"Nothing to migrate: {model_path} is already in the current format.")
        return

    print(f"Migrating {model_path}")
    for c in changes:
        print(f"  - {c}")
    print(f"  resulting keys ({len(model_data)}): {sorted(model_data)}")

    if dry_run:
        print("--dry-run: nothing written.")
        return

    if out_dir is None:
        out_model = os.path.join(checkpoint_dir, f"model_{step:06d}.migrated.pt")
        out_meta = os.path.join(checkpoint_dir, f"meta_{step:06d}.migrated.json")
    else:
        os.makedirs(out_dir, exist_ok=True)
        out_model = os.path.join(out_dir, f"model_{step:06d}.pt")
        out_meta = os.path.join(out_dir, f"meta_{step:06d}.json")
    assert not os.path.exists(out_model), f"refusing to overwrite {out_model}"
    torch.save(model_data, out_model)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    print(f"Wrote {out_model}\n      {out_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="directory holding model_<step>.pt and meta_<step>.json")
    parser.add_argument("--step", type=int, required=True, help="checkpoint step to migrate")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="write the converted checkpoint here (default: alongside, .migrated suffix)")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    migrate(args.checkpoint_dir, args.step, args.out_dir, args.dry_run)
