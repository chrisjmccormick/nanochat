"""
Pre-computed optimizer hyperparameter schedules.

A run's optimizer is defined up front: every learning rate, beta and weight decay
for every step is computed here, before training starts, into per-step tables of
*update coefficients* — the numbers the fused kernels actually multiply by. The
optimizer then holds no hyperparameters of its own and the training loop has
nothing to set per step; both just read row `step` of each group's tables.

Folding the schedules all the way down into coefficients (rather than passing raw
lr/betas and doing the arithmetic in the kernel each step) buys three things:

- The bias corrections leave the kernel. `1 - beta**t` is the correct correction
  only while beta is *constant*; the general form is the running sum of the EMA's
  weights, w[t] = beta[t]*w[t-1] + (1 - beta[t]), which is trivial to compute on
  the host and stays exact for any beta schedule. The kernel loses a `pow` and
  loses its dependence on a step count entirely.
- The LR scale factors collapse into the tables at build time — 1/sqrt(dmodel)
  and sqrt(B/B_ref) for AdamW, sqrt(fan_out/fan_in) for Muon — instead of being
  applied in three different files.
- Nothing about the schedule is left for the training loop to do per step, which
  is what lets the whole optimizer step eventually become CUDA-graph capturable
  (stage 2: the tables move to the device and are indexed by a device counter;
  the kernels do not change).

Usage: describe each scalar with a `Ramp` (or a bare float for a constant),
collect them into `AdamWGroup`/`MuonGroup` specs, and hand those to
`build_param_groups(specs, num_steps)` to get the param groups the optimizers in
nanochat/optim.py consume.
"""

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

# A scheduled scalar: a Ramp, a constant, or a pre-computed per-step sequence.
Schedulable = "Ramp | float | Sequence[float]"

# -----------------------------------------------------------------------------
# Schedule specification

@dataclass
class Ramp:
    """A scalar's trajectory over a run of N optimizer steps: warm up from
    `start` to `peak`, hold at `peak`, then cool down from `peak` to `end`.
    Each window is given either in steps or as a fraction of the run. A window of
    zero length (the default) means the ramp simply begins and/or ends at `peak` —
    there is nothing to interpolate, so the corresponding `start`/`end` is unused.
    That way a caller can pass `warmup_frac=args.warmup_ratio` without special-
    casing the run that sets it to zero.

    For step index i in [0, N), with warmup length W and cooldown length C:

        i < W        ->  start + (peak - start) * (i + 1) / W
        i <= N - C   ->  peak
        otherwise    ->  end + (peak - end) * f,   f = (N - i) / C

    So warmup reaches `peak` exactly on the last step of its window, and cooldown
    approaches `end` from above, arriving one step past the end of the run. That
    is the convention nanochat's hand-written schedules have always used; every
    one of them is reproduced exactly by this form (see NOTES in the ops session)
    except Muon's momentum warmup, which used `i/W` and so now leads by one step
    out of 400.

    `shape="cosine"` swaps the cooldown's linear interpolation for a half-cosine
    (which is how the Muon weight-decay decay is expressed).
    """
    peak: float
    start: float | None = None
    end: float | None = None
    warmup_steps: int | None = None
    warmup_frac: float | None = None
    cooldown_steps: int | None = None
    cooldown_frac: float | None = None
    shape: str = "linear"

    def __mul__(self, k: float) -> "Ramp":
        """Scale the whole ramp, so one shared LR *shape* can be reused at each
        group's own peak: `lrm * embedding_lr`, `lrm * matrix_lr`, ..."""
        return replace(self, peak=self.peak * k,
                       start=None if self.start is None else self.start * k,
                       end=None if self.end is None else self.end * k)

    __rmul__ = __mul__

    def materialize(self, num_steps: int) -> np.ndarray:
        """The scalar's value at every step, as an (N,) float64 array."""
        N = num_steps
        W = _window(self.warmup_steps, self.warmup_frac, N, "warmup")
        C = _window(self.cooldown_steps, self.cooldown_frac, N, "cooldown")
        assert self.shape in ("linear", "cosine"), f"unknown ramp shape {self.shape!r}"
        assert W >= 0 and C >= 0, "ramp windows must be non-negative"
        assert W + C <= N, f"warmup ({W}) + cooldown ({C}) exceed the run ({N} steps)"
        start = self.peak if self.start is None else self.start
        end = self.peak if self.end is None else self.end

        i = np.arange(N, dtype=np.float64)
        v = np.full(N, float(self.peak), dtype=np.float64)
        if W > 0:
            v[:W] = start + (self.peak - start) * (i[:W] + 1.0) / W
        if C > 0:
            tail = slice(N - C + 1, N)  # the hold covers i <= N - C
            f = (N - i[tail]) / C
            if self.shape == "cosine":
                f = 0.5 * (1.0 + np.cos(math.pi * (1.0 - f)))
            v[tail] = end + (self.peak - end) * f
        return v


def _window(steps, frac, num_steps, name):
    assert steps is None or frac is None, f"give {name}_steps or {name}_frac, not both"
    if steps is not None:
        return int(steps)
    if frac is not None:
        return round(frac * num_steps)
    return 0


def _as_table(spec, num_steps, what) -> np.ndarray:
    """Materialize a Schedulable into an (N,) float64 array."""
    if isinstance(spec, Ramp):
        return spec.materialize(num_steps)
    if isinstance(spec, (int, float)):
        return np.full(num_steps, float(spec), dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64)
    assert arr.shape == (num_steps,), f"{what}: expected {num_steps} values, got {arr.shape}"
    return arr


def _bias_correction(beta: np.ndarray) -> np.ndarray:
    """Running sum of an EMA's weights: w[t] = beta[t]*w[t-1] + (1 - beta[t]),
    starting from w[-1] = 0. Collapses to the familiar 1 - beta**(t+1) when beta
    is constant, and stays exact when it isn't (which `1 - beta**t` does not)."""
    w = np.empty_like(beta)
    acc = 0.0
    for t, b in enumerate(beta):
        acc = b * acc + (1.0 - b)
        w[t] = acc
    return w


# -----------------------------------------------------------------------------
# Parameter group specifications
#
# Every hyperparameter is required: the optimizers have no defaults to fall back
# on, so a group is either fully specified here or it is a construction error.

@dataclass
class AdamWGroup:
    """One AdamW parameter group and its schedules."""
    params: list[Tensor]
    lr: Schedulable
    betas: tuple[Schedulable, Schedulable]
    eps: float
    weight_decay: Schedulable


@dataclass
class MuonGroup:
    """One Muon parameter group and its schedules. All params must share a shape
    (they get stacked into one tensor), and that shape's sqrt(fan_out/fan_in)
    LR scaling is folded into the group's lr table at build time."""
    params: list[Tensor]
    lr: Schedulable
    momentum: Schedulable
    beta2: Schedulable
    weight_decay: Schedulable
    ns_steps: int = 5


# -----------------------------------------------------------------------------
# Update coefficients
#
# One NamedTuple per optimizer, used for two things: the group's *tables* (each
# field an (N,) tensor, one row per step) and the group's *coefficients* (each
# field a 0-D tensor holding the current step's row, which is what the fused
# kernel reads). Same field names, same order — so refilling the coefficients
# from the tables is a zip, and the kernels take one argument instead of ten.

class AdamWCoeffs(NamedTuple):
    """What an AdamW step multiplies by. `p` and the moments are the only other
    inputs — there is no step count and no beta**t left in the kernel."""
    wd_mul: Tensor           # 1 - lr*wd             decoupled weight decay
    one_minus_beta1: Tensor  # 1 - beta1             exp_avg lerp weight
    one_minus_beta2: Tensor  # 1 - beta2             exp_avg_sq lerp weight
    rsqrt_bias2: Tensor      # 1/sqrt(bias2)         second-moment bias correction
    step_size: Tensor        # lr / bias1            lr schedule x first-moment bias correction
    eps: Tensor              # epsilon               (constant, never scheduled)


class MuonCoeffs(NamedTuple):
    """What a Muon step multiplies by. Muon's second moment is self-normalizing
    (the v_norm/v_norm_new rescale), so it needs no bias correction."""
    momentum: Tensor            # nesterov momentum
    one_minus_momentum: Tensor  # 1 - momentum        momentum_buffer lerp weight
    one_minus_beta2: Tensor     # 1 - beta2           variance-reduction lerp weight
    lr: Tensor                  # lr * sqrt(max(1, fan_out/fan_in))
    lr_wd: Tensor               # lr * weight_decay   cautious decay


def _tables(coeffs_cls, **fields):
    """Package per-step numpy arrays as fp32 CPU tensors."""
    return coeffs_cls(**{k: torch.tensor(v, dtype=torch.float32, device="cpu")
                         for k, v in fields.items()})


def _scalars(coeffs_cls):
    """The 0-D CPU tensors the optimizer refills each step and hands to the kernel.
    Allocated once per group, mutated in place, so pointers stay stable."""
    return coeffs_cls(**{f: torch.zeros((), dtype=torch.float32, device="cpu")
                         for f in coeffs_cls._fields})


# -----------------------------------------------------------------------------
# Build

def build_param_groups(specs: list, num_steps: int) -> list[dict]:
    """Turn schedule specs into the param groups nanochat/optim.py consumes.

    Each returned group carries `tabs` (per-step coefficient tables, N rows) and
    `coeffs` (the 0-D tensors the optimizer refills from `tabs` each step). It
    carries no lr/betas/weight_decay: those exist only as folded coefficients.
    """
    assert num_steps > 0, "the run's step count must be known before the optimizer is built"
    groups = []
    for spec in specs:
        assert len(spec.params) > 0, f"empty parameter group: {type(spec).__name__}"
        if isinstance(spec, AdamWGroup):
            groups.append(_build_adamw(spec, num_steps))
        elif isinstance(spec, MuonGroup):
            groups.append(_build_muon(spec, num_steps))
        else:
            raise TypeError(f"expected AdamWGroup or MuonGroup, got {type(spec).__name__}")
    return groups


def _build_adamw(spec: AdamWGroup, N: int) -> dict:
    lr = _as_table(spec.lr, N, "lr")
    beta1 = _as_table(spec.betas[0], N, "beta1")
    beta2 = _as_table(spec.betas[1], N, "beta2")
    wd = _as_table(spec.weight_decay, N, "weight_decay")
    tabs = _tables(
        AdamWCoeffs,
        wd_mul=1.0 - lr * wd,
        one_minus_beta1=1.0 - beta1,
        one_minus_beta2=1.0 - beta2,
        rsqrt_bias2=1.0 / np.sqrt(_bias_correction(beta2)),
        step_size=lr / _bias_correction(beta1),
        eps=np.full(N, float(spec.eps)),
    )
    return dict(kind="adamw", params=list(spec.params),
                tabs=tabs, coeffs=_scalars(AdamWCoeffs), lr_table=lr)


def _build_muon(spec: MuonGroup, N: int) -> dict:
    params = list(spec.params)
    shape = params[0].shape
    assert all(p.shape == shape for p in params), \
        "a Muon group's params are stacked into one tensor, so they must share a shape"
    # Muon scales the LR by sqrt(fan_out/fan_in) for tall matrices; it depends only
    # on the group's shape, so it folds into the table rather than the step.
    lr = _as_table(spec.lr, N, "lr") * max(1.0, shape[-2] / shape[-1]) ** 0.5
    momentum = _as_table(spec.momentum, N, "momentum")
    beta2 = _as_table(spec.beta2, N, "beta2")
    wd = _as_table(spec.weight_decay, N, "weight_decay")
    tabs = _tables(
        MuonCoeffs,
        momentum=momentum,
        one_minus_momentum=1.0 - momentum,
        one_minus_beta2=1.0 - beta2,
        lr=lr,
        lr_wd=lr * wd,
    )
    return dict(kind="muon", params=params, ns_steps=spec.ns_steps,
                tabs=tabs, coeffs=_scalars(MuonCoeffs), lr_table=lr)
