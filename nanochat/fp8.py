"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

Drop-in FP8 compute for the flattened GPT's matmuls in ~150 lines (vs torchao's
~2000). We only need the "tensorwise" recipe (one scalar scale per tensor), not
the full generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor,
tensor subclass dispatch tables, etc.)

How FP8 training works
======================
A linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this integrates with the flattened GPT
==========================================
The pre-flattening model swapped nn.Linear modules for a Float8Linear subclass
(torchao's API). The flattened GPT has no Linear modules — every matmul goes
through nanochat.gpt.linear(x, w) on a raw Parameter — so FP8 is now a dispatch
inside that helper: when the module-level `enabled` flag is set and the weight
passes `eligible()` (FP8 hardware alignment + not tiny), the matmul routes to
fp8_linear() below instead of F.linear. base_train sets the flag under --fp8;
evaluations temporarily clear it with disable_fp8(). torch.compile guards on the
flag, so flipping it swaps to a separately-compiled (and cached) graph rather
than silently running stale code — the same recompile cost profile as the old
module-swapping disable_fp8.

The compute core (_Float8Matmul) is unchanged from the module-based version: a
single autograd.Function that takes full-precision inputs, quantizes to FP8
internally, calls _scaled_mm, and returns full-precision outputs. Marked
@allow_in_graph so torch.compile treats it as one opaque node rather than trying
to trace inside. Both this and torchao call the exact same cuBLAS _scaled_mm
kernel — the GPU matmul is identical; only the "glue" ops (amax, scale, cast)
sit outside Inductor's fusion reach here, and those are tiny next to the matmul.

Two consumers, one recipe
=========================
The autograd path above (gpt.linear -> fp8_linear) is what tutorials/flat_train.py
and scripts/base_train-flat.py still use. The fwd-bwd trainer has no linear() to
hook: train_step.forward_backward_fp8 writes the three GEMMs out itself at every
site, so it consumes the SECOND half of this file — the same tensorwise recipe
(same amax, same float64 division, same e4m3/e5m2 split, same fast_accum policy)
exposed as traceable pieces instead of one opaque autograd.Function.
"""

from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch import Tensor

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12

# Module-level switch consulted by nanochat.gpt.linear on every matmul.
enabled = False


def enable_fp8_training():
    """Route eligible matmuls in gpt.linear through FP8. Requires sm89+ (H100/Ada)."""
    global enabled
    enabled = True


@contextmanager
def disable_fp8():
    """Temporarily route matmuls back to bf16 (evals/sampling stay full precision).
    No-op when FP8 was never enabled."""
    global enabled
    prev = enabled
    enabled = False
    try:
        yield
    finally:
        enabled = prev


def eligible(w):
    """FP8 hardware requires dims divisible by 16; also skip small matrices
    (ve_gate, smear_gate) where quantization overhead outweighs the matmul."""
    return w.shape[0] % 16 == 0 and w.shape[1] % 16 == 0 and min(w.shape) >= 128


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


def fp8_linear(x, w):
    """FP8 replacement for F.linear(x, w.to(x.dtype)), no-bias. The weight stays
    in master precision; only the three GEMMs run in FP8."""
    # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
    # reduced precision input, and we no longer rely on autocast to do this.
    x = x.to(COMPUTE_DTYPE)
    # _scaled_mm only works on 2D tensors, so flatten batch dimensions
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    output = _Float8Matmul.apply(x_2d, w)
    return output.reshape(*orig_shape[:-1], output.shape[-1])


# =============================================================================
# Primitives for the handwritten path (train_step.forward_backward_fp8)
#
# forward_backward_fp8 writes out the three GEMMs of each "linear" itself, at 21
# sites, so it needs _Float8Matmul's pieces separately — and TRACEABLE (no
# autograd.Function, no allow_in_graph), so inductor can fuse each amax/cast
# into its neighbouring pointwise work. torch.compile is not optional for that
# body: in eager every cast materializes an fp32 temp the size of its input.
#
# All three GEMMs are written in "NT" form,  C[M,N] = A[M,K] @ B[N,K].T, because
# the cuBLAS FP8 kernels only accept operands whose CONTRACTION dim is contiguous
# (equivalently: _scaled_mm wants mat1 row-major and mat2 column-major):
#
#   forward   y[T,out]   = x[T,in]    @ w[out,in].T     A=x   (K=in)   B=w   (K=in)
#   dgrad     dx[T,in]   = dy[T,out]  @ wT[in,out].T    A=dy  (K=out)  B=wT  (K=out)
#   wgrad     dw[out,in] = dyT[out,T] @ xT[in,T].T      A=dyT (K=T)    B=xT  (K=T)
#
# Read down the A/B columns: each tensor is needed in exactly two layouts — once
# with its own last dim as K (row() below), once with its FIRST dim as K (col(),
# a real transposing copy). Those copies are the tax tensorwise FP8 pays for the
# backward pass; TransformerEngine calls the same thing a "transpose cache".
# =============================================================================

E4M3 = torch.float8_e4m3fn   # activations and weights: more mantissa bits
E5M2 = torch.float8_e5m2     # gradients: wider exponent range


class F8(NamedTuple):
    """One quantized GEMM operand: fp8 data whose contraction dim is last and
    contiguous, plus the inverse scale _scaled_mm multiplies back in."""
    d: Tensor
    inv: Tensor


def tensor_scale(x, fp8_dtype):
    """Tensorwise dynamic scale for x — one 0-dim fp32 device scalar, no host
    sync. abs()/amax() run in x's own dtype: both are exact in any float format,
    so this is bit-identical to _to_fp8's `.float().abs().max()` without
    materializing an fp32 copy of x in eager. The division upcasts to float64
    exactly like torchao's, which is what keeps compile and eager agreeing."""
    amax = x.abs().amax()
    return (torch.finfo(fp8_dtype).max / amax.double().clamp(min=EPS)).float()


def slice_scales(bank, fp8_dtype):
    """Per-slice scales for a (K, out, in) parameter bank as ONE batched
    reduction -> (K,) fp32. Same granularity as the old per-layer Float8Linear
    (one scale per matrix), but one kernel launch instead of K."""
    amax = bank.abs().amax(dim=(-2, -1))
    return (torch.finfo(fp8_dtype).max / amax.double().clamp(min=EPS)).float()


def _cast(x, fp8_dtype, scale):
    """Scale into FP8 range, saturate (the cast itself wraps, it does not
    saturate), then narrow. _Float8Matmul's _to_fp8, minus the amax."""
    fp8_max = torch.finfo(fp8_dtype).max
    return (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)


def row(x, fp8_dtype, scale) -> F8:
    """Quantize x as it lies: the GEMM contracts over x's LAST dim."""
    return F8(_cast(x, fp8_dtype, scale), scale.reciprocal())


def col(x, fp8_dtype, scale) -> F8:
    """Quantize x.T: the GEMM contracts over x's FIRST dim. The .t().contiguous()
    is a real fp8 copy in eager; under inductor it fuses into the cast, making it
    one transposing pass over x. (Casting the `x.mT` view instead does NOT work:
    TensorIterator propagates the transposed strides, so the result comes back
    K-strided and _scaled_mm rejects it.)"""
    return F8(_cast(x, fp8_dtype, scale).t().contiguous(), scale.reciprocal())


class BankF8(NamedTuple):
    """A weight bank quantized in BOTH GEMM layouts, sharing one scale set:
    row[i] is slice i as it lies (the forward's B operand), colT[i] is slice i
    transposed-contiguous (the dgrad's B operand). Built once per optimizer
    step — the live weights are frozen across a step's micro-batches, so
    re-quantizing them inside every micro-batch is pure waste. `s` is the
    FORWARD scale vector; take s[i].reciprocal() at the use site (passing a
    bare select view as a _scaled_mm scale breaks inductor's lowering)."""
    row: Tensor   # (K, out, in) fp8   [or (out, in) for a 2D weight]
    colT: Tensor  # (K, in, out) fp8   [or (in, out)]
    s: Tensor     # (K,) fp32 forward scales  [or 0-dim]


def quantize_bank(bank, fp8_dtype=E4M3):
    """(K, out, in) parameter bank -> BankF8. Same per-slice scales and cast
    values as slice_scales + per-slice row()/col() — bit-identical, batched."""
    s = slice_scales(bank, fp8_dtype)
    fp8_max = torch.finfo(fp8_dtype).max
    d = (bank.float() * s.view(-1, 1, 1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return BankF8(d, d.transpose(-2, -1).contiguous(), s)


def quantize_weight_2d(w, fp8_dtype=E4M3):
    """2D weight (the lm_head) -> BankF8, tensorwise scale."""
    s = tensor_scale(w, fp8_dtype)
    d = _cast(w, fp8_dtype, s)
    return BankF8(d, d.t().contiguous(), s)


def quantize_weights(model):
    """Every weight role forward_backward_fp8 quantizes, in both layouts.
    Call once per optimizer step and pass the result to the body via `w8`
    (compile this call — eager materializes fp32 temps per bank)."""
    q = {name: quantize_bank(getattr(model, name))
         for name in ("c_q", "c_v", "c_k", "attn_proj", "mlp_fc", "mlp_proj")}
    q["lm_head"] = quantize_weight_2d(model.lm_head)
    return q


def mm(a: F8, b: F8, out_dtype, fast_accum=False):
    """C[M,N] = a[M,K] @ b[N,K].T through the cuBLAS FP8 kernel. fast_accum keeps
    the dot-product accumulation in lower precision — standard for the forward,
    off in the backward, exactly as in _Float8Matmul above."""
    # Cheap insurance, and it makes CPU tests representative: the CPU _scaled_mm
    # accepts any layout, the CUDA one requires this.
    assert a.d.stride(-1) == 1 and b.d.stride(-1) == 1, \
        "FP8 GEMM operands must be K-contiguous — build them with row()/col()"
    return torch._scaled_mm(a.d, b.d.t(), scale_a=a.inv, scale_b=b.inv,
                            out_dtype=out_dtype, use_fast_accum=fast_accum)


def check_eligible(model):
    """forward_backward_fp8 quantizes a FIXED set of sites, so eligibility is a
    precondition rather than a per-site dispatch (a mixed body would need a bf16
    twin of every site, i.e. the thing the separate fp8 body exists to avoid).
    Raises if any site can't run FP8 at this config. Returns a one-line summary."""
    sites = [("c_q", model.c_q[0]), ("c_k", model.c_k[0]), ("c_v", model.c_v[0]),
             ("attn_proj", model.attn_proj[0]), ("mlp_fc", model.mlp_fc[0]),
             ("mlp_proj", model.mlp_proj[0]), ("lm_head", model.lm_head)]
    bad = [f"{n}{tuple(w.shape)}" for n, w in sites if not eligible(w)]
    if bad:
        raise ValueError(
            f"--fp8 needs every matrix bank + lm_head to be FP8-eligible (dims %16, "
            f"min dim >= 128); ineligible at this config: {', '.join(bad)}")
    V, Vp = model.config.vocab_size, model.padded_vocab_size
    if V % 16 or Vp % 16:
        raise ValueError(f"--fp8 needs vocab_size and padded_vocab_size divisible "
                         f"by 16 (the lm_head backward contracts over V): {V}, {Vp}")
    return (f"FP8: {len(sites)} matmul roles quantized per bank slice "
            f"(e4m3 activations/weights, e5m2 grads, tensorwise); "
            f"attention/norms/CE/embeddings stay {COMPUTE_DTYPE}")
