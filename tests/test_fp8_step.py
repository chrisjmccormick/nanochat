"""FP8 training step: nanochat/fp8.py's primitives and train_step's
forward_backward_fp8.

There is no exact tier here — FP8 *is* a numerics change, so no threshold
separates "correct" from "wrong" the way the fp64 tier does for the bf16 body
(tests/test_grad_parity.py). These tests therefore pin the things that CAN be
pinned exactly:

  - layout and scale plumbing: operands K-contiguous, each scale applied to its
    own operand, and the three NT forms (forward / dgrad / wgrad) reproducing
    their reference matmuls to fp8 resolution — a swapped scale or a transposed
    operand misses by ~100%, not by 1%;
  - eligibility bookkeeping;

and then BOUND the thing that can't be pinned: forward_backward_fp8's loss and
gradients against the bf16 body's on the same weights and batch.

Runs on CPU as well as sm89+ CUDA. Note the CPU _scaled_mm accepts any operand
layout while cuBLAS does not, so on CPU it is fp8.mm's own asserts that keep the
layout tier honest — which is exactly why they live in fp8.mm.

Run: python -m pytest tests/test_fp8_step.py -v -s
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

import nanochat.gpt as gpt_mod
import nanochat.flash_attention as fa_mod
from nanochat import fp8
from nanochat.fp8 import E4M3, E5M2
from nanochat.train_step import (forward_backward, forward_backward_fp8,
                                 init_grad_buffers, zero_grad32)
from test_grad_parity import build_model, perturb, make_batch, rel_err

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _scaled_mm_available():
    try:
        a = torch.zeros(16, 16, dtype=E4M3, device=DEVICE)
        s = torch.ones((), device=DEVICE)
        torch._scaled_mm(a, a.t(), scale_a=s, scale_b=s, out_dtype=torch.float32)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _scaled_mm_available(),
    reason="no torch._scaled_mm here (needs sm89+ CUDA, or a CPU build with fp8)")


def cos_sim(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@contextmanager
def compute_dtype(dtype):
    """Run at a chosen COMPUTE_DTYPE with the SDPA/naive attention. Models must
    be BUILT inside it (init_weights casts the embeddings). On CPU this is what
    already happens, so the CPU tier and the GPU tier measure the same thing."""
    saved = (gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA)
    gpt_mod.COMPUTE_DTYPE = dtype
    fa_mod._override_impl = "sdpa"
    fa_mod.USE_FA = fa_mod._resolve_use_fa()
    try:
        yield
    finally:
        gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA = saved


# -----------------------------------------------------------------------------
# Primitives

def test_quantize_layout_and_roundtrip():
    """row()/col() must hand _scaled_mm K-contiguous data, and dequantizing must
    land within e4m3's ~6% spacing (rms far tighter)."""
    x = torch.randn(64, 128, device=DEVICE) * 3.0
    s = fp8.tensor_scale(x, E4M3)
    r, c = fp8.row(x, E4M3, s), fp8.col(x, E4M3, s)

    assert r.d.shape == (64, 128) and r.d.stride(-1) == 1
    assert c.d.shape == (128, 64) and c.d.stride(-1) == 1, "col() must be contiguous, not a strided view"
    assert r.d.dtype == E4M3 and c.d.dtype == E4M3
    assert torch.equal(c.d.float(), r.d.float().t()), "col() must be row()'s transpose, same quantization"

    # scale maps amax exactly onto the format's max, and inv undoes it
    assert abs((x.abs().max() * s).item() - torch.finfo(E4M3).max) < 1e-3
    # e4m3 keeps 3 mantissa bits: 1/8 relative spacing -> 6.25% worst case,
    # ~3.6% rms if the rounding error were uniform (observed ~2.6%)
    deq = r.d.float() * r.inv
    assert (deq - x).abs().max().item() / x.abs().max().item() < 0.07
    assert rel_err(deq, x) < 0.04


def test_scale_is_tensorwise_and_saturating():
    """One scalar for the whole tensor (not per row), and outliers saturate
    rather than wrap — the cast alone wraps, which would flip sign."""
    x = torch.randn(32, 64, device=DEVICE)
    x[0, 0] = 500.0                              # beyond e4m3's 448 before scaling
    s = fp8.tensor_scale(x, E4M3)
    assert s.shape == ()
    q = fp8.row(x, E4M3, s)
    deq = q.d.float() * q.inv
    assert deq[0, 0].item() > 400.0, "the outlier must survive as the largest value, not wrap"
    assert deq.abs().max().item() == pytest.approx(500.0, rel=0.05)


def test_nt_forms_match_reference():
    """The three GEMMs forward_backward_fp8 writes out, each against its bf16
    reference. This is the layout contract: transpose either operand or swap the
    scales and these land at ~100% error instead of ~1%."""
    T, D, O = 256, 128, 192
    x = torch.randn(T, D, device=DEVICE)
    w = torch.randn(O, D, device=DEVICE) * 0.1
    dy = torch.randn(T, O, device=DEVICE) * 0.01
    sx, sw, sd = (fp8.tensor_scale(t, dt) for t, dt in ((x, E4M3), (w, E4M3), (dy, E5M2)))

    y = fp8.mm(fp8.row(x, E4M3, sx), fp8.row(w, E4M3, sw), torch.float32, fast_accum=True)
    dx = fp8.mm(fp8.row(dy, E5M2, sd), fp8.col(w, E4M3, sw), torch.float32)
    dw = fp8.mm(fp8.col(dy, E5M2, sd), fp8.col(x, E4M3, sx), torch.float32)

    # Tolerances are the FORMAT's noise floor, not a quality target: e4m3 keeps 3
    # mantissa bits (~3.6% rms), e5m2 keeps 2 (~7%), and a K-long dot product of
    # independently-rounded terms holds that ratio rather than averaging it away.
    # Observed on CPU: fwd 3.7e-2, dgrad 5.8e-2, wgrad 5.5e-2. A transposed
    # operand or a swapped scale lands at ~1e0, so the gap is enormous either way.
    refs = {"fwd":   (y,  x @ w.mT, 0.06),
            "dgrad": (dx, dy @ w,   0.10),
            "wgrad": (dw, dy.mT @ x, 0.10)}
    for name, (got, ref, tol) in refs.items():
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        r = rel_err(got, ref)
        print(f"  {name:6s} {tuple(got.shape)} rel_err {r:.3e}")
        assert r < tol, (name, r)
        assert cos_sim(got, ref) > 0.995, name


def test_slice_scales_match_per_slice():
    """The batched per-bank reduction must equal K separate tensor_scale calls
    (per-SLICE granularity, matching the old per-layer Float8Linear)."""
    bank = torch.randn(5, 64, 128, device=DEVICE)
    bank[2] *= 100.0                             # one slice with a very different range
    got = fp8.slice_scales(bank, E4M3)
    assert got.shape == (5,)
    for i in range(5):
        assert got[i].item() == pytest.approx(fp8.tensor_scale(bank[i], E4M3).item(), rel=1e-6)


def test_mm_rejects_bad_layout():
    """CPU's _scaled_mm is layout-permissive; fp8.mm is not."""
    x = torch.randn(32, 64, device=DEVICE)
    s = fp8.tensor_scale(x, E4M3)
    strided = fp8.F8(fp8.row(x, E4M3, s).d.t(), s.reciprocal())   # K-strided, not K-contiguous
    with pytest.raises(AssertionError, match="K-contiguous"):
        fp8.mm(strided, fp8.row(x, E4M3, s), torch.float32)


def test_check_eligible():
    model = build_model(depth=2, model_dim=128, n_head=1, seq_len=64,
                        window_pattern="L", device=DEVICE, vocab=256)
    assert "e4m3" in fp8.check_eligible(model)
    small = build_model(depth=2, model_dim=64, n_head=1, seq_len=64,
                        window_pattern="L", device=DEVICE, vocab=256)
    with pytest.raises(ValueError, match="FP8-eligible"):
        fp8.check_eligible(small)                # 64 < the 128 size floor


# -----------------------------------------------------------------------------
# The body

def _tiny_model():
    model = build_model(depth=4, model_dim=128, n_head=1, seq_len=128,
                        window_pattern="SL", device=DEVICE, vocab=256)
    perturb(model)
    return model


def test_fp8_vs_bf16_forward_backward():
    """Both bodies, same weights, same batch. FP8 is a numerics change, so this
    bounds rather than pins: the loss agrees to ~1e-3, and every gradient keeps
    its direction (cosine ~1) at a bounded relative error. The scalar-reduction
    params (resid/x0/backout lambdas) sit on cancellation-heavy sums and are
    bounded loosely here, exactly as in the bf16 integration tier.

    Deliberately run at fp32 COMPUTE_DTYPE so this measures FP8's error ALONE.
    In bf16 the comparison is meaningless for the scalar params: measured against
    an fp32 reference, the bf16 body's own x0_lambdas gradient is already 2.9e-1
    off and the fp8 body's is 1.5e-1 — i.e. FP8 lands CLOSER, which is what pure
    noise looks like (agent-ops diag_fp8_noise.log).

    Observed: loss 3e-6, matrix banks 8-12e-2 with cosine 0.993-0.997,
    embeddings ~3e-2. That ~10% per-step gradient noise is what tensorwise FP8
    costs — e5m2's 2 mantissa bits dominate it — and it is the same noise the old
    fp8_linear path trained through; the run, not this test, judges convergence."""
    with compute_dtype(torch.float32):
        model = _tiny_model()
        idx, targets, cu = make_batch(256, 256, 128, DEVICE)
        init_grad_buffers(model, dtype=torch.float32)

        loss_ref = forward_backward(model, idx, targets, cu, loss_scale=1.0)
        ref = {n: p.grad32.clone() for n, p in model.named_parameters()}
        zero_grad32(model)
        loss_8 = forward_backward_fp8(model, idx, targets, cu, loss_scale=1.0)

    d_loss = abs(loss_8.item() - loss_ref.item()) / abs(loss_ref.item())
    print(f"\n  loss bf16 {loss_ref.item():.6f} | fp8 {loss_8.item():.6f} (rel {d_loss:.2e})")
    assert d_loss < 5e-3
    scalar_roles = {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda", "smear_gate"}
    for name, p in model.named_parameters():
        r, c = rel_err(p.grad32, ref[name]), cos_sim(p.grad32, ref[name])
        print(f"  {name:16s} rel_err {r:.3e}  cos {c:.6f}")
        assert r < (0.5 if name in scalar_roles else 0.2), (name, r)
        assert c > (0.9 if name in scalar_roles else 0.98), (name, c)


def test_fp8_grad_accumulation():
    """loss_scale/accumulation plumbing is untouched by FP8: two half-scaled
    micro-batches equal the sum of two separately-scaled ones."""
    model = _tiny_model()
    b1 = make_batch(256, 256, 128, DEVICE, seed=11)
    b2 = make_batch(256, 256, 128, DEVICE, seed=12)

    init_grad_buffers(model, dtype=torch.float32)
    forward_backward_fp8(model, b1[0], b1[1], b1[2], loss_scale=0.5)
    forward_backward_fp8(model, b2[0], b2[1], b2[2], loss_scale=0.5)
    both = {n: p.grad32.clone() for n, p in model.named_parameters()}

    zero_grad32(model)
    forward_backward_fp8(model, b1[0], b1[1], b1[2], loss_scale=0.5)
    first = {n: p.grad32.clone() for n, p in model.named_parameters()}
    zero_grad32(model)
    forward_backward_fp8(model, b2[0], b2[1], b2[2], loss_scale=0.5)
    for name, p in model.named_parameters():
        assert rel_err(both[name], first[name] + p.grad32) < 1e-6, name
    zero_grad32(model)
    assert all(p.grad32.abs().max() == 0 for p in model.parameters())


def test_fp8_traces_fullgraph():
    """Dynamo tier, GPU not required: fullgraph=True on backend='eager' fails
    loudly on any graph break, so this catches the whole class of "the FP8 glue
    made the body untraceable" locally, before a box is even booted. It also
    exercises the COMPILED CE branch (torch.compiler.is_compiling() is true
    under any backend) against the eager chunked twin."""
    try:
        model = _tiny_model()
        idx, targets, cu = make_batch(256, 256, 128, DEVICE)
        init_grad_buffers(model, dtype=torch.float32)
        loss_e = forward_backward_fp8(model, idx, targets, cu, loss_scale=1.0)
        eager = {n: p.grad32.clone() for n, p in model.named_parameters()}
        zero_grad32(model)
        fb = torch.compile(forward_backward_fp8, dynamic=False, fullgraph=True, backend="eager")
        loss_c = fb(model, idx, targets, cu, loss_scale=1.0)
        assert abs(loss_c.item() - loss_e.item()) / abs(loss_e.item()) < 1e-6
        for name, p in model.named_parameters():
            assert rel_err(p.grad32, eager[name]) < 1e-6, name
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(DEVICE != "cuda", reason="compile tier needs a GPU")
def test_compiled_fp8_is_the_same_body():
    """The shipping config is --fp8 --compile-fwdbwd (fullgraph), so the compiled
    body must trace from a COLD start and be the same computation.

    "Compiled == eager to 1e-4" — the bf16 body's tier — is the WRONG assertion
    here. An e4m3 bucket is 1/8 wide, so any roundoff-level difference between
    the two modes flips a few elements across a boundary and comes back
    amplified: measured at this config, compiled-vs-eager is 1.5e-2 for the bf16
    body and 5-9e-2 for the fp8 body — the same ~6x FP8 applies to everything it
    is handed (agent-ops diag_fp8_noise.log). Asserting tightness there would
    just be asserting that roundoff doesn't exist.

    What must hold instead, and does:
      - the compiled body is deterministic run to run (only the embedding
        scatters move, on atomics, at ~1e-8);
      - the loss agrees closely — the forward GEMMs are the same cuBLAS calls;
      - the fp8-vs-bf16 error profile is the SAME in both modes: compiled fp8
        differs from compiled bf16 exactly as eager fp8 differs from eager bf16
        (measured within ~1%). That is the structural check — a layout or fusion
        bug in the compiled path moves it, amplification cannot."""
    try:
        with compute_dtype(torch.bfloat16):   # the shipping dtype
            model = build_model(depth=4, model_dim=256, n_head=2, seq_len=512,
                                window_pattern="SL", device="cuda", vocab=4096)
            perturb(model)
            idx, targets, cu = make_batch(4096, 2048, 512, "cuda")
            init_grad_buffers(model, dtype=torch.float32)

            def grads(fn):
                zero_grad32(model)
                loss = fn(model, idx, targets, cu, loss_scale=1.0)
                return loss.item(), {n: p.grad32.clone() for n, p in model.named_parameters()}

            c_bf16 = torch.compile(forward_backward, dynamic=False, fullgraph=True)
            c_fp8 = torch.compile(forward_backward_fp8, dynamic=False, fullgraph=True)
            _, e_b = grads(forward_backward)
            loss_e, e_8 = grads(forward_backward_fp8)
            _, k_b = grads(c_bf16)
            loss_c, k_8 = grads(c_fp8)
            _, k_8_again = grads(c_fp8)

        assert abs(loss_c - loss_e) / abs(loss_e) < 2e-3, (loss_e, loss_c)
        scalar_roles = {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"}
        for name, _ in model.named_parameters():
            rerun = rel_err(k_8_again[name], k_8[name])
            r_e, r_c = rel_err(e_8[name], e_b[name]), rel_err(k_8[name], k_b[name])
            print(f"  {name:16s} fp8/bf16 eager {r_e:.3e} compiled {r_c:.3e} | rerun {rerun:.1e}")
            assert rerun < 1e-7, (name, rerun)
            if name not in scalar_roles:   # scalars are noise-on-noise; see the fp32 tier
                assert abs(r_c - r_e) <= 0.25 * max(r_e, 1e-3), (name, r_e, r_c)
    finally:
        torch._dynamo.reset()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
