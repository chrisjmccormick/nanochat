"""Gradient parity: the handwritten train_step.forward_backward vs autograd
through GPT.forward on the SAME banked model (grads land in .grad32 and .grad
respectively, so both can coexist).

Tier 1 (exact): tiny config, fp64 end to end, SDPA/naive attention -> every
    parameter's grad matches autograd to < 1e-8 relative.
Tier 2 (integration): real d12 config, bf16, FA3 raw kernels, one full-size
    packed batch -> per-bank rel err reported (bf16 + attention-atomics noise;
    asserted only at a loose threshold, judged in the session log).

Run: python -m pytest tests/test_grad_parity.py -v -s
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import pytest
import torch

import nanochat.gpt as gpt_mod
import nanochat.flash_attention as fa_mod
from nanochat.gpt import GPT, GPTConfig
from nanochat.train_step import forward_backward, init_grad_buffers, zero_grad32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(depth, model_dim, n_head, seq_len, window_pattern, device, vocab):
    config = GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=depth,
                       n_head=n_head, n_kv_head=n_head, n_embd=model_dim,
                       window_pattern=window_pattern)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    torch.manual_seed(0)
    model.init_weights()
    return model


def perturb(model):
    """The zero-init roles (attn_proj, mlp_proj, smear, backout) kill whole
    backward paths at init; give every parameter a live value so parity
    actually covers the smear chain, backout, and the MLP/attention couplings."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ("attn_proj", "mlp_proj", "smear_gate"):
                scale = 0.5 if name == "smear_gate" else 0.02
                noise = torch.empty(p.shape).uniform_(-scale, scale, generator=g)
                p.add_(noise.to(p.device, p.dtype))
            elif name == "smear_lambda":
                p.fill_(0.3)
            elif name == "backout_lambda":
                p.fill_(0.2)


def make_batch(vocab, T, seq_len, device, seed=3, ragged=False):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (T,), generator=g).to(device)
    targets = torch.randint(0, vocab, (T,), generator=g).to(device)
    ignore = torch.rand(T, generator=g) < 0.02
    targets[ignore.to(device)] = -1
    if not ragged:
        bounds = list(range(0, T + 1, seq_len))
    else:
        bounds, pos = [0], 0
        while pos < T:
            pos = min(T, pos + int(torch.randint(100, seq_len + 1, (1,), generator=g)))
            bounds.append(pos)
    # dataloader convention: pad the tail with repeats of T (zero-length docs)
    bounds += [T] * 4
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    return idx, targets, cu


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


@pytest.fixture
def fp64_sdpa():
    """COMPUTE_DTYPE=fp64 (gpt + train_step read nanochat.gpt's binding) and the
    SDPA/naive attention fallback, restored afterwards."""
    saved = (gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA)
    gpt_mod.COMPUTE_DTYPE = torch.float64
    fa_mod._override_impl = "sdpa"
    fa_mod.USE_FA = fa_mod._resolve_use_fa()
    yield
    gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA = saved


def _tiny_fp64_model():
    model = build_model(depth=4, model_dim=64, n_head=2, seq_len=128,
                        window_pattern="SL", device=DEVICE, vocab=256)
    model.double()
    perturb(model)
    return model


def test_rms_r_matches_aten():
    """_rms_fwd's r must be the 1/rms the F.rms_norm kernel divided by (ATen
    default eps = finfo(accumulate dtype).eps), up to reduction-order ulps —
    checked by reconstructing y as x*r. A wrong eps choice (e.g. bf16's own
    0.0078) fails this by ~1e-2."""
    from nanochat.train_step import _rms_fwd
    tol = {torch.float64: 1e-13, torch.float32: 1e-5, torch.bfloat16: 8e-3}
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        x = (torch.randn(256, 128, dtype=torch.float64, device=DEVICE) * 3).to(dtype)
        y, r = _rms_fwd(x, 128)
        acc = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        y_manual = (x.to(acc) * r).to(dtype)
        d = (y.double() - y_manual.double()).abs().max().item()
        scale = y.double().abs().max().item()
        assert d / scale < tol[dtype], f"rms r mismatch for {dtype}: {d/scale:.3e}"
    # Pin the eps CHOICE where the candidates actually separate: at mean(x^2)
    # ~1e-8, the kernel's divisor with fp32(-upcast) eps reconstructs y exactly,
    # while bf16's own eps (7.8e-3) is ~100% off and eps=0 is ~250% off.
    # (Verified against the CUDA kernel 2026-07-29; guards the _rms_fwd default.)
    x = (torch.randn(1024, 128, dtype=torch.float64, device=DEVICE) * 1e-4).bfloat16()
    y, r = _rms_fwd(x, 128)
    y_manual = (x.float() * r).bfloat16()
    d = (y.double() - y_manual.double()).abs().max().item()
    scale = y.double().abs().max().item()
    assert d / scale < 1e-2, f"rms eps choice no longer matches the kernel: {d/scale:.3e}"


def test_grad_parity_exact_fp64(fp64_sdpa):
    model = _tiny_fp64_model()
    idx, targets, cu = make_batch(256, 256, 128, DEVICE)

    loss_ref = model(idx, cu, targets)
    loss_ref.backward()
    init_grad_buffers(model, dtype=torch.float64)
    loss = forward_backward(model, idx, targets, cu, loss_scale=1.0)

    assert abs(loss.item() - loss_ref.item()) / abs(loss_ref.item()) < 1e-12
    for name, p in model.named_parameters():
        r = rel_err(p.grad32, p.grad)
        print(f"  {name:16s} rel_err {r:.3e}")
        assert r < 1e-8, (name, r)


def test_grad_accumulation_fp64(fp64_sdpa):
    """Two micro-batches at loss_scale=1/2 must equal autograd on (l1+l2)/2."""
    model = _tiny_fp64_model()
    b1 = make_batch(256, 256, 128, DEVICE, seed=11)
    b2 = make_batch(256, 256, 128, DEVICE, seed=12)

    loss_ref = model(b1[0], b1[2], b1[1]) / 2 + model(b2[0], b2[2], b2[1]) / 2
    loss_ref.backward()
    init_grad_buffers(model, dtype=torch.float64)
    forward_backward(model, b1[0], b1[1], b1[2], loss_scale=0.5)
    forward_backward(model, b2[0], b2[1], b2[2], loss_scale=0.5)

    for name, p in model.named_parameters():
        r = rel_err(p.grad32, p.grad)
        assert r < 1e-8, (name, r)
    # and zero_grad32 actually zeroes
    zero_grad32(model)
    assert all(p.grad32.abs().max() == 0 for p in model.parameters())


@pytest.mark.skipif(DEVICE != "cuda", reason="needs a GPU (d12 sizes)")
def test_grad_parity_fp32_sdpa_d12():
    """d12 config in fp32 with the naive/SDPA attention: the strong guard for
    the scalar parameters, whose grads are cancellation-heavy reductions that
    the bf16 tier can only check loosely. Everything lands ~1e-6 here."""
    saved = (gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA)
    try:
        gpt_mod.COMPUTE_DTYPE = torch.float32
        fa_mod._override_impl = "sdpa"
        fa_mod.USE_FA = fa_mod._resolve_use_fa()
        model = build_model(depth=12, model_dim=768, n_head=6, seq_len=2048,
                            window_pattern="SSSL", device="cuda", vocab=32768)
        perturb(model)
        idx, targets, cu = make_batch(32768, 8192, 2048, "cuda")
        loss_ref = model(idx, cu, targets)
        loss_ref.backward()
        # explicit fp32 buffers: this tier is the precise guard, and under the
        # fp32 COMPUTE_DTYPE patch the autograd reference grads are fp32 too
        init_grad_buffers(model, dtype=torch.float32)
        loss = forward_backward(model, idx, targets, cu, loss_scale=1.0)
        assert abs(loss.item() - loss_ref.item()) / abs(loss_ref.item()) < 1e-6
        for name, p in model.named_parameters():
            r = rel_err(p.grad32, p.grad)
            print(f"  {name:16s} rel_err {r:.3e}")
            assert r < 1e-4, (name, r)
    finally:
        gpt_mod.COMPUTE_DTYPE, fa_mod._override_impl, fa_mod.USE_FA = saved


@pytest.mark.skipif(DEVICE != "cuda" or fa_mod.FA_VERSION != "fa3",
                    reason="integration tier needs the FA3 GPU path")
def test_grad_parity_integration_d12():
    """Real d12 config, bf16, FA3, one packed batch (32768 tokens — the eager
    autograd REFERENCE pass OOMs at the full 131072; per-token shapes/kernels
    are identical). bf16 puts the matrix banks' noise floor around 5e-3; the
    scalar params (resid/x0/backout lambdas) sit atop cancellation-heavy
    reductions and can reach a few 1e-1 of bf16 noise — they are guarded
    tightly by the fp32 tier above, and only sanity-bounded here."""
    import time
    model = build_model(depth=12, model_dim=768, n_head=6, seq_len=2048,
                        window_pattern="SSSL", device="cuda", vocab=32768)
    perturb(model)
    idx, targets, cu = make_batch(32768, 32768, 2048, "cuda", ragged=True)

    # warm both paths first — cold timings are dominated by lazy kernel init
    model(idx, cu, targets).backward()
    model.zero_grad(set_to_none=True)
    init_grad_buffers(model)
    forward_backward(model, idx, targets, cu, loss_scale=1.0, deterministic_attention=True)
    zero_grad32(model)

    torch.cuda.synchronize(); t0 = time.time()
    loss_ref = model(idx, cu, targets)
    loss_ref.backward()
    torch.cuda.synchronize(); t_autograd = time.time() - t0

    torch.cuda.synchronize(); t0 = time.time()
    loss = forward_backward(model, idx, targets, cu, loss_scale=1.0,
                            deterministic_attention=True)
    torch.cuda.synchronize(); t_manual = time.time() - t0

    print(f"\n  loss autograd {loss_ref.item():.6f} | handwritten {loss.item():.6f}")
    print(f"  eager fwd+bwd time (warm, 32768 tok): autograd {t_autograd*1e3:.0f}ms | handwritten {t_manual*1e3:.0f}ms")
    scalar_roles = {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"}
    for name, p in model.named_parameters():
        r = rel_err(p.grad32, p.grad)
        print(f"  {name:16s} rel_err {r:.3e}")
        assert r < (0.5 if name in scalar_roles else 0.05), (name, r)
    assert abs(loss.item() - loss_ref.item()) / abs(loss_ref.item()) < 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
