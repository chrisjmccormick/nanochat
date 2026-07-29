"""Gate C: the explicit optimizer flow (train_step.py) vs the old MuonAdamW.

- Mantissa invariants: split -> reconstruct is bit-exact; after any step the
  bf16 live param equals the truncated top 16 bits of its master.
- Step parity: identical synthetic grads through the old MuonAdamW (fp32-live
  params, per-layer slices grouped per ROLE so the batched-GEMM shapes match
  the banks) and the new explicit step; reconstructed masters agree ~1e-6.
- Schedule tables: the named builders reproduce schedules._build_* coefficients
  exactly (with the Muon aspect fold moved to the (K,1,1) multipliers).

Run: python -m pytest tests/test_optimizer_step.py -v -s   (needs a GPU)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_grad_parity import build_model, perturb  # noqa: E402

from nanochat.optim import MuonAdamW  # noqa: E402
from nanochat.schedules import Ramp, AdamWGroup, MuonGroup, build_param_groups  # noqa: E402
from nanochat import train_step as ts  # noqa: E402

# Scoped per test rather than module-wide: the fused kernels need CUDA, but the
# shape agreement between the allocator and the kernel is pure bookkeeping and
# must be checkable anywhere (it is what broke d24).
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="optimizer kernels are CUDA")

DEV = "cuda"
N_STEPS = 1000   # schedule length (Muon momentum's 400-step warmup must fit); we run 3


@needs_cuda
def test_mantissa_invariants():
    x = torch.randn(4096, device=DEV) * torch.logspace(-20, 20, 4096, device=DEV)
    p = torch.nn.Parameter(x.clone())
    mant = ts._split_master(p)
    rec = ts._master(p.data, mant)
    assert torch.equal(rec.view(torch.int32), x.view(torch.int32))  # split->reconstruct bit-exact
    # a perturbed master splits losslessly and live == truncated(master)
    m2 = rec * 1.000123 + 1e-6
    live, mant2 = torch.empty_like(p.data), torch.empty_like(mant)
    ts._writeback(m2, live, mant2)
    assert torch.equal(ts._master(live, mant2).view(torch.int32), m2.view(torch.int32))
    assert torch.equal(live.view(torch.int16), (m2.view(torch.int32) >> 16).to(torch.int16))


def _hyper():
    return dict(embedding_lr=0.3, unembedding_lr=0.008, matrix_lr=0.02, scalar_lr=0.5,
                weight_decay=0.05, batch_lr_scale=1.0, warmup_steps=2,
                warmdown_ratio=0.5, final_lr_frac=0.05)


def _old_specs(pl, model_dim, h):
    """EXACT mirror of the base_train.py spec block, per-ROLE Muon groups (same
    batched shapes as the banks, so kernel rounding matches slice for slice)."""
    adamw_lr_scale = h["batch_lr_scale"] * (model_dim / 768) ** -0.5
    lrm = Ramp(peak=1.0, start=0.0, warmup_steps=h["warmup_steps"],
               end=h["final_lr_frac"], cooldown_frac=h["warmdown_ratio"])
    muon_momentum = Ramp(peak=0.97, start=0.85, warmup_steps=400,
                         end=0.90, cooldown_frac=h["warmdown_ratio"])
    muon_wd = Ramp(peak=h["weight_decay"], end=0.0, cooldown_frac=1.0, shape="cosine")
    specs = [
        AdamWGroup(pl["lm_head"],      lr=lrm * (h["unembedding_lr"] * adamw_lr_scale),     betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
        AdamWGroup(pl["wte"],          lr=lrm * (h["embedding_lr"] * adamw_lr_scale),       betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        AdamWGroup(pl["value_embeds"], lr=lrm * (h["embedding_lr"] * adamw_lr_scale * 0.5), betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        AdamWGroup(pl["resid"],        lr=lrm * (h["scalar_lr"] * h["batch_lr_scale"] * 0.01), betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        AdamWGroup(pl["x0"],           lr=lrm * (h["scalar_lr"] * h["batch_lr_scale"]),     betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        AdamWGroup(pl["smear"],        lr=lrm * 0.2,                                        betas=(0.8, 0.95),  eps=1e-10, weight_decay=0.0),
    ]
    for role in ("c_q", "c_k", "c_v", "attn_proj", "mlp_fc", "mlp_proj", "ve_gate"):
        specs.append(MuonGroup(pl[role], lr=lrm * (h["matrix_lr"] * h["batch_lr_scale"]),
                               momentum=muon_momentum, beta2=0.9,
                               weight_decay=muon_wd, ns_steps=5))
    return specs


def _run_step_parity(compiled):
    h = _hyper()
    model = build_model(depth=4, model_dim=64, n_head=2, seq_len=128,
                        window_pattern="SL", device=DEV, vocab=256)
    perturb(model)

    # fp32-live reference copies, taken BEFORE the masters are split
    banks = ("c_q", "c_k", "c_v", "attn_proj", "mlp_fc", "mlp_proj", "ve_gate")
    pl = {role: [torch.nn.Parameter(getattr(model, role)[i].detach().float().clone())
                 for i in range(getattr(model, role).shape[0])] for role in banks}
    singles = ("lm_head", "wte", "resid_lambdas", "x0_lambdas",
               "smear_gate", "smear_lambda", "backout_lambda")
    ref = {name: torch.nn.Parameter(getattr(model, name).detach().float().clone()) for name in singles}
    pl["lm_head"], pl["wte"] = [ref["lm_head"]], [ref["wte"]]
    pl["value_embeds"] = [torch.nn.Parameter(model.value_embeds[i].detach().float().clone())
                          for i in range(model.value_embeds.shape[0])]
    pl["resid"], pl["x0"] = [ref["resid_lambdas"]], [ref["x0_lambdas"]]
    pl["smear"] = [ref["smear_gate"], ref["smear_lambda"], ref["backout_lambda"]]

    old = MuonAdamW(build_param_groups(_old_specs(pl, 64, h), num_steps=N_STEPS))

    ts.init_optimizer_state(model)
    sched = ts.build_schedules(model_dim=64, num_iterations=N_STEPS, device=DEV, **h)
    muls = ts.bank_muls(model)
    t = ts.make_step_counter(DEV)

    g = torch.Generator(device=DEV)
    if not compiled:
        torch._dynamo.config.disable = True  # both sides run their original eager math
    try:
        for step in range(3):
            g.manual_seed(100 + step)
            for name, p in model.named_parameters():
                p.grad32.copy_(torch.randn(p.shape, generator=g, device=DEV) * 1e-3)
            # mirror the same grads into the old optimizer's per-layer params
            for role in banks:
                bank_grad = getattr(model, role).grad32
                for i, q in enumerate(pl[role]):
                    q.grad = bank_grad[i].clone()
            # .float(): the embedding grad buffers are bf16 now; the fp32-live
            # reference gets the same VALUES upcast (the kernel upcasts too)
            for i, q in enumerate(pl["value_embeds"]):
                q.grad = model.value_embeds.grad32[i].clone().float()
            for name in singles:
                ref[name].grad = getattr(model, name).grad32.clone().float()

            old.step()
            ts.optimizer_step(model, sched, muls, t)  # NOTE: mutates grad32 (nesterov lerp)
            ts.zero_grad32(model)
    finally:
        if not compiled:
            torch._dynamo.config.disable = False
            torch._dynamo.reset()  # drop the skip-frame decisions, or later tests silently run eager

    assert t.item() == 3
    worst = 0.0
    for name, p in model.named_parameters():
        # scalar params are fp32-LIVE (no mantissa pair) — compare directly
        rec = ts._master(p.data, p.mantissa) if hasattr(p, "mantissa") else p.detach().float()
        if name in banks or name == "value_embeds":
            refv = torch.stack([q.detach() for q in pl[name if name in banks else "value_embeds"]])
        else:
            refv = ref[name].detach()
        r = ((rec.double() - refv.double()).norm() / refv.double().norm()).item()
        worst = max(worst, r)
        print(f"  {name:16s} master rel_err {r:.3e}")
        if hasattr(p, "mantissa"):
            # live/mantissa pairing stays lossless after real steps
            assert torch.equal(p.data.view(torch.int16),
                               (rec.view(torch.int32) >> 16).to(torch.int16)), name
    print(f"  worst {worst:.3e}")
    return worst


@needs_cuda
def test_step_parity_math_eager():
    """Both flows run their ORIGINAL eager math on identical grads: the new
    mantissa-master step must reproduce the old fp32-live step essentially
    bit-for-bit (measured 0.0 on the GH200)."""
    assert _run_step_parity(compiled=False) < 1e-6


@needs_cuda
def test_step_parity_compiled():
    """Same comparison through the shipped torch.compile kernels. Inductor
    rounds each graph's bf16 polar-express region differently (the OLD compiled
    kernel differs from its own eager math by ~3e-3/step the same way), so this
    tier only bounds that noise (~3e-2 over 3 steps at these tiny 64x64 slices);
    the math identity is the eager test above."""
    assert _run_step_parity(compiled=True) < 5e-2


@needs_cuda
def test_schedule_tables_match_old_builder():
    h = _hyper()
    sched = ts.build_schedules(model_dim=64, num_iterations=N_STEPS, device=DEV, **h)
    # dummy params to drive the old shape-folding builder
    dummy_tall = [torch.nn.Parameter(torch.empty(8, 4, device=DEV))]   # fold sqrt(2)? no: 8/4=2 -> sqrt(2)
    dummy_sq = [torch.nn.Parameter(torch.empty(4, 4, device=DEV))]
    dummy_lm = [torch.nn.Parameter(torch.empty(64, 8, device=DEV))]
    adamw_lr_scale = h["batch_lr_scale"] * (64 / 768) ** -0.5
    lrm = Ramp(peak=1.0, start=0.0, warmup_steps=h["warmup_steps"],
               end=h["final_lr_frac"], cooldown_frac=h["warmdown_ratio"])
    muon_momentum = Ramp(peak=0.97, start=0.85, warmup_steps=400,
                         end=0.90, cooldown_frac=h["warmdown_ratio"])
    muon_wd = Ramp(peak=h["weight_decay"], end=0.0, cooldown_frac=1.0, shape="cosine")
    groups = build_param_groups([
        AdamWGroup(dummy_lm, lr=lrm * (h["unembedding_lr"] * adamw_lr_scale), betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        MuonGroup(dummy_sq, lr=lrm * (h["matrix_lr"] * h["batch_lr_scale"]), momentum=muon_momentum, beta2=0.9, weight_decay=muon_wd, ns_steps=5),
        MuonGroup(dummy_tall, lr=lrm * (h["matrix_lr"] * h["batch_lr_scale"]), momentum=muon_momentum, beta2=0.9, weight_decay=muon_wd, ns_steps=5),
    ], num_steps=N_STEPS)
    # AdamW: identical coefficients, eps having left the tables
    old_a = groups[0]["tabs"]
    for f in ("wd_mul", "one_minus_beta1", "one_minus_beta2", "rsqrt_bias2", "step_size"):
        assert torch.equal(getattr(old_a, f), getattr(sched.lm_head, f)), f
    # Muon: square group (fold 1) == the shared canonical table
    old_sq = groups[1]["tabs"]
    for f in ("momentum", "one_minus_momentum", "one_minus_beta2", "lr", "lr_wd"):
        assert torch.equal(getattr(old_sq, f), getattr(sched.matrix, f)), f
    # Muon: tall group's folded lr == canonical lr * the bank's aspect multiplier
    # (sqrt(2) here is irrational -> fp32-ulp tolerance; the real banks' aspects
    # are exactly 1.0 or 2.0, where the two routes agree bit-for-bit)
    old_tall = groups[2]["tabs"]
    aspect = max(1.0, 8 / 4) ** 0.5
    assert torch.allclose(old_tall.lr, sched.matrix.lr * aspect, rtol=1e-6, atol=0)
    assert torch.allclose(old_tall.lr_wd, sched.matrix.lr_wd * aspect, rtol=1e-6, atol=0)


def test_second_moment_shape_matches_red_dim_every_depth():
    """NorMuon's factored second moment is (K, out, 1) for a tall bank and
    (K, 1, in) for a wide one, so the buffer init_optimizer_state ALLOCATES has
    to follow the same rule the kernel REDUCES by. It didn't for ve_gate, which
    hardcoded the wide shape: ve_gate slices are (n_kv_head, 12), wide at d12/d20
    but square at d24 and tall at d26, so d24+ died in muon_step_fused's lerp_
    with "output with shape [12, 1, 12] doesn't match the broadcast shape
    (12, 12, 12)" — the first thing ever to run this trainer above d12.

    Checked here on shapes alone, across every depth in play, because the bug is
    a shape disagreement and needs no GPU, no compile and no step to expose."""
    from nanochat.train_step import red_dim, second_moment_shape

    class FakeBank:
        def __init__(self, shape): self.shape = shape

    for depth in (12, 20, 24, 26, 32):
        model_dim = ((depth * 64 + 127) // 128) * 128
        n_kv_head = model_dim // 128
        n_ve = (depth + 1) // 2
        banks = {
            "c_q":       FakeBank((depth, model_dim, model_dim)),
            "mlp_fc":    FakeBank((depth, 4 * model_dim, model_dim)),
            "mlp_proj":  FakeBank((depth, model_dim, 4 * model_dim)),
            "ve_gate":   FakeBank((n_ve, n_kv_head, 12)),
        }
        for name, bank in banks.items():
            k = bank.shape[0]
            got = second_moment_shape(bank, k)
            rd = red_dim(bank)
            # the reduced dim must be the one that collapses to 1
            assert got[rd] == 1, (depth, name, got, rd)
            # and the OTHER matrix dim must be preserved in full
            other = -1 if rd == -2 else -2
            assert got[other] == bank.shape[other], (depth, name, got, rd)
            assert got[0] == k, (depth, name, got)
    # the specific case that broke: d24's ve_gate is square, and a square bank
    # takes the tall branch (>=), so the buffer must be (K, out, 1)
    assert red_dim(FakeBank((12, 12, 12))) == -1
    assert second_moment_shape(FakeBank((12, 12, 12)), 12) == (12, 12, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
