"""Handwritten training step for base pretraining: explicit forward+backward
(no autograd) over the banked GPT, accumulating into fp32 `.grad32` buffers.

forward_backward() is the 5th deliberately-duplicated forward body (after
GPT.forward, GPT.forward_inference, fast_engine's decode/prefill) — its forward
half mirrors GPT.forward line for line, then the backward half walks the same
ops in reverse. GPT.forward stays autograd-capable and is the gradient-parity
oracle (tests/test_grad_parity.py). forward_backward_fp8() is the 6th: the same
body with its 21 matrix-bank/lm_head GEMMs routed through torch._scaled_mm
(--fp8; see nanochat/fp8.py and tests/test_fp8_step.py).

Design ledger (see agent-ops nanochat/2026-07-28_0449pm_handwritten-fwd-bwd/PLAN.md):
- Gradients accumulate into `p.grad32` — full-size, persistent, explicitly
  zeroed, fp32 by default. NOT `.grad`, so autograd through forward() can still
  run on the same model (eval/parity) without colliding.
- Attention runs through the raw FA3 ops (flash_attention.flash_attn_varlen_fwd_lse
  / flash_attn_varlen_bwd), stashing out+LSE; non-bf16/no-FA falls back to the
  naive materialized implementation inside those helpers (tests only).
- rms_norms: we stash the norm OUTPUT plus the per-vector 1/rms `r`. In output
  space the backward is dx = r*(dy - y*mean(y*dy)) for ANY eps, so the pre-norm
  input is never needed. Cheap norms (the MLP-side xm) are recomputed from the
  stashed pre-norm x1 instead of stashed.
- Weight-grad matmuls run in the compute dtype (bf16), then accumulate as
  `.float()` into grad32 — the same numerics autograd produces for a bf16
  matmul whose weight lives in fp32.
- loss_scale (1/grad_accum_steps) replaces the loss division of the autograd
  loop; the returned loss is the plain (unscaled) mean CE for logging.
"""

from typing import NamedTuple
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# Read via the gpt module so the parity tests' COMPUTE_DTYPE monkeypatch (fp64
# tier) applies to GPT.forward and this file in one place.
from nanochat import gpt as gpt_mod
from nanochat import fp8
from nanochat.fp8 import E4M3, E5M2
from nanochat.flash_attention import flash_attn_varlen_fwd_lse, flash_attn_varlen_bwd
from nanochat.optim import polar_express_coeffs
from nanochat.schedules import Ramp, MuonCoeffs, _as_table, _bias_correction, _tables


# -----------------------------------------------------------------------------
# grad32 buffers

def init_grad_buffers(model, dtype=None):
    """Attach a full-size, zeroed `.grad32` to every parameter. Call once, after
    allocation; zero with zero_grad32() between steps.

    Written out one tensor per block so the dtype choice and the memory cost of
    every gradient buffer are readable here rather than implied by a rule.

    Gradients are fp32 everywhere EXCEPT the token/value embeddings, which
    accumulate in bf16: they are the two biggest tensors in the model, fp32
    grads would double their scatter traffic and (world>1) comm bytes, and bf16
    matches the autograd baseline's numerics anyway (bf16 params -> bf16 .grad).
    An explicit `dtype` overrides everything -- that is the fp64 parity tier.

    The 3-D banks also get `grad32_slices`: per-slice VIEWS built OUTSIDE any
    graph. The forward/backward bodies accumulate through these, never through
    `grad32[i]` -- an in-graph bank slice functionalizes into a whole-bank
    select_scatter copy (10-20x the cost of the slice add at speedrun bank
    sizes), while a view created out of graph arrives as an input and mutates
    genuinely in place.
    """
    d = model.dims
    dev = model.wte.device
    g = torch.float32 if dtype is None else dtype     # matrix / scalar gradients
    ge = torch.bfloat16 if dtype is None else dtype   # embedding gradients

    # Embeddings and unembedding.
    model.wte.grad32          = torch.zeros(d.vocab, d.d_model, dtype=ge, device=dev)
    model.lm_head.grad32      = torch.zeros(d.vocab, d.d_model, dtype=g,  device=dev)
    model.value_embeds.grad32 = torch.zeros(d.ve_slot, d.vocab, d.kv, dtype=ge, device=dev)
    model.value_embeds.grad32_slices = list(model.value_embeds.grad32.unbind(0))

    # Attention banks.
    model.c_q.grad32       = torch.zeros(d.layer, d.q,       d.d_model, dtype=g, device=dev)
    model.c_k.grad32       = torch.zeros(d.layer, d.kv,      d.d_model, dtype=g, device=dev)
    model.c_v.grad32       = torch.zeros(d.layer, d.kv,      d.d_model, dtype=g, device=dev)
    model.attn_proj.grad32 = torch.zeros(d.layer, d.d_model, d.q,       dtype=g, device=dev)
    model.c_q.grad32_slices       = list(model.c_q.grad32.unbind(0))
    model.c_k.grad32_slices       = list(model.c_k.grad32.unbind(0))
    model.c_v.grad32_slices       = list(model.c_v.grad32.unbind(0))
    model.attn_proj.grad32_slices = list(model.attn_proj.grad32.unbind(0))

    # MLP banks.
    model.mlp_fc.grad32   = torch.zeros(d.layer, d.mlp,     d.d_model, dtype=g, device=dev)
    model.mlp_proj.grad32 = torch.zeros(d.layer, d.d_model, d.mlp,     dtype=g, device=dev)
    model.mlp_fc.grad32_slices   = list(model.mlp_fc.grad32.unbind(0))
    model.mlp_proj.grad32_slices = list(model.mlp_proj.grad32.unbind(0))

    # VE gates.
    model.ve_gate.grad32 = torch.zeros(d.ve_slot, d.n_kv_head, d.gate_ch, dtype=g, device=dev)
    model.ve_gate.grad32_slices = list(model.ve_gate.grad32.unbind(0))

    # Scalars.
    model.resid_lambdas.grad32  = torch.zeros(d.layer, dtype=g, device=dev)
    model.x0_lambdas.grad32     = torch.zeros(d.layer, dtype=g, device=dev)
    model.smear_gate.grad32     = torch.zeros(1, d.smear_ch, dtype=g, device=dev)
    model.smear_lambda.grad32   = torch.zeros(1, dtype=g, device=dev)
    model.backout_lambda.grad32 = torch.zeros(1, dtype=g, device=dev)


def zero_grad32(model):
    for p in model.parameters():
        p.grad32.zero_()


# -----------------------------------------------------------------------------
# rms_norm forward/backward in output space

def _rms_fwd(x, width):
    """F.rms_norm plus the per-vector 1/rms its backward needs. `r` is computed
    in fp32 (fp64 stays fp64) with ATen's default eps, matching what the kernel
    divided by (verified in tests/test_grad_parity.py)."""
    y = F.rms_norm(x, (width,))
    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    eps = torch.finfo(xf.dtype).eps
    r = (xf.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return y, r


def _rms_bwd(dy, y, r):
    """dx = r*(dy - y*mean(y*dy)): exact for any eps because r is the forward's
    actual 1/rms and y the actual output (substitute x = y/r in the usual form)."""
    acc = dy.dtype if dy.dtype in (torch.float32, torch.float64) else torch.float32
    yf, dyf = y.to(acc), dy.to(acc)
    dx = r * (dyf - yf * (yf * dyf).mean(dim=-1, keepdim=True))
    return dx.to(dy.dtype)


def _rms_bwd_scaled(dy, ys, r, s):
    """Backward through ys = s * rms_norm(x), given the SCALED output ys — which
    is exactly what the attention kernel consumed, so it stashes directly with
    no recompute pass. Substituting y = ys/s into _rms_bwd's form:
    dx = r*(s*dy - ys*mean(ys*dy)/s). Exact algebra (fp64 tier verifies)."""
    acc = dy.dtype if dy.dtype in (torch.float32, torch.float64) else torch.float32
    yf, dyf = ys.to(acc), dy.to(acc)
    dx = r * (s * dyf - yf * ((yf * dyf).mean(dim=-1, keepdim=True) / s))
    return dx.to(dy.dtype)


# -----------------------------------------------------------------------------
# landing bank gradients

def _land_scalar_grads(model, g_resid, g_x0):
    """Land the per-layer resid/x0 scalar sums (collected in REVERSED layer
    order) as one stacked add each. Matrix/VE/gate banks don't come through
    here any more — their wgrads add directly into `grad32_slices[i]`, the
    per-slice views init_grad_buffers creates OUTSIDE the graph (an in-graph
    `bank.grad32[i].add_` functionalizes into a whole-bank select_scatter
    copy; a pre-made view input mutates in place)."""
    model.resid_lambdas.grad32.add_(torch.stack(g_resid[::-1]))
    model.x0_lambdas.grad32.add_(torch.stack(g_x0[::-1]))


# -----------------------------------------------------------------------------
# forward_backward

@torch.no_grad()
def forward_backward(model, idx, targets, cu_seqlens, loss_scale=1.0,
                     deterministic_attention=False, loss_chunk=8192):
    """One micro-batch: forward, stash, explicit backward into `.grad32`.
    Returns the detached mean CE loss (unscaled; grads carry loss_scale)."""
    cfg = model.config
    assert idx.ndim == 1
    T = idx.size(0)
    nl = cfg.n_layer
    nh, nkv, hd = cfg.n_head, cfg.n_kv_head, model.head_dim
    D = cfg.n_embd
    half = hd // 2
    V = cfg.vocab_size
    Vp = model.padded_vocab_size
    max_seq_len = cfg.sequence_len
    dt = gpt_mod.COMPUTE_DTYPE
    g32 = model.c_q.grad32.dtype       # matrix/scalar grad dtype (fp32; fp64 in the exact tier)
    gemb = model.wte.grad32.dtype      # embedding grad dtype (bf16 by default — see init_grad_buffers)
    gch = model.ve_gate_channels

    assert T > 1, "Training forward pass should have T > 1"
    assert T <= model.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {model.cos.size(1)}"
    cos, sin = model.cos[0, :T], model.sin[0, :T]  # (T, 1, half)

    # ==== forward half (mirrors GPT.forward — keep the two visibly line-parallel) ====
    x = F.embedding(idx, model.wte)
    x = x.to(dt)
    xe, r_e = _rms_fwd(x, D)                     # post-norm embedding, pre-smear

    # Smear
    gate = model.smear_lambda.to(dt) * torch.sigmoid(xe[1:, :24] @ model.smear_gate.to(dt).mT)
    x = torch.cat([xe[:1], xe[1:] + gate * xe[:-1]], dim=0)

    x0 = x
    backout_layer = nl // 2
    x_backout = None
    stash = []
    for i in range(nl):
        x_in = x
        b = model.resid_lambdas[i] * x_in + model.x0_lambdas[i] * x0
        xn, r_xn = _rms_fwd(b, D)
        q = (xn @ model.c_q[i].to(dt).mT).view(T, nh, hd)
        k = (xn @ model.c_k[i].to(dt).mT).view(T, nkv, hd)
        v = (xn @ model.c_v[i].to(dt).mT).view(T, nkv, hd)
        j = model.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, model.value_embeds[j]).view(T, nkv, hd).to(dt)
            g = 3 * torch.sigmoid(xn[..., :gch] @ model.ve_gate[j].to(dt).mT)
            v = v + g.unsqueeze(-1) * ve         # ve/g recomputed in backward, not stashed
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        qn, r_q = _rms_fwd(q, hd)
        kn, r_k = _rms_fwd(k, hd)
        qf = qn * 1.2                            # stash the SCALED q/k (the kernel's inputs);
        kf = kn * 1.2                            # backward folds the 1.2 via _rms_bwd_scaled
        y, lse = flash_attn_varlen_fwd_lse(qf, kf, v, cu_seqlens, max_seq_len, model.window_sizes[i])
        y = y.contiguous()
        x1 = b + y.view(T, -1) @ model.attn_proj[i].to(dt).mT
        xm, _ = _rms_fwd(x1, D)                  # xm recomputed in backward from stashed x1
        a = F.relu(xm @ model.mlp_fc[i].to(dt).mT)
        x = x1 + a.square() @ model.mlp_proj[i].to(dt).mT
        if i == backout_layer:
            x_backout = x
        stash.append(dict(x_in=x_in, xn=xn, r_xn=r_xn, qf=qf, kf=kf, r_q=r_q, r_k=r_k,
                          v=v, y=y, lse=lse, x1=x1, a=a))

    x_pre = x - model.backout_lambda.to(dt) * x_backout
    xf, r_f = _rms_fwd(x_pre, D)

    # lm_head + softcap + CE loss + dlogits. Two implementations of the same
    # math, split by mode:
    #  - eager: IN PLACE on the one (T, V) fp32 buffer, row-chunked so temps
    #    stay chunk-sized (no second full-size fp32 tensor);
    #  - compiled: a plain out-of-place version — under inductor the in-place
    #    chunk loop inverts into a liability (functionalization re-materializes
    #    the buffer; 16 unrolled subgraphs block fusion), so let the compiler
    #    do the memory planning and fusion it's good at.
    softcap = 15.0
    buf_dtype = torch.float64 if dt == torch.float64 else torch.float32
    valid = targets >= 0
    n_valid = valid.sum()
    if torch.compiler.is_compiling():
        # Written for inductor's fusion, not for readability of the in-place
        # eager twin below: t is an explicit CSE target (materialize once, no
        # tanh recompute in the dz pass), and the onehot is a broadcast compare
        # (a scatter_add here forces an extra full pass over the buffer).
        # (An in-graph row-chunked fwd+bwd was tried 2026-07-29 and reverted:
        # with no dependencies between chunks, inductor schedules their buffers
        # to coexist — no memory saved, 0.6% slower. Real chunking needs a
        # graph BOUNDARY per chunk: a separately-compiled chunk fn in a Python
        # loop, worth building only when a config is actually memory-bound.)
        lm = model.lm_head.to(dt)
        logits = xf @ lm.mT                      # (T, Vp) compute-dtype
        t = torch.tanh(logits[..., :V].to(buf_dtype) / softcap)
        cap = softcap * t
        vmask = valid.to(buf_dtype)
        y_safe = targets.clamp_min(0).unsqueeze(1)
        cap_y = cap.gather(1, y_safe).squeeze(1)
        m = cap.amax(dim=1, keepdim=True)
        e = (cap - m).exp()
        ssum = e.sum(dim=1, keepdim=True)
        lse = (ssum.log() + m).squeeze(1)
        loss = ((lse - cap_y) * vmask).sum() / n_valid.to(buf_dtype)
        onehot = torch.arange(V, device=targets.device).unsqueeze(0) == y_safe
        dz = ((e / ssum - onehot.to(buf_dtype)) * (1.0 - t * t)
              * (vmask * (loss_scale / n_valid.to(buf_dtype))).unsqueeze(1)).to(dt)
        del logits
        g_lm = (dz.mT @ xf).to(g32)              # padded rows get no grad, as in autograd
        (model.lm_head.grad32 if V == Vp else model.lm_head.grad32[:V]).add_(g_lm)
        dxf = dz @ lm[:V]
        del dz
    else:
        logits = xf @ model.lm_head.to(dt).mT    # (T, Vp) compute-dtype
        buf = logits[..., :V].to(buf_dtype)      # THE big fp32 buffer; bf16 logits freed below
        buf.div_(softcap).tanh_().mul_(softcap)  # buf = cap = 15*tanh(z/15)
        # Softcap keeps cap in [-15, 15], so softmax probs are bounded below by
        # ~exp(-30)/V — no underflow anywhere in the in-place chain that follows.
        loss_sum = torch.zeros((), dtype=buf_dtype, device=buf.device)
        scale = loss_scale / n_valid.to(buf_dtype)   # 0-dim device tensor; no host sync
        for r0 in range(0, T, loss_chunk):
            c = buf[r0:r0 + loss_chunk]          # in-place view; temps are chunk-sized
            vc = valid[r0:r0 + loss_chunk].to(buf_dtype)
            y_safe = targets[r0:r0 + loss_chunk].clamp_min(0).unsqueeze(1)
            cap_y = c.gather(1, y_safe).squeeze(1)
            m = c.amax(dim=1, keepdim=True)
            c.sub_(m).exp_()                     # c = exp(cap - m)
            ssum = c.sum(dim=1, keepdim=True)
            loss_sum += ((ssum.log() + m).squeeze(1) - cap_y).mul_(vc).sum()
            f = c.log().add_(m)                  # f = cap, recovered exactly
            f.div_(softcap).square_().neg_().add_(1.0)   # f = 1 - tanh^2(z/15)
            c.div_(ssum)                         # c = softmax(cap)
            c.scatter_add_(1, y_safe, (-vc).unsqueeze(1))  # - onehot on valid rows
            c.mul_(f).mul_(vc.unsqueeze(1) * scale)  # softcap chain, ignore-mask, 1/n_valid, loss_scale
        loss = loss_sum / n_valid.to(buf_dtype)
        del logits
        dz = buf.to(dt)                          # mirror autograd's cast back through .float()
        del buf
        lm = model.lm_head.to(dt)
        g_lm = (dz.mT @ xf).to(g32)              # padded rows get no grad, as in autograd
        (model.lm_head.grad32 if V == Vp else model.lm_head.grad32[:V]).add_(g_lm)
        dxf = dz @ lm[:V]
        del dz

    # ==== backward half ====
    # Bank wgrads add directly into grad32_slices views; only the per-layer
    # scalar sums are collected and landed stacked (_land_scalar_grads).
    g_resid = []; g_x0 = []

    d_pre = _rms_bwd(dxf, xf, r_f)
    model.backout_lambda.grad32.add_(-(d_pre * x_backout).sum(dtype=g32))
    d_stream = d_pre                             # grad wrt layer nl-1's output
    d_x0 = torch.zeros_like(x0)
    for i in reversed(range(nl)):
        st = stash[i]
        if i == backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes nl//2
            d_stream = d_stream - model.backout_lambda.to(dt) * d_pre
        # --- MLP backward (relu^2: dh = 2*a*du, self-masking since a = relu(h)) ---
        x1, a = st["x1"], st["a"]
        d_u = d_stream @ model.mlp_proj[i].to(dt)
        model.mlp_proj.grad32_slices[i].add_(d_stream.mT @ a.square())
        d_h = 2.0 * a * d_u
        xm, r_xm = _rms_fwd(x1, D)               # cheap recompute (bitwise: same input)
        model.mlp_fc.grad32_slices[i].add_(d_h.mT @ xm)
        d_xm = d_h @ model.mlp_fc[i].to(dt)
        d_x1 = d_stream + _rms_bwd(d_xm, xm, r_xm)
        # --- attention backward ---
        xn, y = st["xn"], st["y"]
        model.attn_proj.grad32_slices[i].add_(d_x1.mT @ y.view(T, -1))
        d_y = (d_x1 @ model.attn_proj[i].to(dt)).view(T, nh, hd)
        dqf, dkf, dv = flash_attn_varlen_bwd(
            d_y, st["qf"], st["kf"], st["v"], y, st["lse"], cu_seqlens, max_seq_len,
            model.window_sizes[i], deterministic=deterministic_attention)
        # per-(token, head) norm backward with the 1.2 scale folded in
        d_qr = _rms_bwd_scaled(dqf, st["qf"], st["r_q"], 1.2)
        d_kr = _rms_bwd_scaled(dkf, st["kf"], st["r_k"], 1.2)
        # rotary backward = rotation by -theta (transpose of the forward rotation)
        dq1, dq2 = d_qr[..., :half], d_qr[..., half:]
        d_q0 = torch.cat([dq1 * cos - dq2 * sin, dq1 * sin + dq2 * cos], dim=-1)
        dk1, dk2 = d_kr[..., :half], d_kr[..., half:]
        d_k0 = torch.cat([dk1 * cos - dk2 * sin, dk1 * sin + dk2 * cos], dim=-1)
        # --- VE gate backward (ve/g recomputed) ---
        j = model.ve_index[i]
        d_xn_ve = None
        if j >= 0:
            ve = F.embedding(idx, model.value_embeds[j]).view(T, nkv, hd).to(dt)
            sg = torch.sigmoid(xn[..., :gch] @ model.ve_gate[j].to(dt).mT)
            d_g = (dv * ve).sum(dim=-1)          # (T, n_kv_head)
            d_zg = d_g * (3 * sg * (1 - sg))
            model.ve_gate.grad32_slices[j].add_(d_zg.mT @ xn[..., :gch])
            d_ve = (dv * (3 * sg).unsqueeze(-1)).reshape(T, nkv * hd)
            # embedding_dense_backward (autograd's own lowering) beats raw
            # index_add_ atomics ~2x at these shapes — see the GH200 trace hunt
            model.value_embeds.grad32_slices[j].add_(
                torch.ops.aten.embedding_dense_backward(d_ve.to(gemb), idx, Vp, -1, False))
            d_xn_ve = d_zg @ model.ve_gate[j].to(dt)
        # dv passes through the VE add unchanged: v = v0 + g*ve
        d_q0 = d_q0.view(T, nh * hd)
        d_k0 = d_k0.view(T, nkv * hd)
        d_v0 = dv.reshape(T, nkv * hd)
        model.c_q.grad32_slices[i].add_(d_q0.mT @ xn)
        model.c_k.grad32_slices[i].add_(d_k0.mT @ xn)
        model.c_v.grad32_slices[i].add_(d_v0.mT @ xn)
        d_xn = d_q0 @ model.c_q[i].to(dt) + d_k0 @ model.c_k[i].to(dt) + d_v0 @ model.c_v[i].to(dt)
        if d_xn_ve is not None:
            d_xn[:, :gch] += d_xn_ve
        d_b = d_x1 + _rms_bwd(d_xn, xn, st["r_xn"])
        # --- blend backward: b = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        g_resid.append((d_b * st["x_in"]).sum(dtype=g32))
        g_x0.append((d_b * x0).sum(dtype=g32))
        d_x0 = d_x0 + model.x0_lambdas[i] * d_b  # TRAP: x0 feeds every layer, accumulate
        d_stream = model.resid_lambdas[i] * d_b
        stash[i] = None                          # free this layer's stash as we go

    _land_scalar_grads(model, g_resid, g_x0)

    # d_stream is now the grad through layer 0's input, which IS x0 (same tensor)
    d_xs = d_x0 + d_stream                       # grad wrt the smeared embedding
    # --- smear backward: xs = cat([xe[:1], xe[1:] + gate*xe[:-1]]) ---
    sg = torch.sigmoid(xe[1:, :24] @ model.smear_gate.to(dt).mT)   # (T-1, 1), recomputed
    gate = model.smear_lambda.to(dt) * sg
    d_xe = d_xs.clone()
    d_xe[:-1] += gate * d_xs[1:]                 # TRAP: shifted scatter — p's grad reaches p-1
    d_gate = (d_xs[1:] * xe[:-1]).sum(dim=-1, keepdim=True)        # (T-1, 1)
    model.smear_lambda.grad32.add_((d_gate * sg).sum(dtype=g32))
    d_zs = d_gate * model.smear_lambda.to(dt) * sg * (1 - sg)
    model.smear_gate.grad32.add_((d_zs.mT @ xe[1:, :24]).to(g32))
    d_xe[1:, :24] += d_zs @ model.smear_gate.to(dt)
    # --- embedding norm + token embedding scatter ---
    d_emb = _rms_bwd(d_xe, xe, r_e)
    model.wte.grad32.add_(
        torch.ops.aten.embedding_dense_backward(d_emb.to(gemb), idx, Vp, -1, False))

    return loss


# -----------------------------------------------------------------------------
# forward_backward_fp8

@torch.no_grad()
def forward_backward_fp8(model, idx, targets, cu_seqlens, loss_scale=1.0,
                         deterministic_attention=False, loss_chunk=8192, w8=None):
    """forward_backward's FP8 twin — the 6th duplicated body, and deliberately
    line-parallel with the bf16 one above (keep them that way: diff them when
    either changes). The ONLY difference is that the six matrix banks' and
    lm_head's GEMMs — forward, dgrad and wgrad, 21 sites — run through
    torch._scaled_mm on tensorwise-scaled fp8 operands (nanochat/fp8.py holds the
    recipe and the NT-layout rules). Attention, norms, rotary, smear/VE gates,
    the CE, the embeddings and every scalar stay bf16/fp32, exactly as above.

    The stash stays bf16: every stashed tensor has a non-matmul consumer in the
    backward (rms backward, the FA kernel, relu^2), so keeping fp8 copies from
    the forward would ADD memory rather than save it — we re-cast in the backward
    and only the SCALE rides in the stash.

    What the handwritten form buys over the old autograd path (fp8.fp8_linear,
    one linear at a time, which is what `--fp8` used to select):
      - xn is quantized once for q/k/v instead of three times, and transposed
        once for their three weight gradients instead of three times;
      - the forward's per-tensor scales are stashed, so each backward re-cast
        skips a second amax pass over the activation;
      - weight scales are one batched reduction per bank, not one per layer.

    Wants torch.compile (base_train enforces it): the casts must fuse into their
    neighbours. In eager each one materializes an fp32 temp the size of its
    input, which at d12/131k tokens means a 17 GiB temp at the lm_head alone.
    """
    cfg = model.config
    assert idx.ndim == 1
    T = idx.size(0)
    nl = cfg.n_layer
    nh, nkv, hd = cfg.n_head, cfg.n_kv_head, model.head_dim
    D = cfg.n_embd
    half = hd // 2
    V = cfg.vocab_size
    Vp = model.padded_vocab_size
    max_seq_len = cfg.sequence_len
    dt = gpt_mod.COMPUTE_DTYPE
    g32 = model.c_q.grad32.dtype       # matrix/scalar grad dtype (fp32)
    gemb = model.wte.grad32.dtype      # embedding grad dtype (bf16 — see init_grad_buffers)
    gch = model.ve_gate_channels

    assert T > 1, "Training forward pass should have T > 1"
    assert T % 16 == 0, f"FP8 weight gradients contract over T, which must be % 16: got {T}"
    assert T <= model.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {model.cos.size(1)}"
    cos, sin = model.cos[0, :T], model.sin[0, :T]  # (T, 1, half)

    # Weights arrive pre-quantized in both layouts via `w8` (fp8.quantize_weights),
    # built ONCE per optimizer step by the trainer — the live weights are frozen
    # across a step's micro-batches, so per-micro-batch weight casts are pure
    # waste. The fallback keeps the body self-contained for single-call use.
    if w8 is None:
        w8 = fp8.quantize_weights(model)
    w_cq, w_ck, w_cv = w8["c_q"], w8["c_k"], w8["c_v"]
    w_ap, w_fc, w_mp, w_lm = w8["attn_proj"], w8["mlp_fc"], w8["mlp_proj"], w8["lm_head"]

    # ==== forward half (mirrors forward_backward — keep them visibly line-parallel) ====
    x = F.embedding(idx, model.wte)
    x = x.to(dt)
    xe, r_e = _rms_fwd(x, D)                     # post-norm embedding, pre-smear

    # Smear (the (.,24)x(24,1) gate matmul is far below the FP8 size floor)
    gate = model.smear_lambda.to(dt) * torch.sigmoid(xe[1:, :24] @ model.smear_gate.to(dt).mT)
    x = torch.cat([xe[:1], xe[1:] + gate * xe[:-1]], dim=0)

    x0 = x
    backout_layer = nl // 2
    x_backout = None
    stash = []
    for i in range(nl):
        x_in = x
        b = model.resid_lambdas[i] * x_in + model.x0_lambdas[i] * x0
        xn, r_xn = _rms_fwd(b, D)
        s_xn = fp8.tensor_scale(xn, E4M3)        # ONE amax + ONE cast for q/k/v
        xnq = fp8.row(xn, E4M3, s_xn)
        q = fp8.mm(xnq, fp8.F8(w_cq.row[i], w_cq.s[i].reciprocal()), dt, fast_accum=True).view(T, nh, hd)
        k = fp8.mm(xnq, fp8.F8(w_ck.row[i], w_ck.s[i].reciprocal()), dt, fast_accum=True).view(T, nkv, hd)
        v = fp8.mm(xnq, fp8.F8(w_cv.row[i], w_cv.s[i].reciprocal()), dt, fast_accum=True).view(T, nkv, hd)
        j = model.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, model.value_embeds[j]).view(T, nkv, hd).to(dt)
            g = 3 * torch.sigmoid(xn[..., :gch] @ model.ve_gate[j].to(dt).mT)   # (.,12)x(12,nkv): bf16
            v = v + g.unsqueeze(-1) * ve         # ve/g recomputed in backward, not stashed
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        qn, r_q = _rms_fwd(q, hd)
        kn, r_k = _rms_fwd(k, hd)
        qf = qn * 1.2                            # stash the SCALED q/k (the kernel's inputs);
        kf = kn * 1.2                            # backward folds the 1.2 via _rms_bwd_scaled
        y, lse = flash_attn_varlen_fwd_lse(qf, kf, v, cu_seqlens, max_seq_len, model.window_sizes[i])
        y = y.contiguous()
        yv = y.view(T, -1)
        s_y = fp8.tensor_scale(yv, E4M3)
        x1 = b + fp8.mm(fp8.row(yv, E4M3, s_y), fp8.F8(w_ap.row[i], w_ap.s[i].reciprocal()), dt, fast_accum=True)
        xm, _ = _rms_fwd(x1, D)                  # xm recomputed in backward from stashed x1
        s_xm = fp8.tensor_scale(xm, E4M3)
        a = F.relu(fp8.mm(fp8.row(xm, E4M3, s_xm), fp8.F8(w_fc.row[i], w_fc.s[i].reciprocal()), dt, fast_accum=True))
        a2 = a.square()                          # recomputed in backward; its scale rides in the stash
        s_a2 = fp8.tensor_scale(a2, E4M3)
        x = x1 + fp8.mm(fp8.row(a2, E4M3, s_a2), fp8.F8(w_mp.row[i], w_mp.s[i].reciprocal()), dt, fast_accum=True)
        if i == backout_layer:
            x_backout = x
        stash.append(dict(x_in=x_in, xn=xn, r_xn=r_xn, qf=qf, kf=kf, r_q=r_q, r_k=r_k,
                          v=v, y=y, lse=lse, x1=x1, a=a,
                          s_xn=s_xn, s_y=s_y, s_xm=s_xm, s_a2=s_a2))

    x_pre = x - model.backout_lambda.to(dt) * x_backout
    xf, r_f = _rms_fwd(x_pre, D)
    s_xf = fp8.tensor_scale(xf, E4M3)

    # lm_head + softcap + CE loss + dlogits. The CE itself never sees FP8 — only
    # the GEMM that produced the logits did — so this block stays character for
    # character the bf16 body's, mode-split and all:
    #  - eager: IN PLACE on the one (T, V) fp32 buffer, row-chunked so temps
    #    stay chunk-sized (no second full-size fp32 tensor);
    #  - compiled: a plain out-of-place version — under inductor the in-place
    #    chunk loop inverts into a liability (functionalization re-materializes
    #    the buffer; 16 unrolled subgraphs block fusion), so let the compiler
    #    do the memory planning and fusion it's good at.
    softcap = 15.0
    buf_dtype = torch.float64 if dt == torch.float64 else torch.float32
    valid = targets >= 0
    n_valid = valid.sum()
    if torch.compiler.is_compiling():
        # Written for inductor's fusion, not for readability of the in-place
        # eager twin below: t is an explicit CSE target (materialize once, no
        # tanh recompute in the dz pass), and the onehot is a broadcast compare
        # (a scatter_add here forces an extra full pass over the buffer).
        # (An in-graph row-chunked fwd+bwd was tried 2026-07-29 and reverted —
        # see the bf16 body's note.)
        inv_lm = w_lm.s.reciprocal()
        logits = fp8.mm(fp8.row(xf, E4M3, s_xf), fp8.F8(w_lm.row, inv_lm), dt, fast_accum=True)
        t = torch.tanh(logits[..., :V].to(buf_dtype) / softcap)
        cap = softcap * t
        vmask = valid.to(buf_dtype)
        y_safe = targets.clamp_min(0).unsqueeze(1)
        cap_y = cap.gather(1, y_safe).squeeze(1)
        m = cap.amax(dim=1, keepdim=True)
        e = (cap - m).exp()
        ssum = e.sum(dim=1, keepdim=True)
        lse = (ssum.log() + m).squeeze(1)
        loss = ((lse - cap_y) * vmask).sum() / n_valid.to(buf_dtype)
        onehot = torch.arange(V, device=targets.device).unsqueeze(0) == y_safe
        dz = ((e / ssum - onehot.to(buf_dtype)) * (1.0 - t * t)
              * (vmask * (loss_scale / n_valid.to(buf_dtype))).unsqueeze(1)).to(dt)
        del logits
        s_dz = fp8.tensor_scale(dz, E5M2)
        # The (T, V) grad is the biggest tensor in the step: cast one layout,
        # consume it, drop it, then the other — never both fp8 copies alive at once.
        g_lm = fp8.mm(fp8.col(dz, E5M2, s_dz), fp8.col(xf, E4M3, s_xf), dt).to(g32)  # padded rows get no grad
        (model.lm_head.grad32 if V == Vp else model.lm_head.grad32[:V]).add_(g_lm)
        dxf = fp8.mm(fp8.row(dz, E5M2, s_dz), fp8.F8(w_lm.colT[:, :V], inv_lm), dt)
        del dz
    else:
        logits = fp8.mm(fp8.row(xf, E4M3, s_xf), fp8.F8(w_lm.row, w_lm.s.reciprocal()), dt, fast_accum=True)
        buf = logits[..., :V].to(buf_dtype)      # THE big fp32 buffer; bf16 logits freed below
        buf.div_(softcap).tanh_().mul_(softcap)  # buf = cap = 15*tanh(z/15)
        # Softcap keeps cap in [-15, 15], so softmax probs are bounded below by
        # ~exp(-30)/V — no underflow anywhere in the in-place chain that follows.
        loss_sum = torch.zeros((), dtype=buf_dtype, device=buf.device)
        scale = loss_scale / n_valid.to(buf_dtype)   # 0-dim device tensor; no host sync
        for r0 in range(0, T, loss_chunk):
            c = buf[r0:r0 + loss_chunk]          # in-place view; temps are chunk-sized
            vc = valid[r0:r0 + loss_chunk].to(buf_dtype)
            y_safe = targets[r0:r0 + loss_chunk].clamp_min(0).unsqueeze(1)
            cap_y = c.gather(1, y_safe).squeeze(1)
            m = c.amax(dim=1, keepdim=True)
            c.sub_(m).exp_()                     # c = exp(cap - m)
            ssum = c.sum(dim=1, keepdim=True)
            loss_sum += ((ssum.log() + m).squeeze(1) - cap_y).mul_(vc).sum()
            f = c.log().add_(m)                  # f = cap, recovered exactly
            f.div_(softcap).square_().neg_().add_(1.0)   # f = 1 - tanh^2(z/15)
            c.div_(ssum)                         # c = softmax(cap)
            c.scatter_add_(1, y_safe, (-vc).unsqueeze(1))  # - onehot on valid rows
            c.mul_(f).mul_(vc.unsqueeze(1) * scale)  # softcap chain, ignore-mask, 1/n_valid, loss_scale
        loss = loss_sum / n_valid.to(buf_dtype)
        del logits
        dz = buf.to(dt)                          # mirror autograd's cast back through .float()
        del buf
        s_dz = fp8.tensor_scale(dz, E5M2)
        # The (T, V) grad is the biggest tensor in the step: cast one layout,
        # consume it, drop it, then the other — never both fp8 copies alive at once.
        g_lm = fp8.mm(fp8.col(dz, E5M2, s_dz), fp8.col(xf, E4M3, s_xf), dt).to(g32)  # padded rows get no grad
        (model.lm_head.grad32 if V == Vp else model.lm_head.grad32[:V]).add_(g_lm)
        dxf = fp8.mm(fp8.row(dz, E5M2, s_dz), fp8.F8(w_lm.colT[:, :V], w_lm.s.reciprocal()), dt)
        del dz

    # ==== backward half ====
    # Bank wgrads add directly into grad32_slices views; only the per-layer
    # scalar sums are collected and landed stacked (_land_scalar_grads).
    g_resid = []; g_x0 = []

    d_pre = _rms_bwd(dxf, xf, r_f)
    model.backout_lambda.grad32.add_(-(d_pre * x_backout).sum(dtype=g32))
    d_stream = d_pre                             # grad wrt layer nl-1's output
    d_x0 = torch.zeros_like(x0)
    for i in reversed(range(nl)):
        st = stash[i]
        if i == backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes nl//2
            d_stream = d_stream - model.backout_lambda.to(dt) * d_pre
        # --- MLP backward (relu^2: dh = 2*a*du, self-masking since a = relu(h)) ---
        x1, a = st["x1"], st["a"]
        s_ds = fp8.tensor_scale(d_stream, E5M2)
        d_u = fp8.mm(fp8.row(d_stream, E5M2, s_ds), fp8.F8(w_mp.colT[i], w_mp.s[i].reciprocal()), dt)
        model.mlp_proj.grad32_slices[i].add_(fp8.mm(fp8.col(d_stream, E5M2, s_ds), fp8.col(a.square(), E4M3, st["s_a2"]), dt))
        d_h = 2.0 * a * d_u
        xm, r_xm = _rms_fwd(x1, D)               # cheap recompute (bitwise: same input)
        s_dh = fp8.tensor_scale(d_h, E5M2)
        model.mlp_fc.grad32_slices[i].add_(fp8.mm(fp8.col(d_h, E5M2, s_dh), fp8.col(xm, E4M3, st["s_xm"]), dt))
        d_xm = fp8.mm(fp8.row(d_h, E5M2, s_dh), fp8.F8(w_fc.colT[i], w_fc.s[i].reciprocal()), dt)
        d_x1 = d_stream + _rms_bwd(d_xm, xm, r_xm)
        # --- attention backward ---
        xn, y = st["xn"], st["y"]
        s_dx1 = fp8.tensor_scale(d_x1, E5M2)
        model.attn_proj.grad32_slices[i].add_(fp8.mm(fp8.col(d_x1, E5M2, s_dx1), fp8.col(y.view(T, -1), E4M3, st["s_y"]), dt))
        d_y = fp8.mm(fp8.row(d_x1, E5M2, s_dx1), fp8.F8(w_ap.colT[i], w_ap.s[i].reciprocal()), dt).view(T, nh, hd)
        dqf, dkf, dv = flash_attn_varlen_bwd(
            d_y, st["qf"], st["kf"], st["v"], y, st["lse"], cu_seqlens, max_seq_len,
            model.window_sizes[i], deterministic=deterministic_attention)
        # per-(token, head) norm backward with the 1.2 scale folded in
        d_qr = _rms_bwd_scaled(dqf, st["qf"], st["r_q"], 1.2)
        d_kr = _rms_bwd_scaled(dkf, st["kf"], st["r_k"], 1.2)
        # rotary backward = rotation by -theta (transpose of the forward rotation)
        dq1, dq2 = d_qr[..., :half], d_qr[..., half:]
        d_q0 = torch.cat([dq1 * cos - dq2 * sin, dq1 * sin + dq2 * cos], dim=-1)
        dk1, dk2 = d_kr[..., :half], d_kr[..., half:]
        d_k0 = torch.cat([dk1 * cos - dk2 * sin, dk1 * sin + dk2 * cos], dim=-1)
        # --- VE gate backward (ve/g recomputed; all of it below the FP8 floor) ---
        j = model.ve_index[i]
        d_xn_ve = None
        if j >= 0:
            ve = F.embedding(idx, model.value_embeds[j]).view(T, nkv, hd).to(dt)
            sg = torch.sigmoid(xn[..., :gch] @ model.ve_gate[j].to(dt).mT)
            d_g = (dv * ve).sum(dim=-1)          # (T, n_kv_head)
            d_zg = d_g * (3 * sg * (1 - sg))
            model.ve_gate.grad32_slices[j].add_(d_zg.mT @ xn[..., :gch])
            d_ve = (dv * (3 * sg).unsqueeze(-1)).reshape(T, nkv * hd)
            # embedding_dense_backward (autograd's own lowering) beats raw
            # index_add_ atomics ~2x at these shapes — see the GH200 trace hunt
            model.value_embeds.grad32_slices[j].add_(
                torch.ops.aten.embedding_dense_backward(d_ve.to(gemb), idx, Vp, -1, False))
            d_xn_ve = d_zg @ model.ve_gate[j].to(dt)
        # dv passes through the VE add unchanged: v = v0 + g*ve
        d_q0 = d_q0.view(T, nh * hd)
        d_k0 = d_k0.view(T, nkv * hd)
        d_v0 = dv.reshape(T, nkv * hd)
        xnt = fp8.col(xn, E4M3, st["s_xn"])      # ONE transpose for the three weight grads
        s_dq = fp8.tensor_scale(d_q0, E5M2)
        s_dk = fp8.tensor_scale(d_k0, E5M2)
        s_dv = fp8.tensor_scale(d_v0, E5M2)
        model.c_q.grad32_slices[i].add_(fp8.mm(fp8.col(d_q0, E5M2, s_dq), xnt, dt))
        model.c_k.grad32_slices[i].add_(fp8.mm(fp8.col(d_k0, E5M2, s_dk), xnt, dt))
        model.c_v.grad32_slices[i].add_(fp8.mm(fp8.col(d_v0, E5M2, s_dv), xnt, dt))
        d_xn = fp8.mm(fp8.row(d_q0, E5M2, s_dq), fp8.F8(w_cq.colT[i], w_cq.s[i].reciprocal()), dt) \
             + fp8.mm(fp8.row(d_k0, E5M2, s_dk), fp8.F8(w_ck.colT[i], w_ck.s[i].reciprocal()), dt) \
             + fp8.mm(fp8.row(d_v0, E5M2, s_dv), fp8.F8(w_cv.colT[i], w_cv.s[i].reciprocal()), dt)
        if d_xn_ve is not None:
            d_xn[:, :gch] += d_xn_ve
        d_b = d_x1 + _rms_bwd(d_xn, xn, st["r_xn"])
        # --- blend backward: b = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        g_resid.append((d_b * st["x_in"]).sum(dtype=g32))
        g_x0.append((d_b * x0).sum(dtype=g32))
        d_x0 = d_x0 + model.x0_lambdas[i] * d_b  # TRAP: x0 feeds every layer, accumulate
        d_stream = model.resid_lambdas[i] * d_b
        stash[i] = None                          # free this layer's stash as we go

    _land_scalar_grads(model, g_resid, g_x0)

    # d_stream is now the grad through layer 0's input, which IS x0 (same tensor)
    d_xs = d_x0 + d_stream                       # grad wrt the smeared embedding
    # --- smear backward: xs = cat([xe[:1], xe[1:] + gate*xe[:-1]]) ---
    sg = torch.sigmoid(xe[1:, :24] @ model.smear_gate.to(dt).mT)   # (T-1, 1), recomputed
    gate = model.smear_lambda.to(dt) * sg
    d_xe = d_xs.clone()
    d_xe[:-1] += gate * d_xs[1:]                 # TRAP: shifted scatter — p's grad reaches p-1
    d_gate = (d_xs[1:] * xe[:-1]).sum(dim=-1, keepdim=True)        # (T-1, 1)
    model.smear_lambda.grad32.add_((d_gate * sg).sum(dtype=g32))
    d_zs = d_gate * model.smear_lambda.to(dt) * sg * (1 - sg)
    model.smear_gate.grad32.add_((d_zs.mT @ xe[1:, :24]).to(g32))
    d_xe[1:, :24] += d_zs @ model.smear_gate.to(dt)
    # --- embedding norm + token embedding scatter ---
    d_emb = _rms_bwd(d_xe, xe, r_e)
    model.wte.grad32.add_(
        torch.ops.aten.embedding_dense_backward(d_emb.to(gemb), idx, Vp, -1, False))

    return loss


# =============================================================================
# Explicit optimizer flow (no torch.optim, no param groups)
#
# State attaches to each Parameter: .grad32 (above), plus per-verb state
# (.mantissa always; .momentum/.second_momentum for Muon; .exp_avg/.exp_avg_sq
# for AdamW) — sharded params carry shard-size state. Policy (schedule tables,
# per-bank multipliers, ns_steps) appears at call sites in the written-out
# optimizer_step, not as attributes.
#
# Masters use the mantissa trick (Larry Dial via modded-nanogpt train_gpt.py):
# the fp32 master's bit pattern is (live_bf16_bits << 16) | mantissa_uint16.
# Update math runs in fp32 on the reconstructed master; the split back is a
# TRUNCATION (load-bearing: round-to-nearest could carry into the top bits and
# break the lossless live/mantissa pairing).
# =============================================================================

# The bit arithmetic runs in int32 (CUDA has no uint32 shifts as of torch 2.9);
# int32's truncating .to(int16) and the <<16 discard of sign-extension bits make
# it equivalent. Mantissa tensors are STORED uint16, viewed int16 for the math.

def _master(live: Tensor, mantissa: Tensor) -> Tensor:
    """Reconstruct the fp32 master from bf16 live bits + stashed mantissa."""
    bits = (live.view(torch.int16).to(torch.int32) << 16) | \
           (mantissa.view(torch.int16).to(torch.int32) & 0xFFFF)
    return bits.view(torch.float32)


def _writeback(master: Tensor, live: Tensor, mantissa: Tensor) -> None:
    """Truncation split of the updated master back into live + mantissa."""
    bits = master.view(torch.int32)
    live.view(torch.int16).copy_((bits >> 16).to(torch.int16))
    mantissa.view(torch.int16).copy_(bits.to(torch.int16))


class AdamWTabs(NamedTuple):
    """schedules.AdamWCoeffs minus the eps field — eps is never scheduled, so it
    rides as a plain kernel argument instead of an (N,) table."""
    wd_mul: Tensor           # 1 - lr*wd             decoupled weight decay
    one_minus_beta1: Tensor  # 1 - beta1             exp_avg lerp weight
    one_minus_beta2: Tensor  # 1 - beta2             exp_avg_sq lerp weight
    rsqrt_bias2: Tensor      # 1/sqrt(bias2)         second-moment bias correction
    step_size: Tensor        # lr / bias1            lr schedule x first-moment bias correction


# -----------------------------------------------------------------------------
# Fused update kernels (ported from nanochat/optim.py, two changes each:
# mantissa reconstruct/writeback replaces the fp32 param read/write, and Muon
# takes per-slice (K,1,1) lr/wd multipliers so bank merging later needs no
# kernel change).

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_fp32(
    p: Tensor,           # fp32 param, updated IN PLACE (live == master)
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    c: AdamWTabs,
    t: Tensor,
    eps: float,
) -> None:
    """AdamW for the fp32-LIVE scalar params (resid/x0 lambdas, smear, backout
    — ~30 floats, replicated). They are exempt from the bf16-live/mantissa
    scheme: the baseline never cast them in forward, and bf16-rounding those
    per-layer residual-stream multipliers cost +0.016 val bpb early in training
    (diagnosed 2026-07-29, agent-ops diag_fp32_masters logs)."""
    grad = grad.to(exp_avg.dtype)
    p.mul_(c.wd_mul[t])
    exp_avg.lerp_(grad, c.one_minus_beta1[t])
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[t])
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
    p.sub_(c.step_size[t] * (exp_avg / denom))


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    live: Tensor,        # bf16 live shard
    mantissa: Tensor,    # uint16, same shape
    grad: Tensor,        # fp32 gradient shard
    exp_avg: Tensor,     # fp32 first moment
    exp_avg_sq: Tensor,  # fp32 second moment
    c: AdamWTabs,        # per-step coefficient tables, device-resident
    t: Tensor,           # (1,) int64 device tensor - the schedule row to read
    eps: float,
) -> None:
    """Fused AdamW step on the reconstructed master. Same folded-coefficient
    scheme as optim.adamw_step_fused (see there for why the gather-by-t makes
    the step host-free); moments are always fp32 here, so the lerp weights need
    no dtype casts."""
    p = _master(live, mantissa)
    grad = grad.to(exp_avg.dtype)  # embeddings hand in bf16 grads; moment math stays fp32
    p.mul_(c.wd_mul[t])
    exp_avg.lerp_(grad, c.one_minus_beta1[t])
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[t])
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
    p.sub_(c.step_size[t] * (exp_avg / denom))
    _writeback(p, live, mantissa)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    grad: Tensor,                   # (K, out, in) fp32 gradient shard — MUTATED (nesterov lerp)
    live: Tensor,                   # (K, out, in) bf16 live shard
    mantissa: Tensor,               # (K, out, in) uint16
    momentum_buffer: Tensor,        # (K, out, in) fp32
    second_momentum_buffer: Tensor, # (K, out, 1) or (K, 1, in) fp32 - factored second moment
    c: MuonCoeffs,                  # per-step coefficient tables, device-resident (UNfolded lr)
    t: Tensor,                      # (1,) int64 device tensor - the schedule row to read
    ns_steps: int,                  # 5 - number of Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
    lr_mul: Tensor,                 # (K, 1, 1) fp32 per-slice LR multiplier (aspect scale today)
    wd_mul: Tensor,                 # (K, 1, 1) fp32 per-slice WD multiplier
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction ->
    cautious update on the reconstructed master. The sqrt(fan_out/fan_in)
    aspect scale is NOT in `c` — it arrives through lr_mul/wd_mul, per slice."""
    dtype = grad.dtype

    # Nesterov momentum
    momentum_buffer.lerp_(grad, c.one_minus_momentum[t].to(dtype))
    g = grad.lerp_(momentum_buffer, c.momentum[t].to(dtype))

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c_ns in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c_ns * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c_ns in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c_ns * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction (NorMuon). The lerp weight stays fp32 — see the long
    # dtype note in optim.muon_step_fused.
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype),
                                 c.one_minus_beta2[t].to(second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + master update + truncation split back to live
    p = _master(live, mantissa)
    mask = (g * p) >= 0
    lr = (c.lr[t] * lr_mul).to(g.dtype)
    lr_wd = (c.lr_wd[t] * wd_mul).to(g.dtype)
    p.sub_(lr * g + lr_wd * p * mask)
    _writeback(p, live, mantissa)


# -----------------------------------------------------------------------------
# Schedules: pure policy bundles -> one namespace of named, device-resident
# table sets. Folding math is schedules.py's, verbatim; the packaging differs
# (no params in the specs, device passed explicitly, and the Muon aspect scale
# deliberately NOT baked into the shared matrix table).

def _adamw_tabs(lr, betas, weight_decay, num_steps, device) -> AdamWTabs:
    N = num_steps
    lr = _as_table(lr, N, "lr")
    beta1 = _as_table(betas[0], N, "beta1")
    beta2 = _as_table(betas[1], N, "beta2")
    wd = _as_table(weight_decay, N, "weight_decay")
    return _tables(
        AdamWTabs, device,
        wd_mul=1.0 - lr * wd,
        one_minus_beta1=1.0 - beta1,
        one_minus_beta2=1.0 - beta2,
        rsqrt_bias2=1.0 / (_bias_correction(beta2) ** 0.5),
        step_size=lr / _bias_correction(beta1),
    )


def _muon_tabs(lr, momentum, beta2, weight_decay, num_steps, device) -> MuonCoeffs:
    N = num_steps
    lr = _as_table(lr, N, "lr")   # canonical: NO per-shape fold (see bank_muls)
    momentum = _as_table(momentum, N, "momentum")
    beta2 = _as_table(beta2, N, "beta2")
    wd = _as_table(weight_decay, N, "weight_decay")
    return _tables(
        MuonCoeffs, device,
        momentum=momentum,
        one_minus_momentum=1.0 - momentum,
        one_minus_beta2=1.0 - beta2,
        lr=lr,
        lr_wd=lr * wd,
    )


def build_schedules(model_dim, num_iterations, device,
                    embedding_lr=0.3, unembedding_lr=0.008, matrix_lr=0.02,
                    scalar_lr=0.5, weight_decay=0.28, batch_lr_scale=1.0,
                    warmup_steps=40, warmdown_ratio=0.65, final_lr_frac=0.05):
    """Named table sets with EXACTLY the hyperparameters of the old base_train
    spec block. `weight_decay` arrives already batch/horizon-scaled. Returns a
    namespace: .matrix (shared MuonCoeffs) + one AdamWTabs per AdamW role,
    .adamw_eps, and .num_steps."""
    dmodel_lr_scale = (model_dim / 768) ** -0.5     # AdamW LRs tuned at d12's 768
    adamw_lr_scale = batch_lr_scale * dmodel_lr_scale

    # One LR shape for the whole run (linear warmup, constant, linear warmdown),
    # scaled to each role's own peak. Muon momentum warms to 0.97 then cools to
    # 0.90 during the LR warmdown; its weight decay cosine-decays to zero.
    lrm = Ramp(peak=1.0, start=0.0, warmup_steps=warmup_steps,
               end=final_lr_frac, cooldown_frac=warmdown_ratio)
    # momentum warmup is 400 steps at any real horizon; the clamp only lets
    # short smoke/debug runs build a valid schedule (identical for N >= ~1150)
    mom_warmup = min(400, int(num_iterations * (1 - warmdown_ratio)))
    muon_momentum = Ramp(peak=0.97, start=0.85, warmup_steps=mom_warmup,
                         end=0.90, cooldown_frac=warmdown_ratio)
    muon_wd = Ramp(peak=weight_decay, end=0.0, cooldown_frac=1.0, shape="cosine")

    N, dev = num_iterations, device
    return SimpleNamespace(
        matrix       = _muon_tabs(lrm * (matrix_lr * batch_lr_scale), muon_momentum, 0.9, muon_wd, N, dev),
        lm_head      = _adamw_tabs(lrm * (unembedding_lr * adamw_lr_scale),     (0.8, 0.96),  0.01,  N, dev),
        wte          = _adamw_tabs(lrm * (embedding_lr * adamw_lr_scale),       (0.8, 0.995), 0.001, N, dev),
        value_embeds = _adamw_tabs(lrm * (embedding_lr * adamw_lr_scale * 0.5), (0.8, 0.995), 0.01,  N, dev),
        resid        = _adamw_tabs(lrm * (scalar_lr * batch_lr_scale * 0.01),   (0.8, 0.95),  0.05,  N, dev),
        x0           = _adamw_tabs(lrm * (scalar_lr * batch_lr_scale),          (0.96, 0.95), 0.0,   N, dev),
        smear        = _adamw_tabs(lrm * 0.2,                                   (0.8, 0.95),  0.0,   N, dev),
        adamw_eps    = 1e-10,
        lrm_table    = lrm.materialize(num_iterations),  # host-side copy, for logging only
        num_steps    = N,
    )


def norm_muon_reduce_dim(out_features, in_features):
    """NorMuon's variance-reduction axis for an (out_features, in_features)
    slice: -1 reduces over in_features, -2 over out_features.

    The policy is to keep the SMALLER axis, so the factored second moment costs
    min(out, in) floats per slice instead of max(out, in). That makes it genuine
    shape math -- but it is called ONCE per bank, in init_optimizer_state, with
    named dimensions, and the answer is stored on the Parameter as `.red_dim`.

    Everything downstream reads that stored value. The old code re-derived it
    from `p.shape` at two call sites that had to agree, because the second-moment
    buffer's shape is decided by this choice: disagree and the kernel's lerp_
    writes a (K, out, 1) update into a (K, 1, in) buffer. They agreed by luck
    until d24, where ve_gate's (n_kv_head, gate_ch) goes wide at d12/d20, square
    at d24 and tall at d26.
    """
    return -1 if out_features >= in_features else -2


def bank_muls(model, ns_steps=5):
    """Per-bank Muon call-site policy: the (K,1,1) per-slice lr/wd multipliers
    and the NorMuon reduction axis.

    The multiplier is the bank's sqrt(max(1, fan_out/fan_in)) aspect scale --
    Muon's tall-matrix LR correction -- deliberately kept OUT of the shared
    matrix coefficient table so that table stays one set of numbers valid for
    every bank. Today every slice of a bank is uniform, so each multiplier is a
    constant fill; merged banks later change these lines, not the kernels.

    red_dim is READ off the Parameter here rather than recomputed -- see
    norm_muon_reduce_dim for why that matters.
    """
    d = model.dims
    dev = model.c_q.device

    aspect_c_q       = max(1.0, d.q       / d.d_model) ** 0.5   # 1.0 (square)
    aspect_c_k       = max(1.0, d.kv      / d.d_model) ** 0.5   # 1.0 (square, or wide under GQA)
    aspect_c_v       = max(1.0, d.kv      / d.d_model) ** 0.5   # 1.0
    aspect_attn_proj = max(1.0, d.d_model / d.q)       ** 0.5   # 1.0
    aspect_mlp_fc    = max(1.0, d.mlp     / d.d_model) ** 0.5   # 2.0 (4x expansion)
    aspect_mlp_proj  = max(1.0, d.d_model / d.mlp)     ** 0.5   # 1.0 (wide)
    aspect_ve_gate   = max(1.0, d.n_kv_head / d.gate_ch) ** 0.5 # 1.0 at d12

    return {
        "c_q": dict(ns_steps=ns_steps, red_dim=model.c_q.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_c_q, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_c_q, dtype=torch.float32, device=dev)),
        "c_k": dict(ns_steps=ns_steps, red_dim=model.c_k.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_c_k, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_c_k, dtype=torch.float32, device=dev)),
        "c_v": dict(ns_steps=ns_steps, red_dim=model.c_v.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_c_v, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_c_v, dtype=torch.float32, device=dev)),
        "attn_proj": dict(ns_steps=ns_steps, red_dim=model.attn_proj.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_attn_proj, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_attn_proj, dtype=torch.float32, device=dev)),
        "mlp_fc": dict(ns_steps=ns_steps, red_dim=model.mlp_fc.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_mlp_fc, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_mlp_fc, dtype=torch.float32, device=dev)),
        "mlp_proj": dict(ns_steps=ns_steps, red_dim=model.mlp_proj.red_dim,
                    lr_mul=torch.full((d.layer, 1, 1), aspect_mlp_proj, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.layer, 1, 1), aspect_mlp_proj, dtype=torch.float32, device=dev)),
        "ve_gate": dict(ns_steps=ns_steps, red_dim=model.ve_gate.red_dim,
                    lr_mul=torch.full((d.ve_slot, 1, 1), aspect_ve_gate, dtype=torch.float32, device=dev),
                    wd_mul=torch.full((d.ve_slot, 1, 1), aspect_ve_gate, dtype=torch.float32, device=dev)),
    }


# -----------------------------------------------------------------------------
# Optimizer state: attach to Parameters, masters split in place.

def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _split_master(p):
    """Split p into bf16 live (swapped into p.data in place — Parameter identity
    preserved) + this rank's uint16 mantissa shard. Params initialized bf16
    (the embeddings) upcast to a master with zero mantissa — already lossless."""
    master = p.detach().float()
    bits = master.view(torch.int32)
    p.data = (bits >> 16).to(torch.int16).view(torch.bfloat16)
    return bits.to(torch.int16).view(torch.uint16)


def _shard_slice(axis_len, world, rank):
    """This rank's chunk of a sharded axis (zero-padded: ceil(axis_len/world)).
    Takes the axis LENGTH, not the tensor -- the caller names which axis it is
    sharding, rather than this function assuming dim 0."""
    chunk = -(-axis_len // world)
    start = rank * chunk
    return slice(start, min(axis_len, start + chunk))


@torch.no_grad()
def init_optimizer_state(model, ddp_rank=0, ddp_world_size=1):
    """Allocate every optimizer-side tensor, written out one parameter at a time.

    Each block states the parameter's axes by name, which axis (if any) is
    sharded, and the shape and dtype of each piece of state it carries. That is
    the point of the repetition: the full memory cost of training this model,
    and the reason behind each dtype, is readable top to bottom here rather than
    implied by a loop over an abstract parameter list.

    Runs after GPT.__init__ has allocated the weights on-device and they have
    been filled (init_weights or load_model_state). It CONVERTS the fp32 matmul
    weights in place into the bf16-live + uint16-mantissa master pair, so it must
    run exactly once and after the weights hold their final values.

    Sharding, at world > 1:
      - Muon banks shard over their bank axis (`layer`, or `ve_slot` for the VE
        gates), zero-padded by ceil division so a ragged tail is legal.
      - AdamW params shard over `vocab` rows, which must divide evenly.
      - `grad32` always stays FULL size on every rank -- it is the source buffer
        for the reduce-scatter, not a shard.
      - At world == 1 every shard IS the whole tensor and the state keeps the
        parameter's natural shape, which is what the fused kernels expect.
    """
    world, rank = ddp_world_size, ddp_rank
    d = model.dims
    dev = model.wte.device
    init_grad_buffers(model)

    # =========================================================================
    # Muon, sharded over the `layer` axis.
    # =========================================================================
    layer_sl = _shard_slice(d.layer, world, rank)
    own_layer = layer_sl.stop - layer_sl.start

    # --- c_q : query projection --- live (layer, q, d_model) bf16 --------------
    mant = _split_master(model.c_q)
    model.c_q.shard = layer_sl
    model.c_q.red_dim = norm_muon_reduce_dim(d.q, d.d_model)
    model.c_q.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.c_q.momentum = torch.zeros(own_layer, d.q, d.d_model, dtype=torch.float32, device=dev)
    model.c_q.second_momentum = torch.zeros(
        (own_layer, d.q, 1) if model.c_q.red_dim == -1 else (own_layer, 1, d.d_model),
        dtype=torch.float32, device=dev)

    # --- c_k : key projection --- live (layer, kv, d_model) bf16 ---------------
    mant = _split_master(model.c_k)
    model.c_k.shard = layer_sl
    model.c_k.red_dim = norm_muon_reduce_dim(d.kv, d.d_model)
    model.c_k.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.c_k.momentum = torch.zeros(own_layer, d.kv, d.d_model, dtype=torch.float32, device=dev)
    model.c_k.second_momentum = torch.zeros(
        (own_layer, d.kv, 1) if model.c_k.red_dim == -1 else (own_layer, 1, d.d_model),
        dtype=torch.float32, device=dev)

    # --- c_v : value projection --- live (layer, kv, d_model) bf16 -------------
    mant = _split_master(model.c_v)
    model.c_v.shard = layer_sl
    model.c_v.red_dim = norm_muon_reduce_dim(d.kv, d.d_model)
    model.c_v.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.c_v.momentum = torch.zeros(own_layer, d.kv, d.d_model, dtype=torch.float32, device=dev)
    model.c_v.second_momentum = torch.zeros(
        (own_layer, d.kv, 1) if model.c_v.red_dim == -1 else (own_layer, 1, d.d_model),
        dtype=torch.float32, device=dev)

    # --- attn_proj : attention output --- live (layer, d_model, q) bf16 --------
    mant = _split_master(model.attn_proj)
    model.attn_proj.shard = layer_sl
    model.attn_proj.red_dim = norm_muon_reduce_dim(d.d_model, d.q)
    model.attn_proj.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.attn_proj.momentum = torch.zeros(own_layer, d.d_model, d.q, dtype=torch.float32, device=dev)
    model.attn_proj.second_momentum = torch.zeros(
        (own_layer, d.d_model, 1) if model.attn_proj.red_dim == -1 else (own_layer, 1, d.q),
        dtype=torch.float32, device=dev)

    # --- mlp_fc : MLP up-projection --- live (layer, mlp, d_model) bf16 --------
    mant = _split_master(model.mlp_fc)
    model.mlp_fc.shard = layer_sl
    model.mlp_fc.red_dim = norm_muon_reduce_dim(d.mlp, d.d_model)
    model.mlp_fc.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.mlp_fc.momentum = torch.zeros(own_layer, d.mlp, d.d_model, dtype=torch.float32, device=dev)
    model.mlp_fc.second_momentum = torch.zeros(
        (own_layer, d.mlp, 1) if model.mlp_fc.red_dim == -1 else (own_layer, 1, d.d_model),
        dtype=torch.float32, device=dev)

    # --- mlp_proj : MLP down-projection --- live (layer, d_model, mlp) bf16 ----
    mant = _split_master(model.mlp_proj)
    model.mlp_proj.shard = layer_sl
    model.mlp_proj.red_dim = norm_muon_reduce_dim(d.d_model, d.mlp)
    model.mlp_proj.mantissa = mant[layer_sl].clone() if world > 1 else mant
    model.mlp_proj.momentum = torch.zeros(own_layer, d.d_model, d.mlp, dtype=torch.float32, device=dev)
    model.mlp_proj.second_momentum = torch.zeros(
        (own_layer, d.d_model, 1) if model.mlp_proj.red_dim == -1 else (own_layer, 1, d.mlp),
        dtype=torch.float32, device=dev)

    # =========================================================================
    # Muon, REPLICATED. ve_gate is tiny and ragged (ve_slot is half of layer),
    # so every rank all_reduces the gradient and runs the full-size update
    # rather than paying comm to shard a few thousand floats.
    # =========================================================================

    # --- ve_gate : value-embedding gates --- live (ve_slot, n_kv_head, gate_ch) bf16
    model.ve_gate.red_dim = norm_muon_reduce_dim(d.n_kv_head, d.gate_ch)
    model.ve_gate.mantissa = _split_master(model.ve_gate)
    model.ve_gate.momentum = torch.zeros(d.ve_slot, d.n_kv_head, d.gate_ch, dtype=torch.float32, device=dev)
    model.ve_gate.second_momentum = torch.zeros(
        (d.ve_slot, d.n_kv_head, 1) if model.ve_gate.red_dim == -1 else (d.ve_slot, 1, d.gate_ch),
        dtype=torch.float32, device=dev)

    # =========================================================================
    # AdamW, sharded over `vocab` rows (must divide evenly -- no zero padding
    # here, unlike the Muon bank axis).
    # =========================================================================
    assert d.vocab % world == 0, \
        f"AdamW row-sharding needs vocab % world == 0 ({d.vocab} % {world})"
    own_vocab = d.vocab // world
    vocab_sl = slice(rank * own_vocab, (rank + 1) * own_vocab)

    # --- lm_head : unembedding --- live (vocab, d_model) bf16 ------------------
    mant = _split_master(model.lm_head)
    model.lm_head.mantissa = mant[vocab_sl].clone() if world > 1 else mant
    model.lm_head.exp_avg = torch.zeros(
        (d.vocab, d.d_model) if world == 1 else (own_vocab, d.d_model), dtype=torch.float32, device=dev)
    model.lm_head.exp_avg_sq = torch.zeros(
        (d.vocab, d.d_model) if world == 1 else (own_vocab, d.d_model), dtype=torch.float32, device=dev)

    # --- wte : token embedding --- live (vocab, d_model) bf16 ------------------
    # Already COMPUTE_DTYPE from allocation, so the master upcast is lossless and
    # the mantissa starts at zero. The split still runs so the kernel sees the
    # same live/mantissa pair as every other AdamW parameter.
    mant = _split_master(model.wte)
    model.wte.mantissa = mant[vocab_sl].clone() if world > 1 else mant
    model.wte.exp_avg = torch.zeros(
        (d.vocab, d.d_model) if world == 1 else (own_vocab, d.d_model), dtype=torch.float32, device=dev)
    model.wte.exp_avg_sq = torch.zeros(
        (d.vocab, d.d_model) if world == 1 else (own_vocab, d.d_model), dtype=torch.float32, device=dev)

    # --- value_embeds : VE table --- live (ve_slot, vocab, kv) bf16 ------------
    # Sharded over the FLATTENED (ve_slot * vocab) row axis, not over ve_slot:
    # ve_slot alone is too small to divide across a world, and the rows are
    # interchangeable for AdamW's elementwise update.
    ve_rows = d.ve_slot * d.vocab
    assert ve_rows % world == 0, \
        f"AdamW row-sharding needs ve_slot*vocab % world == 0 ({ve_rows} % {world})"
    own_ve_rows = ve_rows // world
    ve_row_sl = slice(rank * own_ve_rows, (rank + 1) * own_ve_rows)
    mant = _split_master(model.value_embeds)
    model.value_embeds.mantissa = mant.view(ve_rows, d.kv)[ve_row_sl].clone() if world > 1 else mant
    model.value_embeds.exp_avg = torch.zeros(
        (d.ve_slot, d.vocab, d.kv) if world == 1 else (own_ve_rows, d.kv), dtype=torch.float32, device=dev)
    model.value_embeds.exp_avg_sq = torch.zeros(
        (d.ve_slot, d.vocab, d.kv) if world == 1 else (own_ve_rows, d.kv), dtype=torch.float32, device=dev)

    # =========================================================================
    # AdamW, REPLICATED -- the ~30 scalars. These are fp32-LIVE with NO mantissa
    # pair: bf16-rounding the per-layer residual-stream multipliers cost +0.016
    # val bpb early in training (diagnosed 2026-07-29, agent-ops
    # diag_fp32_masters logs), so they are exempt from the master scheme and run
    # through adamw_step_fused_fp32 instead.
    # =========================================================================
    model.resid_lambdas.exp_avg     = torch.zeros(d.layer, dtype=torch.float32, device=dev)
    model.resid_lambdas.exp_avg_sq  = torch.zeros(d.layer, dtype=torch.float32, device=dev)
    model.x0_lambdas.exp_avg        = torch.zeros(d.layer, dtype=torch.float32, device=dev)
    model.x0_lambdas.exp_avg_sq     = torch.zeros(d.layer, dtype=torch.float32, device=dev)
    model.smear_gate.exp_avg        = torch.zeros(1, d.smear_ch, dtype=torch.float32, device=dev)
    model.smear_gate.exp_avg_sq     = torch.zeros(1, d.smear_ch, dtype=torch.float32, device=dev)
    model.smear_lambda.exp_avg      = torch.zeros(1, dtype=torch.float32, device=dev)
    model.smear_lambda.exp_avg_sq   = torch.zeros(1, dtype=torch.float32, device=dev)
    model.backout_lambda.exp_avg    = torch.zeros(1, dtype=torch.float32, device=dev)
    model.backout_lambda.exp_avg_sq = torch.zeros(1, dtype=torch.float32, device=dev)


def make_step_counter(device):
    """THE schedule position: one (1,) int64 device tensor, owned by the trainer,
    advanced on-device at the end of optimizer_step (host never syncs on it)."""
    return torch.zeros(1, dtype=torch.int64, device=device)


# -----------------------------------------------------------------------------
# The four verbs. Each is reduce (phase 1, async launch) + update (phase 2,
# wait -> owned-shard update -> async gather). world=1 short-circuits every
# comm and the shard IS the whole tensor — same code path, degenerate.
# NOTE: only exercised at world=1 on this box; world>1 is structured per the
# plan but untested until an 8-GPU validation pass.

def muon_sharded_reduce(p, world):
    if world == 1:
        return p.grad32
    K = p.shape[0]
    chunk = -(-K // world)
    padded = torch.zeros(chunk * world, *p.shape[1:], dtype=torch.float32, device=p.device)
    padded[:K].copy_(p.grad32)
    shard = torch.empty(chunk, *p.shape[1:], dtype=torch.float32, device=p.device)
    work = dist.reduce_scatter_tensor(shard, padded, op=dist.ReduceOp.AVG, async_op=True)
    return (work, shard)


def muon_sharded_update(p, red, tabs, mul, t, world, rank, gathers):
    if world == 1:
        muon_step_fused(red, p, p.mantissa, p.momentum, p.second_momentum,
                        tabs, t, mul["ns_steps"], mul["red_dim"], mul["lr_mul"], mul["wd_mul"])
        return
    work, shard = red
    work.wait()
    sl = p.shard          # recorded at allocation; never re-derived from p.shape
    owned = sl.stop - sl.start
    if owned > 0:
        muon_step_fused(shard[:owned], p[sl], p.mantissa[:owned], p.momentum[:owned],
                        p.second_momentum[:owned], tabs, t, mul["ns_steps"], mul["red_dim"],
                        mul["lr_mul"][sl], mul["wd_mul"][sl])
    src = torch.zeros(shard.shape, dtype=p.dtype, device=p.device)  # zero-pad the ragged tail
    if owned > 0:
        src[:owned].copy_(p[sl])
    buf = torch.empty(shard.shape[0] * world, *p.shape[1:], dtype=p.dtype, device=p.device)
    work = dist.all_gather_into_tensor(buf, src, async_op=True)
    gathers.append((work, buf, p, p.shape[0]))


def muon_replicated_update(p, tabs, mul, t, world):
    """all_reduce the grad, every rank runs the same full-size update."""
    if world > 1:
        dist.all_reduce(p.grad32, op=dist.ReduceOp.AVG)
    muon_step_fused(p.grad32, p, p.mantissa, p.momentum, p.second_momentum,
                    tabs, t, mul["ns_steps"], mul["red_dim"], mul["lr_mul"], mul["wd_mul"])


def adamw_sharded_reduce(p, world):
    if world == 1:
        return p.grad32
    g = p.grad32.view(-1, p.shape[-1])
    rs = g.shape[0] // world
    # comms run in the grad buffer's dtype: bf16 for the embeddings, fp32 else
    shard = torch.empty(rs, g.shape[-1], dtype=p.grad32.dtype, device=p.device)
    work = dist.reduce_scatter_tensor(shard, g, op=dist.ReduceOp.AVG, async_op=True)
    return (work, shard)


def adamw_sharded_update(p, red, tabs, eps, t, world, rank, gathers):
    if world == 1:
        adamw_step_fused(p, p.mantissa, red, p.exp_avg, p.exp_avg_sq, tabs, t, eps)
        return
    work, shard = red
    work.wait()
    rows = p.data.view(-1, p.shape[-1])
    rs = rows.shape[0] // world
    live = rows[rank * rs:(rank + 1) * rs]
    adamw_step_fused(live, p.mantissa, shard, p.exp_avg, p.exp_avg_sq, tabs, t, eps)
    work = dist.all_gather_into_tensor(rows, live, async_op=True)
    gathers.append((work, None, None, None))


def adamw_replicated_update(p, tabs, eps, t, world):
    if world > 1:
        dist.all_reduce(p.grad32, op=dist.ReduceOp.AVG)
    assert p.dtype == torch.float32, "replicated scalars are fp32-live"
    adamw_step_fused_fp32(p, p.grad32, p.exp_avg, p.exp_avg_sq, tabs, t, eps)


# -----------------------------------------------------------------------------
# The written-out step.

@torch.no_grad()
def optimizer_step(model, sched, muls, t):
    """One explicit optimizer step, written out per named tensor. 3-phase: launch
    every async reduce; then in launch order wait -> update owned shard -> launch
    gather; then wait all gathers. Ends by advancing the device step counter.
    Reads p.grad32 (Muon MUTATES it — nesterov lerp); caller zeroes afterwards.
    no_grad is load-bearing for the fp32 scalar kernel's in-place leaf updates
    (the mantissa kernels only dodge autograd's leaf check via their int views)."""
    world, rank = _dist_info()
    m, eps = model, sched.adamw_eps

    # Phase 1: launch all gradient reductions (world=1: plain grad32 views)
    r_cq = muon_sharded_reduce(m.c_q, world)
    r_ck = muon_sharded_reduce(m.c_k, world)
    r_cv = muon_sharded_reduce(m.c_v, world)
    r_ap = muon_sharded_reduce(m.attn_proj, world)
    r_fc = muon_sharded_reduce(m.mlp_fc, world)
    r_mp = muon_sharded_reduce(m.mlp_proj, world)
    r_lm = adamw_sharded_reduce(m.lm_head, world)
    r_wt = adamw_sharded_reduce(m.wte, world)
    r_ve = adamw_sharded_reduce(m.value_embeds, world)

    # Phase 2: wait -> update -> launch gather, in launch order (earlier gathers
    # overlap later updates). Replicated params ride along here, all_reduce inline.
    gathers = []
    muon_sharded_update(m.c_q, r_cq, sched.matrix, muls["c_q"], t, world, rank, gathers)
    muon_sharded_update(m.c_k, r_ck, sched.matrix, muls["c_k"], t, world, rank, gathers)
    muon_sharded_update(m.c_v, r_cv, sched.matrix, muls["c_v"], t, world, rank, gathers)
    muon_sharded_update(m.attn_proj, r_ap, sched.matrix, muls["attn_proj"], t, world, rank, gathers)
    muon_sharded_update(m.mlp_fc, r_fc, sched.matrix, muls["mlp_fc"], t, world, rank, gathers)
    muon_sharded_update(m.mlp_proj, r_mp, sched.matrix, muls["mlp_proj"], t, world, rank, gathers)
    muon_replicated_update(m.ve_gate, sched.matrix, muls["ve_gate"], t, world)
    adamw_sharded_update(m.lm_head, r_lm, sched.lm_head, eps, t, world, rank, gathers)
    adamw_sharded_update(m.wte, r_wt, sched.wte, eps, t, world, rank, gathers)
    adamw_sharded_update(m.value_embeds, r_ve, sched.value_embeds, eps, t, world, rank, gathers)
    adamw_replicated_update(m.resid_lambdas, sched.resid, eps, t, world)
    adamw_replicated_update(m.x0_lambdas, sched.x0, eps, t, world)
    adamw_replicated_update(m.smear_gate, sched.smear, eps, t, world)
    adamw_replicated_update(m.smear_lambda, sched.smear, eps, t, world)
    adamw_replicated_update(m.backout_lambda, sched.smear, eps, t, world)

    # Phase 3: wait gathers, copy padded Muon banks back
    for work, buf, p, K in gathers:
        work.wait()
        if p is not None:
            p.copy_(buf[:K])

    t.add_(1)  # advance the schedule on-device


# -----------------------------------------------------------------------------
# Optimizer checkpointing: written out, one line per state tensor.

def optim_state(model):
    """Every optimizer-side tensor as a flat {name: tensor} dict -- the
    optimizer half of a checkpoint, and the manifest load_optim_state fills.

    Saved PER RANK: at world > 1 these are shard-sized, so each rank writes its
    own file through save_checkpoint's existing rank plumbing.

    This used to probe `hasattr(p, attr)` over named_parameters() for five
    known attribute names. That reads as generic but is not: it silently saved
    exactly whichever attributes happened to exist, so a state tensor that was
    never allocated, or one renamed on the allocation side, simply vanished
    from the checkpoint with nothing to notice. Listing them here makes the
    save set the same kind of statement as the allocation it mirrors --
    init_optimizer_state and this function should be read side by side.

    grad32 is deliberately absent: checkpoints are taken after step+zero, so it
    is all zeros. The bf16 LIVE halves are absent too -- they are parameters,
    so they ride in gpt.model_state.

    Old torch.optim states do not port to this layout; fresh runs only.
    """
    return {
        # Muon banks, sharded over `layer`: bf16-live mantissa + moments
        "c_q.mantissa":              model.c_q.mantissa,
        "c_q.momentum":              model.c_q.momentum,
        "c_q.second_momentum":       model.c_q.second_momentum,
        "c_k.mantissa":              model.c_k.mantissa,
        "c_k.momentum":              model.c_k.momentum,
        "c_k.second_momentum":       model.c_k.second_momentum,
        "c_v.mantissa":              model.c_v.mantissa,
        "c_v.momentum":              model.c_v.momentum,
        "c_v.second_momentum":       model.c_v.second_momentum,
        "attn_proj.mantissa":        model.attn_proj.mantissa,
        "attn_proj.momentum":        model.attn_proj.momentum,
        "attn_proj.second_momentum": model.attn_proj.second_momentum,
        "mlp_fc.mantissa":           model.mlp_fc.mantissa,
        "mlp_fc.momentum":           model.mlp_fc.momentum,
        "mlp_fc.second_momentum":    model.mlp_fc.second_momentum,
        "mlp_proj.mantissa":         model.mlp_proj.mantissa,
        "mlp_proj.momentum":         model.mlp_proj.momentum,
        "mlp_proj.second_momentum":  model.mlp_proj.second_momentum,
        # Muon replicated
        "ve_gate.mantissa":          model.ve_gate.mantissa,
        "ve_gate.momentum":          model.ve_gate.momentum,
        "ve_gate.second_momentum":   model.ve_gate.second_momentum,
        # AdamW sharded over `vocab` rows: mantissa + both moments
        "lm_head.mantissa":          model.lm_head.mantissa,
        "lm_head.exp_avg":           model.lm_head.exp_avg,
        "lm_head.exp_avg_sq":        model.lm_head.exp_avg_sq,
        "wte.mantissa":              model.wte.mantissa,
        "wte.exp_avg":               model.wte.exp_avg,
        "wte.exp_avg_sq":            model.wte.exp_avg_sq,
        "value_embeds.mantissa":     model.value_embeds.mantissa,
        "value_embeds.exp_avg":      model.value_embeds.exp_avg,
        "value_embeds.exp_avg_sq":   model.value_embeds.exp_avg_sq,
        # AdamW replicated scalars: fp32-live, so moments only -- no mantissa
        "resid_lambdas.exp_avg":     model.resid_lambdas.exp_avg,
        "resid_lambdas.exp_avg_sq":  model.resid_lambdas.exp_avg_sq,
        "x0_lambdas.exp_avg":        model.x0_lambdas.exp_avg,
        "x0_lambdas.exp_avg_sq":     model.x0_lambdas.exp_avg_sq,
        "smear_gate.exp_avg":        model.smear_gate.exp_avg,
        "smear_gate.exp_avg_sq":     model.smear_gate.exp_avg_sq,
        "smear_lambda.exp_avg":      model.smear_lambda.exp_avg,
        "smear_lambda.exp_avg_sq":   model.smear_lambda.exp_avg_sq,
        "backout_lambda.exp_avg":    model.backout_lambda.exp_avg,
        "backout_lambda.exp_avg_sq": model.backout_lambda.exp_avg_sq,
    }


@torch.no_grad()
def load_optim_state(model, state):
    """Restore into a model that has already been through init_optimizer_state
    (the state tensors must already exist at the right shapes).

    COPIES rather than assigns -- the opposite of load_model_state. These
    buffers are addressed directly by the compiled kernels, so they have to stay
    the same tensor objects; rebinding them would leave the kernels writing into
    the old storage.
    """
    dst = optim_state(model)
    missing = [k for k in dst if k not in state]
    unexpected = [k for k in state if k not in dst]
    assert not missing, f"optimizer checkpoint is missing: {missing}"
    assert not unexpected, f"optimizer checkpoint has unknown entries: {unexpected}"
    for name, t in dst.items():
        loaded = state[name]
        assert tuple(loaded.shape) == tuple(t.shape), \
            f"{name}: checkpoint holds {tuple(loaded.shape)}, allocated {tuple(t.shape)}"
        t.copy_(loaded.to(t.device))
