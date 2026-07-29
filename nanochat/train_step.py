"""Handwritten training step for base pretraining: explicit forward+backward
(no autograd) over the banked GPT, accumulating into fp32 `.grad32` buffers.

forward_backward() is the 5th deliberately-duplicated forward body (after
GPT.forward, GPT.forward_inference, fast_engine's decode/prefill) — its forward
half mirrors GPT.forward line for line, then the backward half walks the same
ops in reverse. GPT.forward stays autograd-capable and is the gradient-parity
oracle (tests/test_grad_parity.py).

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

import torch
import torch.nn.functional as F

# Read via the gpt module so the parity tests' COMPUTE_DTYPE monkeypatch (fp64
# tier) applies to GPT.forward and this file in one place.
from nanochat import gpt as gpt_mod
from nanochat.flash_attention import flash_attn_varlen_fwd_lse, flash_attn_varlen_bwd


# -----------------------------------------------------------------------------
# grad32 buffers

def init_grad_buffers(model, dtype=torch.float32):
    """Attach a full-size, zeroed `.grad32` to every parameter. Call once, after
    the model is on its device; zero with zero_grad32() between steps."""
    for p in model.parameters():
        p.grad32 = torch.zeros(p.shape, dtype=dtype, device=p.device)


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
    max_seq_len = cfg.sequence_len
    dt = gpt_mod.COMPUTE_DTYPE
    g32 = model.wte.grad32.dtype
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
        qn, r_q = _rms_fwd(q, hd)                # stash PRE-1.2 so the norm bwd is exact;
        kn, r_k = _rms_fwd(k, hd)                # qf/kf recomputed bitwise in backward
        qf = qn * 1.2
        kf = kn * 1.2
        y, lse = flash_attn_varlen_fwd_lse(qf, kf, v, cu_seqlens, max_seq_len, model.window_sizes[i])
        y = y.contiguous()
        x1 = b + y.view(T, -1) @ model.attn_proj[i].to(dt).mT
        xm, _ = _rms_fwd(x1, D)                  # xm recomputed in backward from stashed x1
        a = F.relu(xm @ model.mlp_fc[i].to(dt).mT)
        x = x1 + a.square() @ model.mlp_proj[i].to(dt).mT
        if i == backout_layer:
            x_backout = x
        stash.append(dict(x_in=x_in, xn=xn, r_xn=r_xn, qn=qn, kn=kn, r_q=r_q, r_k=r_k,
                          v=v, y=y, lse=lse, x1=x1, a=a))

    x_pre = x - model.backout_lambda.to(dt) * x_backout
    xf, r_f = _rms_fwd(x_pre, D)

    # lm_head + softcap + CE loss + dlogits, IN PLACE on the one (T, V) buffer.
    softcap = 15.0
    logits = xf @ model.lm_head.to(dt).mT        # (T, Vp) compute-dtype
    buf_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    buf = logits[..., :V].to(buf_dtype)          # THE big fp32 buffer; bf16 logits freed below
    del logits
    valid = targets >= 0
    n_valid = valid.sum()
    buf.div_(softcap).tanh_().mul_(softcap)      # buf = cap = 15*tanh(z/15)
    # Softcap keeps cap in [-15, 15], so softmax probs are bounded below by
    # ~exp(-30)/V — no underflow anywhere in the in-place chain that follows.
    loss_sum = torch.zeros((), dtype=buf_dtype, device=buf.device)
    scale = loss_scale / n_valid.to(buf_dtype)   # 0-dim device tensor; no host sync
    for r0 in range(0, T, loss_chunk):
        c = buf[r0:r0 + loss_chunk]              # in-place view; temps are chunk-sized
        vc = valid[r0:r0 + loss_chunk].to(buf_dtype)
        y_safe = targets[r0:r0 + loss_chunk].clamp_min(0).unsqueeze(1)
        cap_y = c.gather(1, y_safe).squeeze(1)
        m = c.amax(dim=1, keepdim=True)
        c.sub_(m).exp_()                         # c = exp(cap - m)
        ssum = c.sum(dim=1, keepdim=True)
        loss_sum += ((ssum.log() + m).squeeze(1) - cap_y).mul_(vc).sum()
        f = c.log().add_(m)                      # f = cap, recovered exactly
        f.div_(softcap).square_().neg_().add_(1.0)   # f = 1 - tanh^2(z/15)
        c.div_(ssum)                             # c = softmax(cap)
        c.scatter_add_(1, y_safe, (-vc).unsqueeze(1))  # - onehot on valid rows
        c.mul_(f).mul_(vc.unsqueeze(1) * scale)  # softcap chain, ignore-mask, 1/n_valid, loss_scale
    loss = loss_sum / n_valid.to(buf_dtype)

    # ==== backward half ====
    dz = buf.to(dt)                              # mirror autograd's cast back through .float()
    del buf
    lm = model.lm_head.to(dt)
    model.lm_head.grad32[:V].add_((dz.mT @ xf).to(g32))  # padded rows get no grad, as in autograd
    dxf = dz @ lm[:V]
    del dz

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
        model.mlp_proj.grad32[i].add_((d_stream.mT @ a.square()).to(g32))
        d_h = 2.0 * a * d_u
        xm, r_xm = _rms_fwd(x1, D)               # cheap recompute (bitwise: same input)
        model.mlp_fc.grad32[i].add_((d_h.mT @ xm).to(g32))
        d_xm = d_h @ model.mlp_fc[i].to(dt)
        d_x1 = d_stream + _rms_bwd(d_xm, xm, r_xm)
        # --- attention backward ---
        xn, y = st["xn"], st["y"]
        model.attn_proj.grad32[i].add_((d_x1.mT @ y.view(T, -1)).to(g32))
        d_y = (d_x1 @ model.attn_proj[i].to(dt)).view(T, nh, hd)
        qf = st["qn"] * 1.2                      # bitwise identical to forward's kernel inputs
        kf = st["kn"] * 1.2
        dqf, dkf, dv = flash_attn_varlen_bwd(
            d_y, qf, kf, st["v"], y, st["lse"], cu_seqlens, max_seq_len,
            model.window_sizes[i], deterministic=deterministic_attention)
        d_qn = 1.2 * dqf
        d_kn = 1.2 * dkf
        d_qr = _rms_bwd(d_qn, st["qn"], st["r_q"])   # per-(token, head) norm, dim = head_dim
        d_kr = _rms_bwd(d_kn, st["kn"], st["r_k"])
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
            model.ve_gate.grad32[j].add_((d_zg.mT @ xn[..., :gch]).to(g32))
            d_ve = (dv * (3 * sg).unsqueeze(-1)).reshape(T, nkv * hd)
            model.value_embeds.grad32[j].index_add_(0, idx, d_ve.to(g32))
            d_xn_ve = d_zg @ model.ve_gate[j].to(dt)
        # dv passes through the VE add unchanged: v = v0 + g*ve
        d_q0 = d_q0.view(T, nh * hd)
        d_k0 = d_k0.view(T, nkv * hd)
        d_v0 = dv.reshape(T, nkv * hd)
        model.c_q.grad32[i].add_((d_q0.mT @ xn).to(g32))
        model.c_k.grad32[i].add_((d_k0.mT @ xn).to(g32))
        model.c_v.grad32[i].add_((d_v0.mT @ xn).to(g32))
        d_xn = d_q0 @ model.c_q[i].to(dt) + d_k0 @ model.c_k[i].to(dt) + d_v0 @ model.c_v[i].to(dt)
        if d_xn_ve is not None:
            d_xn[:, :gch] += d_xn_ve
        d_b = d_x1 + _rms_bwd(d_xn, xn, st["r_xn"])
        # --- blend backward: b = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        model.resid_lambdas.grad32[i].add_((d_b * st["x_in"]).sum(dtype=g32))
        model.x0_lambdas.grad32[i].add_((d_b * x0).sum(dtype=g32))
        d_x0 = d_x0 + model.x0_lambdas[i] * d_b  # TRAP: x0 feeds every layer, accumulate
        d_stream = model.resid_lambdas[i] * d_b
        stash[i] = None                          # free this layer's stash as we go

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
    model.wte.grad32.index_add_(0, idx, d_emb.to(g32))

    return loss
