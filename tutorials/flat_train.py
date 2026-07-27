# 1. Helpers
#  - Config object
#  - Fused optimizer helpers
#  - Minimal model helpers

# One global config object, functions reference `cfg` directly,
# no function arguments for config values.

# Two modules: 
# - model
# - optim
#
# They do ~2 things:
# - Hold tensors
# - Initialize those tensors
#    - Or load them from a checkpoint.
#
# They do not hold exection code.

# --- Model ---

# 2. Parameter definitions
# - Bank style, with layers as the first dimension.
#    - Keep W_in and W_out as separate parameters.
#    - q, k, v, o either get stacked from the start, or before optims, so maybe from start.

# 3. Parameter initialization.

# --- Optim ---
# 4. State and schedule definitions
# - Hold state tensors, by name, same as model.
# - Hold schedule tensors, also by name.
#
# For init:
# - Precompute schedules.

# --- Main ---

# 5. Train step function (single optimizer step):
# - For-loop over micro-steps:
#   - Forward
#   - Loss
#   - loss.backward()
# - Pass each parameter to its appropriate optimizer helper.
#   - Hardcode the constants at the call site.


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat import fp8

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

# TODO - Rename this to Config, it captures model and optimizer settings.
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"



# ==== Helpers ====

# ==== Optimizer helpers ====
import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    sched_t: Tensor,        # (1,) - int64 DEVICE tensor, schedule index (0-based)
    nstep_t: Tensor,        # (1,) - int64 DEVICE tensor, updates taken BEFORE this one
    lr_tab: Tensor,         # (S,) - fp32 DEVICE table, per-step learning rate
    beta1_t: Tensor,        # () - 0-D fp32 DEVICE constant, beta1
    beta2_t: Tensor,        # () - 0-D fp32 DEVICE constant, beta2
    eps_t: Tensor,          # () - 0-D fp32 DEVICE constant, epsilon
    wd_t: Tensor,           # () - 0-D fp32 DEVICE constant, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    Hyperparameters live on device: the lr schedule is a pre-computed table
    indexed by the (1,)-shaped step counter, constants are 0-D tensors — one
    compiled kernel serves every group and every step, and the whole update is
    CUDA-graph capturable.
    """
    lr_t = lr_tab[sched_t]                        # (1,), broadcasts below
    steps_done = (nstep_t + 1).to(torch.float32)  # 1-based count for bias correction
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** steps_done
    bias2 = 1 - beta2_t ** steps_done
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.sub_(step_size * (exp_avg / denom))

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    sched_t: Tensor,                # (1,) - int64 DEVICE tensor, schedule index (0-based)
    momentum_tab: Tensor,           # (S,) - fp32 DEVICE table, per-step momentum coefficient
    lr_tab: Tensor,                 # (S,) - fp32 DEVICE table, per-step lr (aspect scale pre-folded)
    wd_tab: Tensor,                 # (S,) - fp32 DEVICE table, per-step weight decay
    beta2_t: Tensor,                # () - 0-D fp32 DEVICE constant, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Hyperparameters live on device: pre-computed tables indexed by the
    (1,)-shaped step counter; the (1,) values broadcast through the update.
    """
    momentum_t = momentum_tab[sched_t]
    lr_t = lr_tab[sched_t]
    wd_t = wd_tab[sched_t]

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# ==== Model helpers ====

def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

def linear(x, w):
    """F.linear with the weight cast to the input dtype (replaces the old Linear
    module): master weights stay fp32 for optimizer precision, but matmuls run in
    the activation dtype (typically bf16 from embeddings). No-op cast when the
    model has been cast to bf16 (RL fast path).
    Under FP8 training (base_train --fp8, H100+), eligible matmuls route through
    torch._scaled_mm instead — see nanochat/fp8.py."""
    if fp8.enabled and fp8.eligible(w):
        return fp8.fp8_linear(x, w)
    return F.linear(x, w.to(dtype=x.dtype))

def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
    # TODO: bump base theta more? e.g. 100K is more common more recently
    # autodetect the device from model embeddings
    if device is None:
        device = wte.device
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
    return cos, sin

def _compute_window_sizes(self, config):
    """
    Compute per-layer window sizes for sliding window attention.

    Returns list of (left, right) tuples for FA3's window_size parameter:
    - left: how many tokens before current position to attend to (-1 = unlimited)
    - right: how many tokens after current position to attend to (0 for causal)

    Pattern string is tiled across layers. Final layer always gets L (full context).
    Characters: L=long (full context), S=short (quarter context)
    """
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    # Map characters to window sizes
    long_window = config.sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    # Tile pattern across layers
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    # Final layer always gets full context
    window_sizes[-1] = (long_window, 0)
    return window_sizes



# ==== Parameter definitions ====

# super().__init__()
# config = config
n_layer, n_embd = config.n_layer, config.n_embd
head_dim = n_embd // config.n_head
assert n_embd % config.n_head == 0
assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
q_dim = config.n_head * head_dim
kv_dim = config.n_kv_head * head_dim
# Compute per-layer window sizes for sliding window attention
# window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
window_sizes = _compute_window_sizes(config)
# Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
# https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
if padded_vocab_size != config.vocab_size:
    print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
padded_vocab_size = padded_vocab_size

# --- All parameters, owned directly by this module (no submodules) ---
# Token embedding and unembedding (used via F.embedding / linear())
wte = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
lm_head = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
# Per-layer attention and MLP weights, F.linear convention: (out_features, in_features)
c_q = nn.ParameterList(nn.Parameter(torch.empty(q_dim, n_embd)) for _ in range(n_layer))
c_k = nn.ParameterList(nn.Parameter(torch.empty(kv_dim, n_embd)) for _ in range(n_layer))
c_v = nn.ParameterList(nn.Parameter(torch.empty(kv_dim, n_embd)) for _ in range(n_layer))
attn_proj = nn.ParameterList(nn.Parameter(torch.empty(n_embd, q_dim)) for _ in range(n_layer))
mlp_fc = nn.ParameterList(nn.Parameter(torch.empty(4 * n_embd, n_embd)) for _ in range(n_layer))
mlp_proj = nn.ParameterList(nn.Parameter(torch.empty(n_embd, 4 * n_embd)) for _ in range(n_layer))
# Value embeddings (ResFormer-style) + their input-dependent gates: alternating layers, last always included
ve_gate_channels = 12
ve_layers = [i for i in range(n_layer) if has_ve(i, n_layer)]
value_embeds = nn.ParameterDict({str(i): nn.Parameter(torch.empty(padded_vocab_size, kv_dim)) for i in ve_layers})
ve_gate = nn.ParameterDict({str(i): nn.Parameter(torch.empty(config.n_kv_head, ve_gate_channels)) for i in ve_layers})
# Per-layer learnable scalars (inspired by modded-nanogpt)
# resid_lambdas: scales the residual stream at each layer
# x0_lambdas: blends initial embedding back in at each layer
# Separate parameters so they can have different optimizer treatment
resid_lambdas = nn.Parameter(torch.empty(n_layer))
x0_lambdas = nn.Parameter(torch.empty(n_layer))
# Smear: mix previous token's embedding into current token (cheap bigram-like info)
smear_gate = nn.Parameter(torch.empty(1, 24))
smear_lambda = nn.Parameter(torch.empty(1))
# Backout: subtract cached mid-layer residual before final norm to remove low-level features
backout_lambda = nn.Parameter(torch.empty(1))

# To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
# Rotary embeddings are small in memory, so we over-compute generously. With varlen
# training the full micro-batch is one sequence (T = batch_size * seq_len), so we need
# enough headroom for that. 64X covers batch sizes up to 64, and the assert in forward
# will catch if we ever exceed.
rotary_seq_len = config.sequence_len * 64
cos, sin = _precompute_rotary_embeddings(rotary_seq_len, head_dim)
register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
register_buffer("sin", sin, persistent=False)

nh, nkv, hd = config.n_head, config.n_kv_head, head_dim
half = hd // 2   # rotary cache is (1, T, 1, half)

# ==== State and schedule definitions ====

# TODO - Define e.g. momentum buffers for each parameter.


# ==== Parameter Initialization ====

"""
Initialize the full model in this one function for maximum clarity.

wte (embedding):     normal, std=0.8
lm_head:             normal, std=0.001
for each layer:
    c_q:             uniform, std=1/sqrt(n_embd)
    c_k:             uniform, std=1/sqrt(n_embd)
    c_v:             uniform, std=1/sqrt(n_embd)
    attn_proj:       zeros
    mlp_fc:          uniform, std=0.4/sqrt(n_embd)
    mlp_proj:        zeros
value_embeds:        uniform, std=1/sqrt(n_embd)
ve_gate:             uniform in [0, 0.02] (slightly above neutral)
resid_lambdas:       1.15 -> 1.05 linear decay over depth
x0_lambdas:          0.20 -> 0.05 linear decay over depth
smear_gate:          uniform, nn.Linear default bound 1/sqrt(24)
smear_lambda:        zeros (smear disabled at init)
backout_lambda:      0.2
"""
n_layer, n_embd = config.n_layer, config.n_embd

# Embedding and unembedding
torch.nn.init.normal_(wte, mean=0.0, std=0.8)
torch.nn.init.normal_(lm_head, mean=0.0, std=0.001)

# Per-layer matrices: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
for i in range(n_layer):
    torch.nn.init.uniform_(c_q[i], -s, s) # weights use Uniform to avoid outliers
    torch.nn.init.uniform_(c_k[i], -s, s)
    torch.nn.init.uniform_(c_v[i], -s, s)
    torch.nn.init.zeros_(attn_proj[i]) # projections are zero
    torch.nn.init.uniform_(mlp_fc[i], -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
    torch.nn.init.zeros_(mlp_proj[i])

# Value embeddings (init like c_v: uniform with same std) and their gates
for ve in value_embeds.values():
    torch.nn.init.uniform_(ve, -s, s)
for g in ve_gate.values():
    torch.nn.init.uniform_(g, 0.0, 0.02) # small positive so gates start slightly above neutral

# Per-layer scalars
for i in range(n_layer):
    # stronger residual at early layers, weaker at deep layers
    resid_lambdas[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
    # earlier layers get more input embedding blending
    x0_lambdas[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

# Smear/backout scalars. (Previously these were only set in __init__, which runs
# under meta device in the standard build path, leaving them uninitialized
# after to_empty(); now they are properly initialized here with the rest.)
torch.nn.init.uniform_(smear_gate, -24**-0.5, 24**-0.5) # nn.Linear default bound
torch.nn.init.zeros_(smear_lambda)
backout_lambda.fill_(0.2)

# Rotary embeddings
cos, sin = _precompute_rotary_embeddings(rotary_seq_len, head_dim)
cos, sin = cos, sin

# Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
# embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
# because GradScaler cannot unscale fp16 gradients.
if COMPUTE_DTYPE != torch.float16:
    wte.data = wte.data.to(COMPUTE_DTYPE)
    for ve in value_embeds.values():
        ve.data = ve.data.to(COMPUTE_DTYPE)


# ==== Train step function ====
# TODO - Inputs
@torch.compile(dynamic=False, fullgraph=True)
def train_step():

    # Accumulate gradients over multiple forward-backward passes (each on a micro-batch of examples)
    # before actually updating the weights.
    for micro_step in range(micro_steps):
        #def forward(self, idx, targets=None, cu_seqlens=None, loss_reduction='mean'):
        """Training / scoring forward: one packed 1D sequence of documents with
        per-document attention isolation via varlen flash attention. Returns the
        loss if targets are given, else the (softcapped, fp32) logits."""
        assert cu_seqlens is not None, "GPT.forward is the packed-varlen path; use forward_inference for KV-cache generation"
        assert idx.ndim == 1
        idx = idx.unsqueeze(0)
        if targets is not None:
            targets = targets.unsqueeze(0)
        max_seq_len = config.sequence_len
        B, T = idx.size()
        nl = config.n_layer
        nh, nkv, hd = config.n_head, config.n_kv_head, head_dim

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {cos.size(1)}"
        assert idx.device == cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {cos.device}"
        assert cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {cos.dtype}"
        cos, sin = cos[:, :T], sin[:, :T] # truncate cache to current sequence length

        # Embed the tokens
        x = F.embedding(idx, wte)
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info).
        # Positions attending across document boundaries are handled by position 0 being excluded.
        assert T > 1, "Training forward pass should have T > 1"
        gate = smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[:, 1:, :24], smear_gate))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        backout_layer = nl // 2  # cache at halfway point
        x_backout = None
        for i in range(nl):
            x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            # --- attention ---
            xn = norm(x)
            # Project the input to get queries, keys, and values
            # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
            q = linear(xn, c_q[i]).view(B, T, nh, hd)
            k = linear(xn, c_k[i]).view(B, T, nkv, hd)
            v = linear(xn, c_v[i]).view(B, T, nkv, hd)
            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            si = str(i)
            if si in value_embeds:
                ve = F.embedding(idx, value_embeds[si]).view(B, T, nkv, hd).to(x.dtype)
                g = 3 * torch.sigmoid(linear(xn[..., :ve_gate_channels], ve_gate[si]))  # (B, T, n_kv_head), range (0, 3)
                v = v + g.unsqueeze(-1) * ve
            
            # Rotary embeddings (relative positional encoding)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
            
            # QK norm and scale
            q, k = norm(q), norm(k)        
            # Sharpen attention (Same concept as Temperature on LM head. 1.44 = more deterministic)
            q = q * 1.2  
            k = k * 1.2 

            # Varlen flash attention: packed 1D sequence with per-document attention isolation
            y = flash_attn.flash_attn_varlen_func(
                q[0], k[0], v[0],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len, max_seqlen_k=max_seq_len,
                causal=True, window_size=window_sizes[i])
            y = y.unsqueeze(0)
            # Re-assemble the heads and project back to residual stream
            x = x + linear(y.contiguous().view(B, T, -1), attn_proj[i])
            # --- MLP (relu^2) ---
            x = x + linear(F.relu(linear(norm(x), mlp_fc[i])).square(), mlp_proj[i])
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        x = x - backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = linear(x, lm_head) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
        
        # training: given the targets, compute and return the loss
        # TODO experiment with chunked cross-entropy?
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

        # loss.backward()

    # Smooth the gradients and update weights.

    # TODO - Pass each parameter's gradients and state to the appropraite helper, hardcode the constants at the call site here.
    adamw_step_fused(m.wte, m.wte.grad, o.wte_m1, o.wte_m2) # , ...) TODO

# Training loop.


