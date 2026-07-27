"""
GPT model (flattened rewrite for nano-math)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Structure: there are no Block/Attention/MLP submodules. Every parameter belongs
directly to the GPT module (per-layer weights live in ParameterLists/Dicts), all
initialization is consolidated in init_weights(), and the transformer math is
written out inline in two deliberately duplicated forward paths:
  - forward():           packed-varlen training/scoring (cu_seqlens required)
  - forward_inference(): batched KV-cache prefill/decode for generation
The paged fast_engine defines its own two bodies (decode_body/prefill_body) over
these same parameters; the four functions must implement identical math.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat import fp8

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

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


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        n_layer, n_embd = config.n_layer, config.n_embd
        self.head_dim = n_embd // config.n_head
        assert n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        q_dim = config.n_head * self.head_dim
        kv_dim = config.n_kv_head * self.head_dim
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # --- All parameters, owned directly by this module (no submodules) ---
        # Token embedding and unembedding (used via F.embedding / linear())
        self.wte = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
        self.lm_head = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
        # Per-layer attention and MLP weights, F.linear convention: (out_features, in_features)
        self.c_q = nn.ParameterList(nn.Parameter(torch.empty(q_dim, n_embd)) for _ in range(n_layer))
        self.c_k = nn.ParameterList(nn.Parameter(torch.empty(kv_dim, n_embd)) for _ in range(n_layer))
        self.c_v = nn.ParameterList(nn.Parameter(torch.empty(kv_dim, n_embd)) for _ in range(n_layer))
        self.attn_proj = nn.ParameterList(nn.Parameter(torch.empty(n_embd, q_dim)) for _ in range(n_layer))
        self.mlp_fc = nn.ParameterList(nn.Parameter(torch.empty(4 * n_embd, n_embd)) for _ in range(n_layer))
        self.mlp_proj = nn.ParameterList(nn.Parameter(torch.empty(n_embd, 4 * n_embd)) for _ in range(n_layer))
        # Value embeddings (ResFormer-style) + their input-dependent gates: alternating layers, last always included
        self.ve_gate_channels = 12
        ve_layers = [i for i in range(n_layer) if has_ve(i, n_layer)]
        self.value_embeds = nn.ParameterDict({str(i): nn.Parameter(torch.empty(padded_vocab_size, kv_dim)) for i in ve_layers})
        self.ve_gate = nn.ParameterDict({str(i): nn.Parameter(torch.empty(config.n_kv_head, self.ve_gate_channels)) for i in ve_layers})
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer
        # x0_lambdas: blends initial embedding back in at each layer
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.empty(n_layer))
        self.x0_lambdas = nn.Parameter(torch.empty(n_layer))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = nn.Parameter(torch.empty(1, 24))
        self.smear_lambda = nn.Parameter(torch.empty(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(torch.empty(1))

        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # Rotary embeddings are small in memory, so we over-compute generously. With varlen
        # training the full micro-batch is one sequence (T = batch_size * seq_len), so we need
        # enough headroom for that. 64X covers batch sizes up to 64, and the assert in forward
        # will catch if we ever exceed.
        self.rotary_seq_len = config.sequence_len * 64
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
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
        smear_gate:          zeros
        smear_lambda:        zeros (smear disabled at init)
        backout_lambda:      zeros (backout disabled at init)
        """
        n_layer, n_embd = self.config.n_layer, self.config.n_embd

        # Embedding and unembedding
        torch.nn.init.normal_(self.wte, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head, mean=0.0, std=0.001)

        # Per-layer matrices: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for i in range(n_layer):
            torch.nn.init.uniform_(self.c_q[i], -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(self.c_k[i], -s, s)
            torch.nn.init.uniform_(self.c_v[i], -s, s)
            torch.nn.init.zeros_(self.attn_proj[i]) # projections are zero
            torch.nn.init.uniform_(self.mlp_fc[i], -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(self.mlp_proj[i])

        # Value embeddings (init like c_v: uniform with same std) and their gates
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve, -s, s)
        for g in self.ve_gate.values():
            torch.nn.init.uniform_(g, 0.0, 0.02) # small positive so gates start slightly above neutral

        # Per-layer scalars
        for i in range(n_layer):
            # stronger residual at early layers, weaker at deep layers
            self.resid_lambdas[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            # earlier layers get more input embedding blending
            self.x0_lambdas[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars: zeros, matching what from-scratch runs have always
        # actually trained with. The pre-flattening __init__ nominally set
        # backout_lambda=0.2 and a kaiming smear_gate, but on the standard
        # meta-device build path those inits never executed — to_empty() left
        # freshly-allocated (driver-zeroed) storage, so the tuned baselines all
        # started from zeros. Now it's explicit rather than luck.
        torch.nn.init.zeros_(self.smear_gate)
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.zeros_(self.backout_lambda)

        # Rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.data = self.wte.data.to(COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.data = ve.data.to(COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.device
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

    def get_device(self):
        return self.wte.device

    def matrix_parameters(self):
        """All per-layer matmul weights, in the same layer-major order the old
        Block modules registered them (c_q, c_k, c_v, attn_proj, [ve_gate],
        mlp_fc, mlp_proj per layer) so optimizer param-group contents and order
        match pre-flattening checkpoints."""
        params = []
        for i in range(self.config.n_layer):
            params += [self.c_q[i], self.c_k[i], self.c_v[i], self.attn_proj[i]]
            si = str(i)
            if si in self.ve_gate:
                params.append(self.ve_gate[si])
            params += [self.mlp_fc[i], self.mlp_proj[i]]
        return params

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(p.numel() for p in self.value_embeds.values())
        nparams_exclude = (self.wte.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = self.wte.numel()
        value_embeds = sum(p.numel() for p in self.value_embeds.values())
        lm_head = self.lm_head.numel()
        transformer_matrices = sum(p.numel() for p in self.matrix_parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = self.matrix_parameters()
        value_embeds_params = list(self.value_embeds.values())
        embedding_params = [self.wte]
        lm_head_params = [self.lm_head]
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, cu_seqlens=None, loss_reduction='mean'):
        """Training / scoring forward: one packed 1D sequence of documents with
        per-document attention isolation via varlen flash attention. Returns the
        loss if targets are given, else the (softcapped, fp32) logits."""
        assert cu_seqlens is not None, "GPT.forward is the packed-varlen path; use forward_inference for KV-cache generation"
        assert idx.ndim == 1
        idx = idx.unsqueeze(0)
        if targets is not None:
            targets = targets.unsqueeze(0)
        max_seq_len = self.config.sequence_len
        B, T = idx.size()
        nl = self.config.n_layer
        nh, nkv, hd = self.config.n_head, self.config.n_kv_head, self.head_dim
        half = hd // 2 # rotary cache is (1, T, 1, half)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        cos, sin = self.cos[:, :T], self.sin[:, :T] # truncate cache to current sequence length

        # Embed the tokens
        x = F.embedding(idx, self.wte)
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info).
        # Positions attending across document boundaries are handled by position 0 being excluded.
        assert T > 1, "Training forward pass should have T > 1"
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[:, 1:, :24], self.smear_gate))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        backout_layer = nl // 2  # cache at halfway point
        x_backout = None
        for i in range(nl):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # --- attention ---
            xn = norm(x)
            # Project the input to get queries, keys, and values
            # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
            q = linear(xn, self.c_q[i]).view(B, T, nh, hd)
            k = linear(xn, self.c_k[i]).view(B, T, nkv, hd)
            v = linear(xn, self.c_v[i]).view(B, T, nkv, hd)
            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            si = str(i)
            if si in self.value_embeds:
                ve = F.embedding(idx, self.value_embeds[si]).view(B, T, nkv, hd).to(x.dtype)
                g = 3 * torch.sigmoid(linear(xn[..., :self.ve_gate_channels], self.ve_gate[si]))  # (B, T, n_kv_head), range (0, 3)
                v = v + g.unsqueeze(-1) * ve
            # Rotary embeddings (relative positional encoding)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
            # QK norm
            q, k = norm(q), norm(k)
            q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
            k = k * 1.2
            # Varlen flash attention: packed 1D sequence with per-document attention isolation
            y = flash_attn.flash_attn_varlen_func(
                q[0], k[0], v[0],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len, max_seqlen_k=max_seq_len,
                causal=True, window_size=self.window_sizes[i])
            y = y.unsqueeze(0)
            # Re-assemble the heads and project back to residual stream
            x = x + linear(y.contiguous().view(B, T, -1), self.attn_proj[i])
            # --- MLP (relu^2) ---
            x = x + linear(F.relu(linear(norm(x), self.mlp_fc[i])).square(), self.mlp_proj[i])
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = linear(x, self.lm_head) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    def forward_inference(self, idx, kv_cache):
        """Inference forward: batched (B, T) prefill or (B, 1) decode over a
        KVCache (see nanochat.engine). Deliberately duplicates forward()'s math
        with the generation-specific state handling written out: rotary offset by
        cache position, smear across decode steps via kv_cache.prev_embedding,
        and flash_attn_with_kvcache in place of the varlen kernel. Returns the
        (softcapped, fp32) logits."""
        B, T = idx.size()
        nl = self.config.n_layer
        nh, nkv, hd = self.config.n_head, self.config.n_kv_head, self.head_dim
        half = hd // 2 # rotary cache is (1, T, 1, half)

        # Rotary embeddings, offset to the current position in the cache
        T0 = kv_cache.get_pos()
        assert T0 + T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T0 + T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        cos, sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed the tokens
        x = F.embedding(idx, self.wte)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear: read prev embedding from cache, store current for next step
        x_pre_smear = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if T > 1:
            # Prefill: apply smear to positions 1+, same as training
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[:, 1:, :24], self.smear_gate))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif x_pre_smear is not None:
            # Decode: single token, use cached prev embedding
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(linear(x[:, :, :24], self.smear_gate))
            x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x
        backout_layer = nl // 2
        x_backout = None
        for i in range(nl):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # --- attention ---
            xn = norm(x)
            q = linear(xn, self.c_q[i]).view(B, T, nh, hd)
            k = linear(xn, self.c_k[i]).view(B, T, nkv, hd)
            v = linear(xn, self.c_v[i]).view(B, T, nkv, hd)
            si = str(i)
            if si in self.value_embeds:
                ve = F.embedding(idx, self.value_embeds[si]).view(B, T, nkv, hd).to(x.dtype)
                g = 3 * torch.sigmoid(linear(xn[..., :self.ve_gate_channels], self.ve_gate[si]))
                v = v + g.unsqueeze(-1) * ve
            # Rotary embeddings (relative positional encoding)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
            # QK norm
            q, k = norm(q), norm(k)
            q = q * 1.2
            k = k * 1.2
            # flash_attn_with_kvcache appends k/v to the cache and attends over it
            k_cache, v_cache = kv_cache.get_layer_cache(i)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=self.window_sizes[i],
            )
            x = x + linear(y.contiguous().view(B, T, -1), self.attn_proj[i])
            # --- MLP (relu^2) ---
            x = x + linear(F.relu(linear(norm(x), self.mlp_fc[i])).square(), self.mlp_proj[i])
            if i == backout_layer:
                x_backout = x
        kv_cache.advance(T) # all layers have written their KV for these positions
        # Subtract mid-layer residual to remove low-level features before logit projection
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        logits = linear(x, self.lm_head)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

# -----------------------------------------------------------------------------
# Helpers for the fp32 master + bf16 live variant of the optimizers
# -----------------------------------------------------------------------------

from nanochat.optim import Fp32DistMuonAdamW, Fp32MuonAdamW

def cast_model_bf16(model) -> None:
    """Cast all parameters to bf16 in place (Parameter objects keep identity, so
    optimizer/param-group references and later graph captures stay valid).
    Rotary cos/sin buffers are already COMPUTE_DTYPE (bf16 on CUDA)."""
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model.to(torch.bfloat16)

def setup_fp32_optimizer(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                         weight_decay=0.0, scalar_lr=0.5):
    """GPT.setup_optimizer's exact param-group split, but constructing the
    fp32-master variant. Call while the model still holds fp32 weights;
    cast to bf16 afterwards."""
    model_dim = model.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()

    matrix_params = model.matrix_parameters()
    value_embeds_params = list(model.value_embeds.values())
    embedding_params = [model.wte]
    lm_head_params = [model.lm_head]
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]
    smear_params = [model.smear_gate, model.smear_lambda, model.backout_lambda]
    assert len(list(model.parameters())) == (len(matrix_params) + len(embedding_params)
                                             + len(lm_head_params) + len(value_embeds_params)
                                             + len(resid_params) + len(x0_params) + len(smear_params))

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    param_groups = [
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
        ))
    Factory = Fp32DistMuonAdamW if ddp else Fp32MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
