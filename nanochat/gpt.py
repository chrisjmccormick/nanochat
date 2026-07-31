"""
GPT model (flattened rewrite for nano-math, banked for fwd-bwd)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers (in fact, no F.linear at all — raw matmuls)
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Structure: there are no Block/Attention/MLP submodules, and per-layer weights are
BANKS — one stacked Parameter per role, layer index on dim 0. `bank[i]` is a free
(out_features, in_features) view for layer i, consumed as `x @ w.to(x.dtype).mT`
(the F.linear convention without F.linear; the cast is a no-op once the model is
bf16). All initialization is consolidated in init_weights(), and the transformer
math is written out inline in two deliberately duplicated forward paths:
  - forward():           packed-varlen training/scoring, unbatched (T, ...)
  - forward_inference(): batched KV-cache prefill/decode for generation, (B, T, ...)
The paged fast_engine defines its own two bodies (decode_body/prefill_body) over
these same parameters, and nanochat/train_step.py adds forward_backward() (the
handwritten no-autograd training step); all bodies must implement identical math.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0, COMPUTE_DTYPE

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
        # Token embedding and unembedding (used via F.embedding / raw matmul)
        self.wte = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
        self.lm_head = nn.Parameter(torch.empty(padded_vocab_size, n_embd))
        # Per-layer attention and MLP weights as banks: layer index on dim 0, each
        # slice in the (out_features, in_features) convention of F.linear.
        self.c_q = nn.Parameter(torch.empty(n_layer, q_dim, n_embd))
        self.c_k = nn.Parameter(torch.empty(n_layer, kv_dim, n_embd))
        self.c_v = nn.Parameter(torch.empty(n_layer, kv_dim, n_embd))
        self.attn_proj = nn.Parameter(torch.empty(n_layer, n_embd, q_dim))
        self.mlp_fc = nn.Parameter(torch.empty(n_layer, 4 * n_embd, n_embd))
        self.mlp_proj = nn.Parameter(torch.empty(n_layer, n_embd, 4 * n_embd))
        # Value embeddings (ResFormer-style) + their input-dependent gates: alternating
        # layers, last always included. Banked over just the VE layers; ve_index maps
        # layer index -> bank slot (-1 = layer has no VE), used by every forward body.
        self.ve_gate_channels = 12
        self.ve_layers = [i for i in range(n_layer) if has_ve(i, n_layer)]
        self.ve_index = [self.ve_layers.index(i) if i in self.ve_layers else -1 for i in range(n_layer)]
        n_ve = len(self.ve_layers)
        self.value_embeds = nn.Parameter(torch.empty(n_ve, padded_vocab_size, kv_dim))
        self.ve_gate = nn.Parameter(torch.empty(n_ve, config.n_kv_head, self.ve_gate_channels))
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
        c_q, c_k, c_v:       uniform, std=1/sqrt(n_embd)   (whole bank)
        attn_proj:           zeros
        mlp_fc:              uniform, std=0.4/sqrt(n_embd) (whole bank)
        mlp_proj:            zeros
        value_embeds:        uniform, std=1/sqrt(n_embd)   (whole bank)
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

        # Matrix banks: uniform init with bound = sqrt(3) * std (same standard deviation
        # as normal). The slices are drawn PER LAYER in the pre-bank iteration order,
        # NOT as one whole-bank call: with the same seed this reproduces the old
        # per-layer model's weights bit-for-bit, so training curves stay directly
        # comparable across the flattening. (uniform_ on a contiguous slice view
        # draws exactly what a standalone tensor of that shape would.)
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for i in range(n_layer):
            torch.nn.init.uniform_(self.c_q[i], -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(self.c_k[i], -s, s)
            torch.nn.init.uniform_(self.c_v[i], -s, s)
            torch.nn.init.uniform_(self.mlp_fc[i], -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
        torch.nn.init.zeros_(self.attn_proj) # projections are zero (no RNG consumed)
        torch.nn.init.zeros_(self.mlp_proj)

        # Value embeddings (init like c_v: uniform with same std) and their gates.
        # Draw order follows the old ParameterDict's iteration: string keys in
        # SORTED order ('1','11','3',...), not ascending layers — again so the
        # same seed reproduces the pre-bank weights exactly.
        for layer in sorted(self.ve_layers, key=str):
            torch.nn.init.uniform_(self.value_embeds[self.ve_index[layer]], -s, s)
        for layer in sorted(self.ve_layers, key=str):
            torch.nn.init.uniform_(self.ve_gate[self.ve_index[layer]], 0.0, 0.02) # small positive so gates start slightly above neutral

        # Per-layer scalars (per-element Python-float math, matching the old init's rounding)
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
            self.value_embeds.data = self.value_embeds.data.to(COMPUTE_DTYPE)

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
        """All matmul weight banks (Muon-updated). One entry per role — each is a
        single stacked Parameter, not a per-layer list."""
        return [self.c_q, self.c_k, self.c_v, self.attn_proj, self.mlp_fc, self.mlp_proj, self.ve_gate]

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
        nparams_exclude = (self.wte.numel() + self.value_embeds.numel() +
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
        # Count each group separately (mirrors the roles in named_parameter_lists)
        wte = self.wte.numel()
        value_embeds = self.value_embeds.numel()
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

    def named_parameter_lists(self):
        """The model's parameters bucketed by the role that decides how each is
        optimized — the one part of optimizer setup that is genuinely model
        knowledge. The base pretraining path (nanochat/train_step.py) addresses
        parameters by name instead, but eval/other scripts still consume these.

        The assert is the guarantee callers rely on: every parameter appears in
        exactly one list, so no parameter can silently go untrained.
        """
        lists = {
            'matrix': self.matrix_parameters(),
            'lm_head': [self.lm_head],
            'wte': [self.wte],
            'value_embeds': [self.value_embeds],
            'resid': [self.resid_lambdas],
            'x0': [self.x0_lambdas],
            'smear': [self.smear_gate, self.smear_lambda, self.backout_lambda],
        }
        covered = sum(len(v) for v in lists.values())
        total = len(list(self.parameters()))
        assert covered == total, f"parameter roles cover {covered} of {total} parameters"
        return lists

    def forward(self, idx, cu_seqlens, targets=None, loss_reduction='mean'):
        """Training / scoring forward: one packed 1D sequence of documents with
        per-document attention isolation via varlen flash attention. This path is
        unbatched — idx/targets are (T,) and activations stay (T, ...) throughout,
        which is the layout the varlen kernel wants. (Use forward_inference for
        the batched KV-cache generation path.) Returns the loss if targets are
        given, else the (softcapped, fp32) logits (T, vocab_size).

        This is the autograd-capable reference body: the eval path, and the
        gradient-parity oracle for train_step.forward_backward()."""
        assert idx.ndim == 1
        max_seq_len = self.config.sequence_len
        T = idx.size(0)
        nl = self.config.n_layer
        nh, nkv, hd = self.config.n_head, self.config.n_kv_head, self.head_dim
        model_dim = self.config.n_embd
        half = hd // 2 # rotary cache is (T, 1, half)
        # RMS norm over the trailing dim, with the expected width bound in. Naming the two
        # widths separately keeps the per-head-ness of QK norm visible at the call site, and
        # unlike (x.size(-1),) it lets rms_norm's shape check actually fire on a bad tensor.
        # Note that these run in bf16, seems ok.
        res_norm = lambda t: F.rms_norm(t, (model_dim,)) # residual stream: one RMS per token
        qk_norm = lambda t: F.rms_norm(t, (hd,)) # queries/keys: one RMS per (token, head)

        # Grab the rotary embeddings for the current sequence length (the cache is (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        cos, sin = self.cos[0, :T], self.sin[0, :T] # truncate to T and drop the batch dim -> (T, 1, half)

        # Embed the tokens
        x = F.embedding(idx, self.wte)
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = res_norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info).
        # Positions attending across document boundaries are handled by position 0 being excluded.
        assert T > 1, "Training forward pass should have T > 1"
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(x[1:, :24] @ self.smear_gate.to(x.dtype).mT)
        x = torch.cat([x[:1], x[1:] + gate * x[:-1]], dim=0)

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        backout_layer = nl // 2  # cache at halfway point
        x_backout = None
        for i in range(nl):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # --- attention ---
            xn = res_norm(x)
            # Project the input to get queries, keys, and values
            # Shape: (T, H, D) - the varlen kernel's native layout, no transpose needed!
            q = (xn @ self.c_q[i].to(x.dtype).mT).view(T, nh, hd)
            k = (xn @ self.c_k[i].to(x.dtype).mT).view(T, nkv, hd)
            v = (xn @ self.c_v[i].to(x.dtype).mT).view(T, nkv, hd)
            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            j = self.ve_index[i]
            if j >= 0:
                ve = F.embedding(idx, self.value_embeds[j]).view(T, nkv, hd).to(x.dtype)
                g = 3 * torch.sigmoid(xn[..., :self.ve_gate_channels] @ self.ve_gate[j].to(x.dtype).mT)  # (T, n_kv_head), range (0, 3)
                v = v + g.unsqueeze(-1) * ve
            # Rotary embeddings (relative positional encoding)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
            # QK norm
            q, k = qk_norm(q), qk_norm(k)
            q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
            k = k * 1.2
            # Varlen flash attention: packed 1D sequence with per-document attention isolation
            y = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len, max_seqlen_k=max_seq_len,
                causal=True, window_size=self.window_sizes[i])
            # Re-assemble the heads and project back to residual stream
            x = x + y.contiguous().view(T, -1) @ self.attn_proj[i].to(x.dtype).mT
            # --- MLP (relu^2) ---
            x = x + F.relu(res_norm(x) @ self.mlp_fc[i].to(x.dtype).mT).square() @ self.mlp_proj[i].to(x.dtype).mT
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = res_norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = x @ self.lm_head.to(x.dtype).mT # (T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        if logits.dtype != torch.float64: # fp32 for softcap+loss; fp64 stays (the exact-parity tier)
            logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits, targets, ignore_index=-1, reduction=loss_reduction)
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
        model_dim = self.config.n_embd
        half = hd // 2 # rotary cache is (1, T, 1, half)
        res_norm = lambda t: F.rms_norm(t, (model_dim,)) # residual stream: one RMS per token
        qk_norm = lambda t: F.rms_norm(t, (hd,)) # queries/keys: one RMS per (token, head)

        # Rotary embeddings, offset to the current position in the cache
        T0 = kv_cache.get_pos()
        assert T0 + T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T0 + T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        cos, sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed the tokens
        x = F.embedding(idx, self.wte)
        x = x.to(COMPUTE_DTYPE)
        x = res_norm(x)

        # Smear: read prev embedding from cache, store current for next step
        x_pre_smear = kv_cache.prev_embedding
        kv_cache.prev_embedding = x[:, -1:, :]
        if T > 1:
            # Prefill: apply smear to positions 1+, same as training
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(x[:, 1:, :24] @ self.smear_gate.to(x.dtype).mT)
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        elif x_pre_smear is not None:
            # Decode: single token, use cached prev embedding
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(x[:, :, :24] @ self.smear_gate.to(x.dtype).mT)
            x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x
        backout_layer = nl // 2
        x_backout = None
        for i in range(nl):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # --- attention ---
            xn = res_norm(x)
            q = (xn @ self.c_q[i].to(x.dtype).mT).view(B, T, nh, hd)
            k = (xn @ self.c_k[i].to(x.dtype).mT).view(B, T, nkv, hd)
            v = (xn @ self.c_v[i].to(x.dtype).mT).view(B, T, nkv, hd)
            j = self.ve_index[i]
            if j >= 0:
                ve = F.embedding(idx, self.value_embeds[j]).view(B, T, nkv, hd).to(x.dtype)
                g = 3 * torch.sigmoid(xn[..., :self.ve_gate_channels] @ self.ve_gate[j].to(x.dtype).mT)
                v = v + g.unsqueeze(-1) * ve
            # Rotary embeddings (relative positional encoding)
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
            k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
            # QK norm
            q, k = qk_norm(q), qk_norm(k)
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
            x = x + y.contiguous().view(B, T, -1) @ self.attn_proj[i].to(x.dtype).mT
            # --- MLP (relu^2) ---
            x = x + F.relu(res_norm(x) @ self.mlp_fc[i].to(x.dtype).mT).square() @ self.mlp_proj[i].to(x.dtype).mT
            if i == backout_layer:
                x_backout = x
        kv_cache.advance(T) # all layers have written their KV for these positions
        # Subtract mid-layer residual to remove low-level features before final norm
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = res_norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        logits = x @ self.lm_head.to(x.dtype).mT
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

# -----------------------------------------------------------------------------
# Helper for the bf16-live-params variant of the optimizers
# -----------------------------------------------------------------------------

def cast_model_bf16(model) -> None:
    """Cast all parameters to bf16 in place (Parameter objects keep identity, so
    optimizer/param-group references and later graph captures stay valid).
    Rotary cos/sin buffers are already COMPUTE_DTYPE (bf16 on CUDA)."""
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model.to(torch.bfloat16)
