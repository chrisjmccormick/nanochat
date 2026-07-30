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


def compute_window_sizes(config):
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
    long_window = config.sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    window_sizes[-1] = (long_window, 0)  # final layer always gets full context
    return window_sizes


@dataclass
class Dims:
    """Every axis in the model, named. Built once from the config; the allocation
    block in GPT.__init__ spells each shape with these names rather than
    recomputing products inline.

    This is the single source of truth for "which axis is this?", which is what
    the optimizer's shard axis and NorMuon reduction axis key off (see
    train_step.init_optimizer_state). Those used to be re-derived from the
    tensor's own proportions -- `red_dim` returned -1 or -2 based on
    `shape[-2] >= shape[-1]` -- which agreed with the intended axis by luck
    until d24, where ve_gate's (n_kv_head, gate_ch) flips from wide to square
    to tall as depth grows. Naming the axis at creation removes the guess.
    """
    layer: int        # transformer layers
    d_model: int      # residual stream width (config.n_embd)
    n_head: int       # query heads
    n_kv_head: int    # key/value heads (GQA)
    head_dim: int     # width of one head
    q: int            # n_head * head_dim     -- fused query width
    kv: int           # n_kv_head * head_dim  -- fused key/value width
    mlp: int          # 4 * d_model           -- MLP hidden width
    vocab: int        # PADDED vocab (what every tensor is actually sized to)
    vocab_used: int   # real vocab; rows in [vocab_used, vocab) never get gradient
    ve_slot: int      # value-embedding bank slots (one per VE layer)
    gate_ch: int      # channels the VE gate reads off the normed stream
    smear_ch: int     # channels the smear gate reads off the embedding

    @staticmethod
    def from_config(config, pad_vocab_size_to=64):
        head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        # Pad vocab for efficiency (DDP, tensor cores). Outputs are cropped in forward().
        padded = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        return Dims(
            layer=config.n_layer,
            d_model=config.n_embd,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            head_dim=head_dim,
            q=config.n_head * head_dim,
            kv=config.n_kv_head * head_dim,
            mlp=4 * config.n_embd,
            vocab=padded,
            vocab_used=config.vocab_size,
            ve_slot=len([i for i in range(config.n_layer) if has_ve(i, config.n_layer)]),
            gate_ch=12,
            smear_ch=24,
        )


def scaling_param_counts(config, pad_vocab_size_to=64):
    """Parameter counts as pure config arithmetic -- no model, no allocation.

    base_train needs these BEFORE it can size anything (the training horizon,
    batch size and LR scaling all derive from the parameter count, and the d12
    reference point needs a count for a model that is never built). It used to
    get them by constructing a throwaway GPT on the meta device and summing
    `p.numel()`; the numbers were always this arithmetic wearing a costume.

    GPT.num_scaling_params() calls this and then asserts it against the real
    allocated tensors, so the two can't drift.

    Different papers use different conventions -- Kaplan et al. excluded
    embedding parameters, Chinchilla included all -- so each group is returned
    separately and the caller picks.
    Ref: https://arxiv.org/abs/2203.15556 (Chinchilla)
    Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al.)
    """
    d = Dims.from_config(config, pad_vocab_size_to)
    wte = d.vocab * d.d_model
    lm_head = d.vocab * d.d_model
    value_embeds = d.ve_slot * d.vocab * d.kv
    transformer_matrices = (
        d.layer * d.q * d.d_model            # c_q
        + d.layer * d.kv * d.d_model         # c_k
        + d.layer * d.kv * d.d_model         # c_v
        + d.layer * d.d_model * d.q          # attn_proj
        + d.layer * d.mlp * d.d_model        # mlp_fc
        + d.layer * d.d_model * d.mlp        # mlp_proj
        + d.ve_slot * d.n_kv_head * d.gate_ch  # ve_gate
    )
    scalars = (
        d.layer                # resid_lambdas
        + d.layer              # x0_lambdas
        + 1 * d.smear_ch       # smear_gate
        + 1                    # smear_lambda
        + 1                    # backout_lambda
    )
    return {
        'wte': wte,
        'value_embeds': value_embeds,
        'lm_head': lm_head,
        'transformer_matrices': transformer_matrices,
        'scalars': scalars,
        'total': wte + value_embeds + lm_head + transformer_matrices + scalars,
    }


def estimate_flops(config, pad_vocab_size_to=64):
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
    counts = scaling_param_counts(config, pad_vocab_size_to)
    # Only matmul weights count: embeddings are a lookup, scalars are elementwise.
    matmul_params = counts['transformer_matrices'] + counts['lm_head']
    h, q, t = config.n_head, config.n_embd // config.n_head, config.sequence_len
    attn_flops = 0
    for window, _right in compute_window_sizes(config):
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    return 6 * matmul_params + attn_flops


class GPT(nn.Module):
    def __init__(self, config, device=None, pad_vocab_size_to=64):
        """Allocate every tensor in the model, ON `device`, at its final dtype.

        There is no meta-device phase and no `.to(device)`: `device` is threaded
        into the allocation itself. The old three-step dance (build on meta ->
        to_empty(device) -> init_weights) existed only because nn.Module's
        default construction site is the CPU; it cost a host-RAM copy of the
        whole model, a PCIe transfer, and an unenforceable rule that __init__
        must not touch data. That rule got violated silently: backout_lambda and
        smear_gate were nominally initialized here, never ran under meta, and
        every tuned baseline actually trained from to_empty()'s zeroed storage.

        Contents are UNINITIALIZED. Fill them exactly once, with either:
          - init_weights()                       fresh run
          - checkpoint_manager.load_model_state  resume / inference

        Dtypes are the ones each tensor ends the setup at, so nothing is ever
        re-cast in place afterwards:
          - wte / value_embeds : COMPUTE_DTYPE. The two biggest tensors; the
            optimizer tolerates reduced-precision embeddings. (The parity tests
            monkeypatch COMPUTE_DTYPE to fp64/fp32 -- reading it here is what
            makes those tiers work.)
          - every other matmul weight : fp32 here, converted to the bf16-live +
            uint16-mantissa master pair by train_step.init_optimizer_state.
            Inference never calls that, so for inference these stay fp32 and
            load_model_state assigns whatever dtype the checkpoint holds.
          - the five scalars : fp32-live permanently, no mantissa pair (bf16
            rounding of the residual multipliers cost +0.016 val bpb).
        """
        super().__init__()
        self.config = config
        d = Dims.from_config(config, pad_vocab_size_to)
        self.dims = d
        self.pad_vocab_size_to = pad_vocab_size_to  # so the count helpers can re-derive Dims
        # Kept as attributes because every forward body reads them off the model.
        self.head_dim = d.head_dim
        self.padded_vocab_size = d.vocab
        self.ve_gate_channels = d.gate_ch
        if d.vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {d.vocab} for efficiency")
        # window_size is (left, right): (-1, 0) full context, (N, 0) sliding window
        self.window_sizes = compute_window_sizes(config)
        # Value embeddings (ResFormer-style) live on alternating layers, last always
        # included. Banked over just the VE layers; ve_index maps layer -> bank slot
        # (-1 = no VE on this layer) and is read by every forward body.
        self.ve_layers = [i for i in range(d.layer) if has_ve(i, d.layer)]
        self.ve_index = [self.ve_layers.index(i) if i in self.ve_layers else -1 for i in range(d.layer)]

        # fp16 is the one COMPUTE_DTYPE that cannot hold the embeddings: GradScaler
        # cannot unscale fp16 gradients, so they stay fp32 there. Read at call time,
        # not import time, so the parity tests' COMPUTE_DTYPE monkeypatch applies.
        embed_dtype = torch.float32 if COMPUTE_DTYPE == torch.float16 else COMPUTE_DTYPE

        # --- Parameters. All owned directly by this module (no submodules), so
        # each name below is also its checkpoint key. Banks stack the layer index
        # on dim 0; each slice keeps F.linear's (out_features, in_features)
        # convention and is consumed as `x @ w.mT`.
        #
        # Written out one tensor per line, deliberately: the shape, the dtype and
        # therefore the memory cost of every tensor in the model is readable in
        # one place, and the axis names say which dimension is which for the
        # sharding and reduction declarations in init_optimizer_state.

        # Token embedding and unembedding (untied). F.embedding / raw matmul.
        self.wte     = nn.Parameter(torch.empty(d.vocab, d.d_model, dtype=embed_dtype, device=device))
        self.lm_head = nn.Parameter(torch.empty(d.vocab, d.d_model, dtype=torch.float32, device=device))

        # Attention weight banks.
        self.c_q       = nn.Parameter(torch.empty(d.layer, d.q,       d.d_model, dtype=torch.float32, device=device))
        self.c_k       = nn.Parameter(torch.empty(d.layer, d.kv,      d.d_model, dtype=torch.float32, device=device))
        self.c_v       = nn.Parameter(torch.empty(d.layer, d.kv,      d.d_model, dtype=torch.float32, device=device))
        self.attn_proj = nn.Parameter(torch.empty(d.layer, d.d_model, d.q,       dtype=torch.float32, device=device))

        # MLP weight banks (relu^2, 4x expansion).
        self.mlp_fc    = nn.Parameter(torch.empty(d.layer, d.mlp,     d.d_model, dtype=torch.float32, device=device))
        self.mlp_proj  = nn.Parameter(torch.empty(d.layer, d.d_model, d.mlp,     dtype=torch.float32, device=device))

        # Value embeddings + their input-dependent gates, banked over VE slots.
        self.value_embeds = nn.Parameter(torch.empty(d.ve_slot, d.vocab,    d.kv,      dtype=embed_dtype, device=device))
        self.ve_gate      = nn.Parameter(torch.empty(d.ve_slot, d.n_kv_head, d.gate_ch, dtype=torch.float32, device=device))

        # Per-layer learnable scalars (modded-nanogpt style). Separate parameters
        # so each can take its own optimizer treatment.
        self.resid_lambdas  = nn.Parameter(torch.empty(d.layer, dtype=torch.float32, device=device))  # residual stream scale
        self.x0_lambdas     = nn.Parameter(torch.empty(d.layer, dtype=torch.float32, device=device))  # blend x0 back in
        self.smear_gate     = nn.Parameter(torch.empty(1, d.smear_ch, dtype=torch.float32, device=device))  # prev-token mix gate
        self.smear_lambda   = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))
        self.backout_lambda = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))

        # --- Buffers. Rotary embeddings are cheap, so over-compute generously:
        # with varlen training the whole micro-batch is one sequence
        # (T = batch_size * seq_len), and 64X covers batch sizes up to 64. The
        # assert in forward() catches it if we ever exceed.
        # These are computed for real right here. Under the old meta path they
        # were fake meta tensors that init_weights had to recompute, which is why
        # the inference loader called init_weights purely to get them.
        self.rotary_seq_len = config.sequence_len * 64
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, d.head_dim, device=device)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False => not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Fill every parameter __init__ allocated. Fresh-run path only; resume
        and inference call checkpoint_manager.load_model_state instead of this.

        wte (embedding):     normal, std=0.8
        lm_head:             normal, std=0.001
        c_q, c_k, c_v:       uniform, std=1/sqrt(d_model)   (whole bank)
        attn_proj:           zeros
        mlp_fc:              uniform, std=0.4/sqrt(d_model) (whole bank)
        mlp_proj:            zeros
        value_embeds:        uniform, std=1/sqrt(d_model)   (whole bank)
        ve_gate:             uniform in [0, 0.02] (slightly above neutral)
        resid_lambdas:       1.15 -> 1.05 linear decay over depth
        x0_lambdas:          0.20 -> 0.05 linear decay over depth
        smear_gate:          zeros
        smear_lambda:        zeros (smear disabled at init)
        backout_lambda:      zeros (backout disabled at init)

        DRAW ORDER: banks are drawn whole, in the order written below. The
        previous version drew matrix slices per-layer interleaved, and
        value_embeds/ve_gate in sorted-by-STRING layer order ('1','11','3',...),
        so that a given seed reproduced the pre-flattening and pre-banking
        models bit-for-bit. Those models no longer exist, so the contortion is
        gone. Consequence: a given seed now produces different weights than it
        did before this commit, and tuned baselines need re-running.
        """
        d = self.dims
        dev = self.wte.device

        # Embedding and unembedding. wte may be narrower than fp32 (COMPUTE_DTYPE),
        # so draw in fp32 and let copy_ round -- drawing straight into bf16 would
        # quantize the distribution rather than the samples.
        self.wte.copy_(torch.empty(d.vocab, d.d_model, dtype=torch.float32, device=dev).normal_(mean=0.0, std=0.8))
        torch.nn.init.normal_(self.lm_head, mean=0.0, std=0.001)

        # Matrix banks: uniform with bound = sqrt(3) * std, which gives Uniform the
        # same standard deviation as the equivalent Normal while avoiding outliers.
        s = 3**0.5 * d.d_model**-0.5
        torch.nn.init.uniform_(self.c_q, -s, s)
        torch.nn.init.uniform_(self.c_k, -s, s)
        torch.nn.init.uniform_(self.c_v, -s, s)
        torch.nn.init.zeros_(self.attn_proj)                      # projections start at zero
        torch.nn.init.uniform_(self.mlp_fc, -s * 0.4, s * 0.4)    # 0.4x init scale for c_fc
        torch.nn.init.zeros_(self.mlp_proj)

        # Value embeddings (same std as c_v) and their gates. value_embeds is
        # COMPUTE_DTYPE, so it takes the same fp32-draw-then-round path as wte.
        self.value_embeds.copy_(
            torch.empty(d.ve_slot, d.vocab, d.kv, dtype=torch.float32, device=dev).uniform_(-s, s))
        torch.nn.init.uniform_(self.ve_gate, 0.0, 0.02)           # start slightly above neutral

        # Per-layer scalars: linear decay over depth. Stronger residual and more
        # input-embedding blending at early layers, both tapering with depth.
        self.resid_lambdas.copy_(torch.linspace(1.15, 1.05, d.layer, dtype=torch.float32, device=dev))
        self.x0_lambdas.copy_(torch.linspace(0.20, 0.05, d.layer, dtype=torch.float32, device=dev))

        # Smear/backout start disabled. This matches what from-scratch runs have
        # always actually trained with: the pre-flattening __init__ nominally set
        # backout_lambda=0.2 and a kaiming smear_gate, but under meta those inits
        # never executed and to_empty() left zeroed storage, so every tuned
        # baseline started from zeros. Now it is explicit rather than luck.
        torch.nn.init.zeros_(self.smear_gate)
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.zeros_(self.backout_lambda)

        # Rotary embeddings are NOT touched here -- __init__ computed them for
        # real. (The inference loader used to call this whole function purely to
        # get them, because under meta they came out as fake tensors.)

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

    def get_device(self):
        return self.wte.device

    def matrix_parameters(self):
        """All matmul weight banks (Muon-updated). One entry per role — each is a
        single stacked Parameter, not a per-layer list."""
        return [self.c_q, self.c_k, self.c_v, self.attn_proj, self.mlp_fc, self.mlp_proj, self.ve_gate]

    def estimate_flops(self):
        """FLOPs per token (forward + backward) for this model's config. See the
        module-level estimate_flops() -- this is the bound-to-an-instance form."""
        return estimate_flops(self.config, self.pad_vocab_size_to)

    def num_scaling_params(self):
        """Parameter counts by group, for scaling-law analysis. Computed from the
        config by the module-level scaling_param_counts(), then checked against
        the tensors actually allocated -- that assert is the whole reason the
        config arithmetic is trustworthy enough for base_train to size a run
        before any model exists."""
        counts = scaling_param_counts(self.config, self.pad_vocab_size_to)
        allocated = sum(p.numel() for p in self.parameters())
        assert counts['total'] == allocated, \
            f"config param math says {counts['total']:,} but {allocated:,} were allocated"
        return counts

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
# Checkpoint state: the model half, written out one line per tensor.
# -----------------------------------------------------------------------------

def model_state(model):
    """The model's weights as a flat {name: tensor} dict -- the model half of a
    checkpoint, and the manifest that load_model_state fills.

    Written out rather than taken from nn.Module.state_dict(). The keys are
    IDENTICAL to what state_dict() produced (GPT owns every parameter directly,
    so its keys were already just these attribute names, and cos/sin are
    persistent=False buffers that state_dict excluded too), so the on-disk
    format is unchanged and existing checkpoints keep loading. What changes is
    that "which tensors get saved" is a statement in the source instead of a
    walk over whatever happens to be registered.

    Tensors come back by reference, not cloned -- torch.save materializes them.
    During training these are the bf16 LIVE halves of the masters; the matching
    uint16 mantissas ride in train_step.optim_state, and BOTH halves are needed
    to resume a run without losing the low bits.
    """
    return {
        "wte":            model.wte,
        "lm_head":        model.lm_head,
        "c_q":            model.c_q,
        "c_k":            model.c_k,
        "c_v":            model.c_v,
        "attn_proj":      model.attn_proj,
        "mlp_fc":         model.mlp_fc,
        "mlp_proj":       model.mlp_proj,
        "value_embeds":   model.value_embeds,
        "ve_gate":        model.ve_gate,
        "resid_lambdas":  model.resid_lambdas,
        "x0_lambdas":     model.x0_lambdas,
        "smear_gate":     model.smear_gate,
        "smear_lambda":   model.smear_lambda,
        "backout_lambda": model.backout_lambda,
    }


@torch.no_grad()
def load_model_state(model, state):
    """Fill the allocated parameters from a checkpoint dict.

    ASSIGNS rather than copies, which is what load_state_dict(assign=True) did:
    the checkpoint's dtype wins. That is how an inference load pulls bf16 live
    weights into parameters that __init__ allocated fp32, with no separate cast
    pass -- and how a training resume gets bf16 live halves that
    init_optimizer_state then re-splits against the saved mantissas.

    Every key must be present and every shape must match (the old strict=True).
    Extra keys in `state` are reported rather than ignored, since silently
    dropping one is how a renamed parameter stops being restored.
    """
    dst = model_state(model)
    missing = [k for k in dst if k not in state]
    unexpected = [k for k in state if k not in dst]
    assert not missing, f"checkpoint is missing parameters: {missing}"
    assert not unexpected, f"checkpoint has unknown parameters: {unexpected}"
    for name, p in dst.items():
        loaded = state[name]
        assert tuple(loaded.shape) == tuple(p.shape), \
            f"{name}: checkpoint holds {tuple(loaded.shape)}, model allocated {tuple(p.shape)}"
        p.data = loaded.to(p.device)


# -----------------------------------------------------------------------------
# Helper for the bf16-live-params variant of the optimizers
# -----------------------------------------------------------------------------

def cast_model_bf16(model) -> None:
    """Cast all parameters to bf16 in place (Parameter objects keep identity, so
    optimizer/param-group references and later graph captures stay valid).
    Rotary cos/sin buffers are already COMPUTE_DTYPE (bf16 on CUDA)."""
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model.to(torch.bfloat16)
