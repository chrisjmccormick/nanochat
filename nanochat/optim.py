"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Four versions are provided: MuonAdamW/DistMuonAdamW (single GPU and distributed) and
Fp32MuonAdamW/Fp32DistMuonAdamW (fp32 master weights, bf16 live params).

These optimizers hold no hyperparameters. A run's learning rates, betas and weight
decays are pre-computed into per-step tables of update coefficients before training
starts (nanochat/schedules.py) and the training loop never touches them: every group
arrives fully specified from schedules.build_param_groups(), and each step() reads
the next row. See that module for what gets folded into what, and why.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.schedules import AdamWCoeffs, MuonCoeffs

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,          # (32768, 768) - parameter tensor
    grad: Tensor,       # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,    # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor, # (32768, 768) - second moment, same shape as p
    c: AdamWCoeffs,     # this group's per-step coefficient TABLES, device-resident
    i: Tensor,          # (1,) int64 device tensor - the schedule row to read
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> param_update.
    All in one compiled graph to eliminate Python overhead between ops.

    Both bias corrections and the LR schedule are already folded into `c` by
    nanochat/schedules.py, so no step count reaches this kernel and there is no
    `beta ** t` to evaluate. The tables live on the device and `i` is a device
    tensor, so each `c.<field>[i]` is a gather inside this graph: the step needs no
    host-to-device copy and nothing from the host at all, which is what makes the
    whole update CUDA-graph capturable. Indexing with a (1,)-shaped tensor (rather
    than a 0-D one) keeps it a real gather that broadcasts, instead of something
    dynamo tries to turn back into a Python scalar.
    """
    # Weight decay (decoupled, applied before the update): wd_mul = 1 - lr*wd
    p.mul_(c.wd_mul[i])
    # Update running averages (lerp_ is cleaner and fuses well).
    # The casts are load-bearing: the tables are fp32 but these moments follow the
    # parameter's dtype, and the embeddings are natively bf16. lerp_ takes a 0-D
    # weight through its scalar overload (which promotes freely) but a (1,) weight
    # through the Tensor overload, which REQUIRES the destination's dtype.
    exp_avg.lerp_(grad, c.one_minus_beta1[i].to(exp_avg.dtype))
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[i].to(exp_avg_sq.dtype))
    # Compute update and apply: rsqrt_bias2 = 1/sqrt(1-beta2^t), step_size = lr/(1-beta1^t)
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[i] + c.eps[i]
    p.sub_(c.step_size[i] * (exp_avg / denom))

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
    c: MuonCoeffs,                  # this group's per-step coefficient TABLES, device-resident
    i: Tensor,                      # (1,) int64 device tensor - the schedule row to read
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.

    The LR schedule, this group's sqrt(fan_out/fan_in) LR scaling, and the weight
    decay are already folded into `c` by nanochat/schedules.py. The tables are
    device-resident and `i` is a device tensor, so reading this step's row is a
    gather inside this graph — no host involvement, no per-step H2D copy.
    ns_steps/red_dim are Python ints and do specialize the graph.
    """
    dtype = stacked_grads.dtype

    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, c.one_minus_momentum[i].to(dtype))
    g = stacked_grads.lerp_(momentum_buffer, c.momentum[i].to(dtype))

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

    # Variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    # This weight belongs in the BUFFER's dtype (fp32), not the polar express's bf16.
    # The buffer and v_mean are both fp32; bf16 only ever reached this line because the
    # old kernel hoisted `beta2 = beta2_t.to(g.dtype)` up top alongside the genuinely
    # bf16 orthogonalization math and reused it here. That cost real precision, via
    # catastrophic cancellation: it rounded beta2 FIRST, so 1 - bf16(0.9) = 0.1015625,
    # 1.6% off. (Pre-computing 1-beta2 on the host and then rounding, as the tables do,
    # already narrowed that to 0.0996094 — 0.4% off. This gets it exact.)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype),
                                 c.one_minus_beta2[i].to(second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update: lr_wd = lr*wd
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(c.lr[i].to(g.dtype) * g + c.lr_wd[i].to(g.dtype) * stacked_params * mask)

# -----------------------------------------------------------------------------
# Shared base: the device-resident schedule index.

class _ScheduledOptimizer(torch.optim.Optimizer):
    """Common machinery for the four MuonAdamW variants.

    Param groups must come from schedules.build_param_groups() fully specified;
    there are no hyperparameter defaults to fall back on here. Each group carries
    `tabs`, its per-step coefficient tables, resident on the same device as its
    parameters. The kernels index them with `_sched_t`, so a step touches the host
    for nothing at all — no schedule arithmetic, no H2D scalar copies — and the
    whole update can be captured in a CUDA graph.

    Two step counters, deliberately:
      - `_sched_t`, a (1,) int64 DEVICE tensor, is what the kernels index with. It
        is incremented on device, so a captured graph advances the schedule itself
        on replay.
      - `_step_idx`, a host int, mirrors it for bounds checks, logging and
        checkpoints. Reading the device counter instead would sync every step.
    Keep them in step: everything that moves the schedule moves both.

    The schedule advances once per step(), so it counts updates actually applied
    (what you want if a GradScaler skips one). The position travels with
    checkpoints; the tables do not, since a resume rebuilds them from the run's
    arguments.
    """

    def __init__(self, param_groups: list[dict]):
        assert param_groups, "no parameter groups"
        for i, g in enumerate(param_groups):
            assert g.get("kind") in ("adamw", "muon"), f"group {i}: unknown kind {g.get('kind')!r}"
            assert "tabs" in g, (
                f"group {i} is missing its schedules — param groups must be built by "
                "nanochat.schedules.build_param_groups()")
        super().__init__(param_groups, defaults={})
        self._num_sched_steps = len(param_groups[0]["tabs"][0])
        for i, g in enumerate(param_groups):
            assert all(len(t) == self._num_sched_steps for t in g["tabs"]), \
                f"group {i}: schedule tables disagree on length"
            dev = g["params"][0].device
            assert all(t.device == dev for t in g["tabs"]), (
                f"group {i}: schedule tables are on {g['tabs'][0].device} but its parameters "
                f"are on {dev} — pass the right device to build_param_groups()")
        device = param_groups[0]["params"][0].device
        self._sched_t = torch.zeros(1, dtype=torch.int64, device=device)
        self._step_idx = 0

    # -- schedule position ----------------------------------------------------

    @property
    def schedule_step(self) -> int:
        """How many updates have been applied / which row the next step will read.
        Served from the host mirror, so reading it never syncs with the device."""
        return self._step_idx

    def set_schedule_step(self, i: int) -> None:
        """Reposition the schedule. Resuming mid-run: pass the resume step.
        Warm-starting into a fresh schedule (keeping the momentum buffers): pass 0."""
        assert 0 <= i <= self._num_sched_steps, f"step {i} outside the {self._num_sched_steps}-step schedule"
        self._step_idx = i
        self._sched_t.fill_(i)

    def _check_bounds(self) -> None:
        """Host-side, before the kernels run: an out-of-range gather on device would
        be a device-side assert (or worse, silently clamp) rather than this message."""
        assert self._step_idx < self._num_sched_steps, (
            f"training ran past the end of the {self._num_sched_steps}-step schedule; "
            "build the param groups with the run's true step count")

    def _advance(self) -> None:
        # every group read row _step_idx during this step, so advance once, at the end
        self._sched_t.add_(1)
        self._step_idx += 1

    # -- checkpointing --------------------------------------------------------
    # Keyed by (group_idx, param_idx) — stable across processes (unlike id()).
    # Schedule tables are deliberately not saved: they are a property of the run's
    # arguments, so a resume rebuilds them and only restores the position.

    def state_dict(self):
        param_states = {}
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                st = self.state.get(p)
                if st:
                    param_states[f"{gi}.{pi}"] = {
                        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in st.items()}
        return {"param_states": param_states, "schedule_step": self._step_idx}

    def load_state_dict(self, state_dict):
        """Dtype-preserving load (modded-nanogpt convention): each loaded tensor
        is cast to the dtype/device of the EXISTING state entry, so fp32
        momentum/master never silently degrade to the param dtype."""
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                saved = state_dict["param_states"].get(f"{gi}.{pi}")
                if saved is None:
                    continue
                st = self.state[p]
                for k, v in saved.items():
                    if isinstance(v, torch.Tensor) and k in st and isinstance(st[k], torch.Tensor):
                        st[k] = v.to(dtype=st[k].dtype, device=st[k].device)
                    else:
                        st[k] = v
        self.set_schedule_step(state_dict.get("schedule_step", 0))

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(_ScheduledOptimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: from nanochat.schedules.build_param_groups() — see that module.
    """

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, then call the fused kernel with this step's coefficients.
        """
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]

            # State init
            if not state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            # Fused update: weight_decay -> momentum -> param_update
            adamw_step_fused(p, p.grad, state["exp_avg"], state["exp_avg_sq"], group["tabs"], self._sched_t)

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, then call the fused kernel with this step's coefficients.
        """
        params: list[Tensor] = group["params"]
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (the group's params all share a shape, asserted at build time)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            group["tabs"], self._sched_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        self._check_bounds()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            else:
                self._step_muon(group)
        self._advance()

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(_ScheduledOptimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (enforced at build time).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: from nanochat.schedules.build_param_groups() — see that module.
    """

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)

            adamw_step_fused(p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'], group["tabs"], self._sched_t)

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                group["tabs"], self._sched_t, group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self._check_bounds()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            else:
                reduce_infos.append(self._reduce_muon(group, world_size))

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            else:
                self._compute_muon(group, info, gather_list, rank)

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
        self._advance()


# -----------------------------------------------------------------------------
# bf16 weights + 32-bit-state optimizer (fp32 master, in-place bf16 writeback)
# -----------------------------------------------------------------------------

class Fp32MuonAdamW(_ScheduledOptimizer):
    """MuonAdamW with fp32 optimizer state and fp32 master weights; bf16 params
    are updated by IN-PLACE copy from the masters, so captured CUDA graphs keep
    reading valid pointers. Single-GPU version.

    Construct while the model still holds the checkpoint's fp32 weights (only
    the embeddings are natively bf16); cast the model to bf16 AFTER."""

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups)
        self._snapshot_masters()

    def _snapshot_masters(self):
        """Eagerly snapshot fp32 masters for every param this rank owns. Called at
        construction, BEFORE the model is cast to bf16, so masters keep the
        checkpoint's full precision."""
        for group in self.param_groups:
            if group["kind"] == "adamw":
                for p in group["params"]:
                    st = self.state[p]
                    st["master"] = p.detach().float().clone()
                    st["exp_avg"] = torch.zeros_like(st["master"])
                    st["exp_avg_sq"] = torch.zeros_like(st["master"])
            else:
                params = group["params"]
                p = params[0]
                st = self.state[p]
                shape = p.shape
                st["master_stack"] = torch.stack([q.detach().float() for q in params])
                st["momentum_buffer"] = torch.zeros_like(st["master_stack"])
                state_shape = ((len(params), shape[-2], 1) if shape[-2] >= shape[-1]
                               else (len(params), 1, shape[-1]))
                st["second_momentum_buffer"] = torch.zeros(
                    state_shape, dtype=torch.float32, device=p.device)

    def _step_adamw(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            adamw_step_fused(state["master"], p.grad.float(),
                             state["exp_avg"], state["exp_avg_sq"], group["tabs"], self._sched_t)
            p.copy_(state["master"])   # bf16 writeback, same storage

    def _step_muon(self, group: dict) -> None:
        params = group["params"]
        if not params or params[0].grad is None:
            return
        state = self.state[params[0]]
        shape = params[0].shape
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params]).float()
        muon_step_fused(
            stacked_grads, state["master_stack"],
            state["momentum_buffer"], state["second_momentum_buffer"],
            group["tabs"], self._sched_t, group["ns_steps"], red_dim,
        )
        for p, m in zip(params, state["master_stack"].unbind(0)):
            p.copy_(m)                 # bf16 writeback, same storage

    @torch.no_grad()
    def step(self):
        self._check_bounds()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            else:
                self._step_muon(group)
        self._advance()


class Fp32DistMuonAdamW(_ScheduledOptimizer):
    """DistMuonAdamW (ZeRO-sharded) with fp32 state + fp32 sharded masters and
    in-place bf16 writeback. Comm runs in the param dtype (bf16 after cast).
    Same 3-phase async structure as DistMuonAdamW."""

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups)
        self._snapshot_masters()

    def _snapshot_masters(self):
        rank, world = dist.get_rank(), dist.get_world_size()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                for p in group["params"]:
                    st = self.state[p]
                    if p.numel() < 1024:
                        p_slice = p
                    else:
                        assert p.shape[0] % world == 0
                        rs = p.shape[0] // world
                        p_slice = p[rank * rs:(rank + 1) * rs]
                    st["master"] = p_slice.detach().float().clone()
                    st["exp_avg"] = torch.zeros_like(st["master"])
                    st["exp_avg_sq"] = torch.zeros_like(st["master"])
            else:
                params = group["params"]
                p = params[0]
                st = self.state[p]
                shape = p.shape
                chunk = (len(params) + world - 1) // world
                start = rank * chunk
                owned = [params[start + i] for i in range(min(chunk, max(0, len(params) - start)))]
                st["master_stack"] = (torch.stack([q.detach().float() for q in owned])
                                      if owned else torch.zeros(0, *shape, dtype=torch.float32, device=p.device))
                st["momentum_buffer"] = torch.zeros(chunk, *shape, dtype=torch.float32, device=p.device)
                state_shape = ((chunk, shape[-2], 1) if shape[-2] >= shape[-1]
                               else (chunk, 1, shape[-1]))
                st["second_momentum_buffer"] = torch.zeros(
                    state_shape, dtype=torch.float32, device=p.device)

    def _reduce_adamw(self, group, world_size):
        param_infos = {}
        for p in group["params"]:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG,
                                                    async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group, world_size):
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        grad_stack = torch.stack([q.grad for q in params])
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG,
                                            async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads,
                    chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group["params"]:
            pinfo = info["param_infos"][p]
            pinfo["future"].wait()
            state = self.state[p]
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            adamw_step_fused(state["master"], pinfo["grad_slice"].float(),
                             state["exp_avg"], state["exp_avg_sq"], group["tabs"], self._sched_t)
            p_slice.copy_(state["master"])
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        p = params[0]
        shape, device = p.shape, p.device
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=p.dtype, device=device)
        if num_owned > 0:
            muon_step_fused(
                info["grad_chunk"][:num_owned].float(), state["master_stack"],
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                group["tabs"], self._sched_t, group["ns_steps"], red_dim,
            )
            updated[:num_owned].copy_(state["master_stack"])   # fp32 -> bf16
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]                  # reuse buffer
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list):
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"],
                                     list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        self._check_bounds()
        reduce_infos = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            else:
                reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            else:
                self._compute_muon(group, info, gather_list, rank)
        self._finish_gathers(gather_list)
        self._advance()
