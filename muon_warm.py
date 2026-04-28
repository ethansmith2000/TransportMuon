import torch
from muon import adam_update


MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)
ROW_NS_MAX_SINGULAR = 1.25
MUON_WARM_ANCHOR_RETRACT_STEPS = 2


def _prepare_muon_matrix(G: torch.Tensor, eps: float = 1e-7):
    assert G.ndim >= 2
    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    return X, transposed


def _restore_muon_matrix(X: torch.Tensor, transposed: bool):
    return X.mT if transposed else X

@torch.compile
def _muon_ns5_prepared(X: torch.Tensor, steps: int):
    a, b, c = MUON_NS_COEFFS
    out = X
    for _ in range(steps):
        gram = out @ out.mT
        gram2 = gram @ gram
        out = a * out + (b * gram + c * gram2) @ out
    return out

def _row_ns_higham_cubic(X, num_iters: int):
    Z = X
    for _ in range(num_iters):
        gram = Z @ Z.mT
        gram_sq = gram @ gram
        correction = -10.0 * gram + 3.0 * gram_sq
        Z = 0.125 * (15.0 * Z + correction @ Z)
    return Z


def _row_ns_higham_cubic_w_cap(X, num_iters: int):
    Z = X

    gram = Z @ Z.mT

    upper_sq = gram.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    scale = (ROW_NS_MAX_SINGULAR / (upper_sq.sqrt() + 1e-7)).clamp(max=1.0)
    Z = Z * scale
    gram = gram * scale.square()

    gram_sq = gram @ gram
    correction = -10.0 * gram + 3.0 * gram_sq
    Z = 0.125 * (15.0 * Z + correction @ Z)

    for _ in range(num_iters - 1):
        gram = Z @ Z.mT
        gram_sq = gram @ gram
        correction = -10.0 * gram + 3.0 * gram_sq
        Z = 0.125 * (15.0 * Z + correction @ Z)
    return Z

def lower_bound_1(G: torch.Tensor):
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    off_abs_sum = G.abs().sum(dim=-1) - diag.abs()
    lower_sq = (diag - off_abs_sum).amin(dim=-1)
    lower = lower_sq.clamp_min(0).sqrt()
    return lower

def _row_ns_quadratic_w_cap(X, num_iters: int):
    Z = X

    gram = Z @ Z.mT

    upper_sq = gram.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    scale = (ROW_NS_MAX_SINGULAR / (upper_sq.sqrt() + 1e-7)).clamp(max=1.0)
    Z = Z * scale
    gram = gram * scale.square()

    Z = 1.5 * Z - 0.5 * (gram @ Z)

    for _ in range(num_iters - 1):
        gram = Z @ Z.mT
        Z = 1.5 * Z - 0.5 * (gram @ Z)
    return Z

def _row_ns_quadratic(X, num_iters: int):
    Z = X
    for _ in range(num_iters):
        gram = Z @ Z.mT
        Z = 1.5 * Z - 0.5 * (gram @ Z)
    return Z


def _row_ns_retract(X, num_iters: int, ns_method: str):
    if ns_method == "higham_cubic":
        return _row_ns_higham_cubic_w_cap(X, num_iters)
    if ns_method == "quadratic":
        return _row_ns_quadratic_w_cap(X, num_iters)
    raise ValueError(f"Unknown muon_warm_retract_method: {ns_method!r}")

# @torch.compile
def _jacobi_scale(R: torch.Tensor, jacobi_eps: float):
    diag = torch.diagonal(R)
    denom = diag.unsqueeze(1) + diag.unsqueeze(0)
    floor = jacobi_eps * (diag.abs().mean() + 1e-8)
    denom_abs = denom.abs().clamp_min(floor)
    denom = torch.copysign(denom_abs, denom)
    return denom.reciprocal()

@torch.compile
def _warm_polar_jacobi_step(
    work_matrix,
    q_prev,
    eta: float,
    jacobi_eps: float,
    retract_method: str,
    retract_steps: int,
):
    # Muon prepares matrices with rows <= columns, so Q has near-orthonormal rows.
    # Polar alignment asks for Q @ M.T to be symmetric.
    R = q_prev @ work_matrix.mT
    skew = 0.5 * (R - R.mT)
    # TODO: If we experiment with a raw Euclidean update for Q, project out its
    # radial component before retraction. This Jacobi-skew update already has
    # tangent form A @ Q with A^T = -A, so there is no radial component to remove.
    A = -2.0 * eta * skew * _jacobi_scale(R, jacobi_eps)
    q_tilde = q_prev + A @ q_prev
    return _row_ns_retract(q_tilde, retract_steps, retract_method)


class MuonWarm(torch.optim.Optimizer):
    """
    Muon optimizer with auxiliary Adam for non-matrix params.
    
    Supports double momentum mode (inspired by AdEMAMix) for muon params:
    - muon_double_momentum_mode=None: Standard single momentum
    - muon_double_momentum_mode="pre_ns": Blend fast/slow momentum before Newton-Schulz
    - muon_double_momentum_mode="post_ns": Apply NS to both, blend after
    
    Args:
        params: Model parameters or param_groups (use get_muon_param_groups helper)
        adam_lr: Learning rate for Adam params (default: 3e-4)
        adam_betas: Beta coefficients for Adam (default: (0.9, 0.95))
        adam_eps: Epsilon for Adam (default: 1e-10)
        adam_weight_decay: Weight decay for Adam params (default: 0)
        muon_lr: Learning rate for Muon params (default: 0.02)
        muon_momentum: Fast momentum beta for Muon (default: 0.95)
        muon_weight_decay: Weight decay for Muon params (default: 0)
        muon_ns_steps: Newton-Schulz iteration steps (default: 5)
        muon_nesterov: Use Nesterov momentum (default: True)
    """
    def __init__(
        self,
        params,
        # Adam params
        adam_lr=3e-4,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-10,
        adam_weight_decay=0,
        # Muon params
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_weight_decay=0,
        muon_ns_steps=5,
        muon_nesterov=True,
        muon_warm_anchor_every=8,
        muon_warm_lr=1.0,
        muon_warm_jacobi_eps=1e-3,
        muon_warm_retract_method="higham_cubic",
        muon_warm_retract_steps=1,
        muon_warm_full_ns_steps=2500,
    ):
        # Store muon-specific settings
        self.muon_ns_steps = muon_ns_steps
        self.muon_nesterov = muon_nesterov
        self.muon_warm_anchor_every = muon_warm_anchor_every
        self.muon_warm_lr = muon_warm_lr
        self.muon_warm_jacobi_eps = muon_warm_jacobi_eps
        self.muon_warm_retract_method = muon_warm_retract_method
        self.muon_warm_retract_steps = muon_warm_retract_steps
        self.muon_warm_full_ns_steps = max(0, int(muon_warm_full_ns_steps))
        if self.muon_warm_retract_method not in ("higham_cubic", "quadratic"):
            raise ValueError(
                "muon_warm_retract_method must be 'higham_cubic' or 'quadratic', "
                f"got {self.muon_warm_retract_method!r}"
            )
        
        # Convert params to param_groups if needed
        if isinstance(params, dict):
            param_groups = [params]
        elif hasattr(params, '__iter__') and not isinstance(params, torch.Tensor):
            param_groups = list(params)
            if len(param_groups) > 0 and not isinstance(param_groups[0], dict):
                # It's an iterable of tensors, not param groups
                param_groups = [{"params": param_groups}]
        else:
            param_groups = [{"params": [params]}]
        
        # Apply defaults to each group
        for group in param_groups:
            if "use_muon" not in group:
                # Default: use muon for 2D+, adam for 1D
                # This shouldn't happen if using get_muon_param_groups
                group["use_muon"] = True
            
            if group["use_muon"]:
                group.setdefault("lr", muon_lr)
                group.setdefault("momentum", muon_momentum)
                group.setdefault("weight_decay", muon_weight_decay)
            else:
                group.setdefault("lr", adam_lr)
                group.setdefault("betas", adam_betas)
                group.setdefault("eps", adam_eps)
                group.setdefault("weight_decay", adam_weight_decay)
        
        super().__init__(param_groups, dict())

    def _muon_update_warm(self, grad, state, group):
        state["momentum_fast"].lerp_(grad, 1 - group["momentum"])
        update = (
            torch.lerp(grad, state["momentum_fast"], group["momentum"])
            if self.muon_nesterov
            else state["momentum_fast"]
        )
        original_shape = update.shape
        if update.ndim == 4:
            update = update.view(len(update), -1)

        work_matrix, transposed = _prepare_muon_matrix(update)
        retract_steps = max(1, int(self.muon_warm_retract_steps))
        q_prev = state.get("muon_warm_q")
        needs_anchor = (
            q_prev is None
            or q_prev.shape != work_matrix.shape
            or q_prev.device != work_matrix.device
            or q_prev.dtype != work_matrix.dtype
            or state["step"] <= self.muon_warm_full_ns_steps
            or (
                self.muon_warm_anchor_every > 0
                and state["step"] % self.muon_warm_anchor_every == 0
            )
        )
        if needs_anchor:
            q_next = _muon_ns5_prepared(work_matrix, self.muon_ns_steps)
            q_next = _row_ns_retract(
                q_next,
                MUON_WARM_ANCHOR_RETRACT_STEPS,
                self.muon_warm_retract_method,
            )
        else:
            q_next = _warm_polar_jacobi_step(
                work_matrix,
                q_prev,
                eta=float(self.muon_warm_lr),
                jacobi_eps=float(self.muon_warm_jacobi_eps),
                retract_method=self.muon_warm_retract_method,
                retract_steps=retract_steps,
            )
        state["muon_warm_q"] = q_next.contiguous()

        update = _restore_muon_matrix(q_next, transposed)
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
        return update.reshape(original_shape) if update.shape != original_shape else update

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state["momentum_fast"] = torch.zeros_like(p)
                        state["step"] = 0
                    
                    state["step"] += 1
                    
                    update = self._muon_update_warm(p.grad, state, group)
                    
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss