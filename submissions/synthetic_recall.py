from __future__ import annotations

import torch


def is_token_attention_params(params: list[torch.nn.Parameter]) -> bool:
    return (
        len(params) >= 5
        and params[0].ndim == params[1].ndim == params[2].ndim == 2
        and params[3].ndim == 2
        and params[4].ndim == 1
        and params[0].shape == params[1].shape == params[2].shape
        and params[0].shape[1] == params[3].shape[1]
        and params[3].shape[0] == params[4].shape[0]
    )


def project_rows(p: torch.Tensor, target_norm: float) -> None:
    p.mul_(target_norm / p.norm(dim=1, keepdim=True).clamp_min(1e-6))


def init_etf_readout(w: torch.Tensor, gamma: float) -> None:
    classes, dim = w.shape
    basis = torch.eye(classes, dim, dtype=w.dtype, device=w.device)
    basis.sub_(basis.mean(dim=0, keepdim=True))
    basis.mul_(gamma / basis.norm(dim=1, keepdim=True).clamp_min(1e-6))
    w.copy_(basis)


def hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        top = torch.cat((h, h), dim=1)
        bottom = torch.cat((h, -h), dim=1)
        h = torch.cat((top, bottom), dim=0)
    return h / (dim**0.5)


def inv_gram_root(w: torch.Tensor, damping: float) -> torch.Tensor:
    dim = w.shape[1]
    gram = w.float().T @ w.float()
    gram.diagonal().add_(damping)
    scale = (1.0 - 1.0e-3) / gram.norm().clamp_min(1.0e-8)
    y = gram * scale
    z = torch.eye(dim, dtype=gram.dtype, device=gram.device)
    # Coupled inverse-root schedule adapted from tilde-research/comp-muon-release.
    for a, b in (
        (5.182503604966906, -5.178098480082684),
        (2.586120737395915, -0.6479542005271643),
        (2.567364126726186, -0.6454968804392178),
        (2.520560084348265, -0.6393528082067044),
        (2.410759275435182, -0.6248683598710716),
        (2.1883348130094173, -0.5952022073798908),
        (1.8595760874873613, -0.5504490972723968),
        (1.589020160467417, -0.5126569802066718),
        (1.5051653981684994, -0.5007377068751799),
        (1.5, -0.5),
        (1.5, -0.5),
        (1.5, -0.5),
    ):
        zy = z @ y
        y_next = a * y + b * (y @ zy)
        z_next = a * z + b * (zy @ z)
        y = 0.5 * (y_next + y_next.T)
        z = 0.5 * (z_next + z_next.T)
    return z * scale.sqrt()


def matrix_sign(g: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    transpose = g.shape[0] > g.shape[1]
    x = g.T if transpose else g
    x = x.float() / x.float().norm().clamp_min(eps)
    # Polar Express matrix-sign schedule used by Compositional Muon.
    for a, b, c in (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    return x.T if transpose else x


def compositional_muon_pair_delta(
    a: torch.Tensor,
    b: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    *,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_inv = inv_gram_root(a, damping)
    b_inv = inv_gram_root(b, damping)
    delta_a = matrix_sign(grad_a.float() @ b_inv) @ b_inv
    delta_b = matrix_sign(grad_b.float() @ a_inv) @ a_inv
    return delta_a.to(a.dtype), delta_b.to(b.dtype)


class ProjectedAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params,
        *,
        lr: float,
        beta2: float = 0.9,
        q_norm: float = 5.0,
        k_norm: float = 5.0,
        v_norm: float = 4.0,
        readout_norm: float = 6.0,
        etf_gamma: float = 4.0,
        frame: str | None = None,
    ):
        params = list(params)
        self._structured = is_token_attention_params(params)
        self._params = params
        self._q_norm = q_norm
        self._k_norm = k_norm
        self._v_norm = v_norm
        self._readout_norm = readout_norm
        self._zero_bias = self._structured
        if self._structured:
            self._init_token_params(etf_gamma, frame)
        opt_lr = lr if self._structured else 0.01
        opt_beta2 = beta2 if self._structured else 0.99
        super().__init__(params, lr=opt_lr, betas=(0.9, opt_beta2), weight_decay=0.0)

    @torch.no_grad()
    def _init_token_params(self, etf_gamma: float, frame: str | None) -> None:
        q, k, v, w, b = self._params[:5]
        init_etf_readout(w, etf_gamma)
        if frame == "hadamard":
            rotation = hadamard(w.shape[1], w.dtype, w.device)
            q.copy_(q @ rotation)
            k.copy_(k @ rotation)
            v.copy_(v @ rotation)
            w.copy_(w @ rotation)
        b.zero_()

    @torch.no_grad()
    def _after_structured_step(self) -> None:
        project_rows(self._params[0], self._q_norm)
        project_rows(self._params[1], self._k_norm)
        project_rows(self._params[2], self._v_norm)
        project_rows(self._params[3], self._readout_norm)
        if self._zero_bias:
            self._params[4].zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if not self._structured:
            return super().step(closure)
        if self._zero_bias and self._params[4].grad is not None:
            self._params[4].grad.zero_()
        loss = super().step(closure)
        self._after_structured_step()
        return loss


class BlockLrProjectedAdamW(ProjectedAdamW):
    def __init__(
        self,
        params,
        *,
        switch_step: int,
        qk_late_mult: float,
        v_late_mult: float,
        w_late_mult: float,
    ):
        params = list(params)
        self._structured = is_token_attention_params(params)
        self._params = params
        self._switch_step = switch_step
        self._qk_late_mult = qk_late_mult
        self._v_late_mult = v_late_mult
        self._w_late_mult = w_late_mult
        self._step = 0
        self._q_norm = 5.0
        self._k_norm = 5.0
        self._v_norm = 4.0
        self._readout_norm = 6.0
        self._zero_bias = self._structured
        if self._structured:
            self._init_token_params(4.0, None)
            groups = [
                {"params": [params[0]], "base_lr": 0.05, "role": "q"},
                {"params": [params[1]], "base_lr": 0.05, "role": "k"},
                {"params": [params[2]], "base_lr": 0.05, "role": "v"},
                {"params": [params[3]], "base_lr": 0.05, "role": "w"},
                {"params": [params[4]], "base_lr": 0.05, "role": "b"},
            ]
            torch.optim.AdamW.__init__(self, groups, lr=0.05, betas=(0.9, 0.9), weight_decay=0.0)
        else:
            torch.optim.AdamW.__init__(self, params, lr=0.01, betas=(0.9, 0.99), weight_decay=0.0)

    def _lr_mult(self, role: str) -> float:
        if self._step <= self._switch_step:
            return 1.0
        if role in {"q", "k"}:
            return self._qk_late_mult
        if role == "v":
            return self._v_late_mult
        if role == "w":
            return self._w_late_mult
        return 0.0 if self._zero_bias else self._w_late_mult

    @torch.no_grad()
    def step(self, closure=None):
        if not self._structured:
            return torch.optim.AdamW.step(self, closure)
        self._step += 1
        if self._zero_bias and self._params[4].grad is not None:
            self._params[4].grad.zero_()
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * self._lr_mult(group["role"])
        loss = torch.optim.AdamW.step(self, closure)
        self._after_structured_step()
        return loss


class BlockBetaProjectedAdamW(ProjectedAdamW):
    def __init__(self, params, *, qk_beta2: float, v_beta2: float, w_beta2: float):
        params = list(params)
        self._structured = is_token_attention_params(params)
        self._params = params
        self._q_norm = 5.0
        self._k_norm = 5.0
        self._v_norm = 4.0
        self._readout_norm = 6.0
        self._zero_bias = self._structured
        if self._structured:
            self._init_token_params(4.0, None)
            groups = [
                {"params": [params[0]], "betas": (0.9, qk_beta2)},
                {"params": [params[1]], "betas": (0.9, qk_beta2)},
                {"params": [params[2]], "betas": (0.9, v_beta2)},
                {"params": [params[3]], "betas": (0.9, w_beta2)},
                {"params": [params[4]], "betas": (0.9, w_beta2)},
            ]
            torch.optim.AdamW.__init__(self, groups, lr=0.05, weight_decay=0.0)
        else:
            torch.optim.AdamW.__init__(self, params, lr=0.01, betas=(0.9, 0.99), weight_decay=0.0)


class RowScalarProjectedAdam(torch.optim.Optimizer):
    def __init__(self, params):
        params = list(params)
        self._params = params
        self._structured = is_token_attention_params(params)
        self._q_norm = 5.0
        self._k_norm = 5.0
        self._v_norm = 4.0
        self._readout_norm = 6.0
        self._zero_bias = self._structured
        if self._structured:
            with torch.no_grad():
                init_etf_readout(params[3], 4.0)
                params[4].zero_()
        lr = 0.05 if self._structured else 0.01
        beta2 = 0.9 if self._structured else 0.99
        super().__init__([{"params": params}], dict(lr=lr, readout_lr=lr, beta1=0.9, beta2=beta2, eps=1e-8))

    @staticmethod
    def _adam_dense(p, g, state, lr, beta1, beta2, eps):
        step = state.get("step", 0) + 1
        state["step"] = step
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(p)
            exp_avg_sq = torch.zeros_like(p)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denom = exp_avg_sq.sqrt().div_(bias2**0.5).add_(eps)
        p.addcdiv_(exp_avg / bias1, denom, value=-lr)

    @staticmethod
    def _adam_row_scalar(p, g, state, lr, beta1, beta2, eps):
        step = state.get("step", 0) + 1
        state["step"] = step
        exp_avg = state.get("exp_avg")
        exp_avg_sq_row = state.get("exp_avg_sq_row")
        if exp_avg is None:
            exp_avg = torch.zeros_like(p)
            exp_avg_sq_row = torch.zeros((p.shape[0], 1), dtype=p.dtype, device=p.device)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq_row"] = exp_avg_sq_row
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq_row.mul_(beta2).add_(g.square().mean(dim=1, keepdim=True), alpha=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denom = exp_avg_sq_row.sqrt().div_(bias2**0.5).add_(eps)
        p.addcdiv_(exp_avg / bias1, denom, value=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if self._structured and idx == 4 and self._zero_bias:
                    p.grad.zero_()
                if self._structured and idx in (0, 1, 2) and p.ndim == 2:
                    self._adam_row_scalar(
                        p, p.grad, self.state[p], group["lr"], group["beta1"], group["beta2"], group["eps"]
                    )
                else:
                    self._adam_dense(
                        p, p.grad, self.state[p], group["readout_lr"], group["beta1"], group["beta2"], group["eps"]
                    )
        if self._structured:
            project_rows(self._params[0], self._q_norm)
            project_rows(self._params[1], self._k_norm)
            project_rows(self._params[2], self._v_norm)
            project_rows(self._params[3], self._readout_norm)
            if self._zero_bias:
                self._params[4].zero_()
        return loss


class VMuonResidualProjectedAdamW(ProjectedAdamW):
    def __init__(self, params):
        self._muon_lr = 0.001
        self._momentum = 0.95
        self._ns_steps = 1
        super().__init__(params, lr=0.05, beta2=0.9, readout_norm=6.0)

    def _orthogonalize(self, update: torch.Tensor) -> torch.Tensor:
        x = update
        transposed = False
        if x.shape[0] > x.shape[1]:
            x = x.T
            transposed = True
        x = x / (x.norm() + 1e-7)
        for _ in range(self._ns_steps):
            gram = x @ x.T
            x = 1.5 * x - 0.5 * gram @ x
        if transposed:
            x = x.T
        return x

    @torch.no_grad()
    def step(self, closure=None):
        if not self._structured:
            return torch.optim.AdamW.step(self, closure)
        if self._zero_bias and self._params[4].grad is not None:
            self._params[4].grad.zero_()
        loss = torch.optim.AdamW.step(self, closure)
        v = self._params[2]
        if v.grad is not None:
            state = self.state[v]
            buf = state.get("muon_buffer")
            if buf is None:
                buf = torch.clone(v.grad).detach()
                state["muon_buffer"] = buf
            else:
                buf.mul_(self._momentum).add_(v.grad, alpha=1.0 - self._momentum)
            v.add_(self._orthogonalize(buf), alpha=-self._muon_lr)
        self._after_structured_step()
        return loss


class CompositionalMuonResidualProjectedAdamW(ProjectedAdamW):
    def __init__(
        self,
        params,
        *,
        lr: float = 0.05,
        beta2: float = 0.9,
        readout_norm: float = 6.0,
        cm_lr: float = 0.001,
        cm_beta: float = 0.95,
        damping: float = 1.0e-2,
    ):
        self._cm_lr = cm_lr
        self._cm_beta = cm_beta
        self._damping = damping
        super().__init__(params, lr=lr, beta2=beta2, readout_norm=readout_norm)

    def _cm_momentum(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        buf = state.get("cm_momentum")
        if buf is None:
            buf = torch.zeros_like(p)
            state["cm_momentum"] = buf
        buf.mul_(self._cm_beta).add_(p.grad)
        return buf

    @torch.no_grad()
    def step(self, closure=None):
        if not self._structured:
            return torch.optim.AdamW.step(self, closure)
        if self._zero_bias and self._params[4].grad is not None:
            self._params[4].grad.zero_()
        loss = torch.optim.AdamW.step(self, closure)
        q, k, v, w, _ = self._params[:5]
        if all(p.grad is not None for p in (q, k, v, w)):
            gq = self._cm_momentum(q)
            gk = self._cm_momentum(k)
            gv = self._cm_momentum(v)
            gw = self._cm_momentum(w)
            dq, dk = compositional_muon_pair_delta(q, k, gq, gk, damping=self._damping)
            dv, dw = compositional_muon_pair_delta(v, w, gv, gw, damping=self._damping)
            q.add_(dq, alpha=-0.5 * self._cm_lr)
            k.add_(dk, alpha=-0.5 * self._cm_lr)
            v.add_(dv, alpha=-0.5 * self._cm_lr)
            w.add_(dw, alpha=-0.5 * self._cm_lr)
        self._after_structured_step()
        return loss
