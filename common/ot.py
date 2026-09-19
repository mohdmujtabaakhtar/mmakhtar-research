"""Entropic optimal transport (Sinkhorn) and a gradient-reversal layer."""
from __future__ import annotations

import torch


@torch.no_grad()
def sinkhorn(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
             epsilon: float = 0.05, iters: int = 50) -> torch.Tensor:
    """Entropy-regularised OT plan for a cost matrix (n, m).

    Solves min_P <P, C> + eps * sum P (log P - 1) s.t. P 1 = a, P^T 1 = b,
    in the log domain for numerical stability. The cost is rescaled by its
    maximum so ``epsilon`` is scale-free. Returns P with shape (n, m).
    """
    c = cost / cost.max().clamp_min(1e-8)
    log_k = -c / epsilon
    log_a, log_b = a.clamp_min(1e-12).log(), b.clamp_min(1e-12).log()
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    return torch.exp(log_k + u[:, None] + v[None, :])


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def grad_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass; multiplies gradients by ``-coeff`` backwards."""
    return _GradReverse.apply(x, coeff)
