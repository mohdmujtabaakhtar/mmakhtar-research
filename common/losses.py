"""Loss functions shared across papers.

Reductions average over feature dimensions as well as the batch, so that each
term is on a comparable scale to a cross-entropy loss regardless of latent size.
"""
from __future__ import annotations

import torch


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, exp(logvar)) || N(0, I) ), averaged over dimensions and batch."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor, training: bool = True) -> torch.Tensor:
    """z = mu + exp(logvar / 2) * eps, eps ~ N(0, I). Returns the mean when not training."""
    if not training:
        return mu
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def l1_sparsity(*tensors: torch.Tensor) -> torch.Tensor:
    """Sum over tensors of the mean absolute value (a dimension-normalised L1 norm)."""
    return sum(t.abs().mean() for t in tensors)
