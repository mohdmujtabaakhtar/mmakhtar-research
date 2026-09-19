"""Distribution divergences and correlation objectives used as alignment losses.

The fusion papers align two foundation-model branches by turning each branch's
features into a probability distribution over feature dimensions (a softmax over
the feature vector) and penalising a divergence between the two:

* Bhattacharyya distance (SATYAM)         -log sum_i sqrt(p_i q_i)
* Jensen-Shannon divergence (GARUDA)      KL(p || m)/2 + KL(q || m)/2, m = (p + q)/2
* Chernoff distance (COFFE)               -log sum_i p_i^s q_i^(1-s)
* Renyi divergence (RENO)                 1/(beta-1) log sum_i (p_i+delta)^beta (q_i+delta)^(1-beta)

``cca_objective`` (TRIO) is a correlation to *maximise*, not a distance.
Every function returns the batch mean.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def to_distribution(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over the last (feature) dimension."""
    return torch.softmax(x / temperature, dim=-1)


def bhattacharyya(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    bc = (p.clamp_min(EPS) * q.clamp_min(EPS)).sqrt().sum(-1)
    return -bc.clamp_min(EPS).log().mean()


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)   # noqa: E731
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


def chernoff(p: torch.Tensor, q: torch.Tensor, s: float = 0.3) -> torch.Tensor:
    coeff = (p.clamp_min(EPS).pow(s) * q.clamp_min(EPS).pow(1 - s)).sum(-1)
    return -coeff.clamp_min(EPS).log().mean()


def renyi(p: torch.Tensor, q: torch.Tensor, beta: float = 2.0, delta: float = 0.2) -> torch.Tensor:
    if beta == 1.0:
        raise ValueError("beta must differ from 1 (beta -> 1 recovers the KL divergence)")
    total = ((p + delta).pow(beta) * (q + delta).pow(1 - beta)).sum(-1)
    return (total.clamp_min(EPS).log() / (beta - 1)).mean()


def _inv_sqrt(mat: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(mat)
    return vecs @ torch.diag(vals.clamp_min(EPS).rsqrt()) @ vecs.T


def cca_objective(x: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    """tr( S_xx^{-1/2} S_xy S_yy^{-1/2} ) for a batch x (N, d), y (N, d).

    Covariances are ridge-regularised with ``reg * I`` so that the objective is well
    defined when the batch is smaller than the feature size.
    """
    n = x.shape[0]
    x, y = x - x.mean(0), y - y.mean(0)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    sxx = x.T @ x / (n - 1) + reg * eye
    syy = y.T @ y / (n - 1) + reg * eye
    sxy = x.T @ y / (n - 1)
    return torch.trace(_inv_sqrt(sxx) @ sxy @ _inv_sqrt(syy))
