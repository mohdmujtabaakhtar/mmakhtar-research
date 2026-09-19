"""Non-Euclidean geometry utilities (Poincare ball and unit hypersphere).

Several papers in this repository (RHYME, MiCuNet, NOVA-ARC, SATYAM, COBALT,
ORBIT, PHOENIX-Mamba) project foundation-model embeddings into hyperbolic
and/or spherical spaces. These are the shared primitives they build on.
Curvature is parameterised by ``c > 0`` for the Poincare ball of radius 1/sqrt(c).
"""
from __future__ import annotations

import torch

EPS = 1e-5


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1 + EPS, 1 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project_to_ball(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Clip points so they stay strictly inside the Poincare ball."""
    max_norm = (1 - 1e-3) / c ** 0.5
    norm = _norm(x)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at the origin: tangent space -> Poincare ball."""
    sqrt_c = c ** 0.5
    norm = _norm(v)
    return project_to_ball(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c)


def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithmic map at the origin: Poincare ball -> tangent space."""
    sqrt_c = c ** 0.5
    norm = _norm(y)
    return artanh(sqrt_c * norm) * y / (sqrt_c * norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Mobius addition x (+) y on the Poincare ball."""
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / den.clamp_min(EPS)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Geodesic distance on the Poincare ball, shape (...,)."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c).norm(dim=-1)
    return 2.0 / sqrt_c * artanh(sqrt_c * diff)


def to_sphere(x: torch.Tensor) -> torch.Tensor:
    """Project onto the unit hypersphere."""
    return x / _norm(x)


def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Great-circle (angular) distance between points on the unit hypersphere."""
    cos = (to_sphere(x) * to_sphere(y)).sum(dim=-1).clamp(-1 + EPS, 1 - EPS)
    return torch.acos(cos)
