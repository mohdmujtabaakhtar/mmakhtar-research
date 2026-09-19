"""Non-Euclidean geometry utilities: Poincare ball, hypersphere and Euclidean space.

Several papers in this repository (RHYME, MiCuNet, NOVA-ARC, SATYAM, COBALT,
ORBIT, PHOENIX-Mamba) project foundation-model embeddings into hyperbolic
and/or spherical spaces. These are the shared primitives they build on.

Curvature is parameterised by ``c > 0`` for the Poincare ball of radius 1/sqrt(c).
All functions broadcast over leading dimensions, so pairwise distances between
a batch and a set of prototypes are ``dist(x[:, None], protos[None])``.

The ``Manifold`` classes give the papers one interface for all three
geometries. That makes "Euclidean counterpart" ablations a one-word config change.
"""
from __future__ import annotations

import torch

EPS = 1e-5


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1 + EPS, 1 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


# ------------------------------------------------------------------ Poincare ball

def clip_tangent(v: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """Rescale tangent vectors whose norm exceeds ``max_norm`` (feature clipping).

    Without it, wide layers produce large tangent vectors, the exponential map
    saturates at the ball boundary and gradients vanish (Guo et al., CVPR 2022).
    """
    norm = _norm(v)
    return torch.where(norm > max_norm, v / norm * max_norm, v)


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


def poincare_pairwise(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """All-pairs Poincare distances between x (N, d) and y (M, d), shape (N, M).

    Uses d = arcosh(1 + 2c||x - y||^2 / ((1 - c||x||^2)(1 - c||y||^2))) / sqrt(c),
    which only needs squared Euclidean distances and is far lighter on memory
    than broadcasting Mobius addition over every pair.
    """
    sq = torch.cdist(x, y).pow(2)
    nx = (1 - c * (x * x).sum(-1, keepdim=True)).clamp_min(EPS)
    ny = (1 - c * (y * y).sum(-1, keepdim=True)).clamp_min(EPS)
    arg = 1 + 2 * c * sq / (nx * ny.transpose(0, 1))
    return torch.acosh(arg.clamp_min(1 + EPS)) / c ** 0.5


def _conformal(x: torch.Tensor, c: float) -> torch.Tensor:
    return 2.0 / (1 - c * (x * x).sum(-1, keepdim=True)).clamp_min(EPS)


def expmap(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at point x of the Poincare ball."""
    sqrt_c = c ** 0.5
    norm = _norm(v)
    step = torch.tanh(sqrt_c * _conformal(x, c) * norm / 2) * v / (sqrt_c * norm)
    return project_to_ball(mobius_add(x, step, c), c)


def logmap(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Logarithmic map at point x of the Poincare ball."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    norm = _norm(diff)
    return 2.0 / (sqrt_c * _conformal(x, c)) * artanh(sqrt_c * norm) * diff / norm


def frechet_mean(points: torch.Tensor, c: float = 1.0, iters: int = 20) -> torch.Tensor:
    """Frechet (Karcher) mean of points (N, d) on the Poincare ball.

    Iterates mu <- exp_mu(mean_i log_mu(x_i)), starting from the tangent-space mean.
    """
    mu = expmap0(logmap0(points, c).mean(0), c)
    for _ in range(iters):
        step = logmap(mu[None], points, c).mean(0)
        mu = expmap(mu, step, c)
        if step.norm() < 1e-6:
            break
    return mu


# ------------------------------------------------------------------ hypersphere

def to_sphere(x: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Project onto the hypersphere of the given radius."""
    return radius * x / _norm(x)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Great-circle distance r * arccos(<x, y> / r^2) between points on the sphere."""
    cos = (to_sphere(x) * to_sphere(y)).sum(dim=-1).clamp(-1 + EPS, 1 - EPS)
    return radius * torch.acos(cos)


# ------------------------------------------------------------------ manifold interface

class Manifold:
    """Common interface: map tangent vectors in, measure distances, average points."""

    name = "base"

    def from_tangent(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to_tangent(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """All-pairs distances between x (N, d) and y (M, d), shape (N, M)."""
        return self.dist(x[:, None], y[None])


class PoincareBall(Manifold):
    name = "hyperbolic"

    def __init__(self, c: float = 1.0):
        self.c = c

    def from_tangent(self, v):
        return expmap0(v, self.c)

    def to_tangent(self, x):
        return logmap0(x, self.c)

    def dist(self, x, y):
        return poincare_distance(x, y, self.c)

    def add(self, x, y):
        return project_to_ball(mobius_add(x, y, self.c), self.c)

    def mean(self, points):
        return frechet_mean(points, self.c)

    def pairwise(self, x, y):
        return poincare_pairwise(x, y, self.c)


class Sphere(Manifold):
    name = "sphere"

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def from_tangent(self, v):
        return to_sphere(v, self.radius)

    def to_tangent(self, x):
        return x

    def dist(self, x, y):
        return spherical_distance(x, y, self.radius)

    def add(self, x, y):
        return to_sphere(x + y, self.radius)

    def mean(self, points):
        return to_sphere(points.mean(0), self.radius)


class Euclidean(Manifold):
    name = "euclidean"

    def from_tangent(self, v):
        return v

    def to_tangent(self, x):
        return x

    def dist(self, x, y):
        return (x - y).norm(dim=-1)

    def add(self, x, y):
        return x + y

    def mean(self, points):
        return points.mean(0)

    def pairwise(self, x, y):
        return torch.cdist(x, y)


def make_manifold(name: str, curvature: float = 1.0, radius: float = 1.0) -> Manifold:
    """``hyperbolic`` / ``sphere`` / ``euclidean``."""
    if name == "hyperbolic":
        return PoincareBall(curvature)
    if name == "sphere":
        return Sphere(radius)
    if name == "euclidean":
        return Euclidean()
    raise ValueError(f"Unknown manifold '{name}'")


# ------------------------------------------------------------------ mixed-curvature helpers

def stereographic_to_ball(x: torch.Tensor, c=1.0) -> torch.Tensor:
    """Map a point inside the radius-1/sqrt(c) ball (e.g. a scaled sphere point) with
    y = x / (1 + sqrt(1 - c ||x||^2)), the inverse stereographic map used by RHYME."""
    sq = (c * (x * x).sum(-1, keepdim=True)).clamp(max=1 - 1e-5)
    return x / (1 + torch.sqrt(1 - sq))


def sphere_logmap_north(x: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at the north pole n = e_1 of the unit sphere:
    log_n(x) = theta (x - cos(theta) n) / sin(theta), theta = arccos(<x, n>)."""
    x = to_sphere(x)
    cos = x[..., :1].clamp(-1 + EPS, 1 - EPS)
    theta = torch.acos(cos)
    north = torch.zeros_like(x)
    north[..., 0] = 1.0
    return theta * (x - cos * north) / torch.sin(theta)


def sphere_expmap_north(v: torch.Tensor) -> torch.Tensor:
    """Exponential map at the north pole for a tangent vector v (first coordinate ignored)."""
    v = torch.cat([torch.zeros_like(v[..., :1]), v[..., 1:]], dim=-1)
    norm = _norm(v)
    north = torch.zeros_like(v)
    north[..., 0] = 1.0
    return torch.cos(norm) * north + torch.sin(norm) * v / norm
