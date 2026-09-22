import numpy as np
import torch

from common import geometry as g
from common.config import load_config
from common.features.spectral import lfcc, linear_filterbank, mfcc, stats_pool
from common.mamba import MambaBlock
from common.ot import grad_reverse, sinkhorn


def test_pairwise_matches_broadcast_distance():
    x, y = g.expmap0(torch.randn(6, 5) * 0.4), g.expmap0(torch.randn(4, 5) * 0.4)
    assert torch.allclose(g.poincare_pairwise(x, y), g.poincare_distance(x[:, None], y[None]), atol=1e-4)
    for name in ("hyperbolic", "sphere", "euclidean"):
        m = g.make_manifold(name)
        a, b = m.from_tangent(torch.randn(6, 5) * 0.4), m.from_tangent(torch.randn(4, 5) * 0.4)
        assert torch.allclose(m.pairwise(a, b), m.dist(a[:, None], b[None]), atol=1e-4)


def test_expmap_logmap_at_a_point_roundtrip():
    x = g.expmap0(torch.randn(8, 4) * 0.3)
    v = torch.randn(8, 4) * 0.2
    assert torch.allclose(g.logmap(x, g.expmap(x, v)), v, atol=1e-3)


def test_frechet_mean_minimises_squared_distance():
    pts = g.expmap0(torch.randn(20, 3) * 0.5)
    mu = g.frechet_mean(pts)
    cost = lambda m: g.poincare_distance(m[None], pts).pow(2).sum()
    for _ in range(10):  # nearby points are never better
        assert cost(mu) <= cost(g.project_to_ball(mu + torch.randn(3) * 0.05)) + 1e-4


def test_clip_tangent():
    v = torch.randn(10, 6) * 10
    assert (g.clip_tangent(v, 1.0).norm(dim=-1) <= 1.0 + 1e-5).all()


def test_sinkhorn_marginals():
    cost = torch.rand(5, 12)
    a = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
    b = torch.full((12,), 1 / 12)
    plan = sinkhorn(cost, a, b, epsilon=0.05, iters=200)
    assert torch.allclose(plan.sum(1), a, atol=1e-3)
    assert torch.allclose(plan.sum(0), b, atol=1e-3)


def test_grad_reverse():
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  a: 1\n  b: [x]\n")
    cfg = load_config(p, ["model.a=2.5", "model.b=[y, z]", "training.epochs=3"])
    assert cfg == {"model": {"a": 2.5, "b": ["y", "z"]}, "training": {"epochs": 3}}


def test_mamba_block_is_causal():
    torch.manual_seed(0)
    blk = MambaBlock(16, d_state=4).eval()
    x = torch.randn(2, 10, 16)
    y1 = blk(x)
    x2 = x.clone()
    x2[:, 6:] += 5.0                         # change only the future
    y2 = blk(x2)
    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-5)
    assert not torch.allclose(y1[:, 6:], y2[:, 6:])


def test_spectral_features():
    wav = (np.random.default_rng(0).normal(size=16000) * 0.1).astype(np.float32)
    assert mfcc(wav).shape[1] == 40 and lfcc(wav).shape[1] == 40
    assert abs(len(mfcc(wav)) - len(lfcc(wav))) <= 1
    assert stats_pool(mfcc(wav)).shape == (80,)
    bank = linear_filterbank()
    assert bank.shape == (70, 201) and bank.max() <= 1.0 + 1e-9
