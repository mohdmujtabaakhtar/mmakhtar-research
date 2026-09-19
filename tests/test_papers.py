"""Unit tests for NOVA-ARC, ORBIT, COBALT and PHOENIX-Mamba."""
import pytest
import torch

from common.training import set_seed
from papers.cobalt.model import Cobalt, CobaltConfig
from papers.hcfd.model import PhoenixConfig, PhoenixMamba
from papers.nova_arc.model import NovaArc, NovaArcConfig
from papers.orbit.model import Orbit, OrbitConfig


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- PHOENIX-Mamba
def phoenix_batch(b=6):
    return {"features": torch.randn(b, 12, 10, generator=g()), "label": torch.tensor([0, 1] * (b // 2))}


@pytest.mark.parametrize("variant", [{}, {"geometry": "euclidean"}, {"backbone": "bigru"},
                                     {"backbone": "cnn"}, {"num_evidence": 1}])
def test_phoenix_variants_forward_backward(variant):
    model = PhoenixMamba(PhoenixConfig(input_dim=10, adapter_dim=16, hyp_dim=8, d_state=4,
                                       num_layers=1, **variant))
    out = model(phoenix_batch())
    m, k = model.cfg.num_evidence, model.cfg.num_modes
    assert out["logits"].shape == (6, 2) and out["q"].shape == (6, m, k)
    loss, parts = model.loss(out, phoenix_batch())
    loss.backward()
    assert torch.isfinite(loss) and model.proto_fake.grad is not None


def test_phoenix_points_stay_inside_ball():
    model = PhoenixMamba(PhoenixConfig(input_dim=10, adapter_dim=16, hyp_dim=8, d_state=4, num_layers=1))
    out = model({"features": torch.randn(4, 12, 10) * 100})
    assert (out["evidence"].norm(dim=-1) < 1).all()


# ---------------------------------------------------------------- COBALT
def cobalt_batch(b=8):
    return {"stream1": torch.randn(b, 16, 5, generator=g()), "stream2": torch.randn(b, 8, 7, generator=g(1)),
            "label": torch.tensor([0, 1] * (b // 2))}


def small_cobalt(**kw):
    return Cobalt(CobaltConfig(input_dims={"stream1": 5, "stream2": 7}, adapter_dim=16, num_tokens=4,
                               hyp_dim=8, num_prototypes=6, hidden=16, **kw))


@pytest.mark.parametrize("variant", [{}, {"geometry": "euclidean"}, {"fusion": "mobius"}, {"fusion": "concat"}])
def test_cobalt_variants_forward_backward(variant):
    model = small_cobalt(**variant)
    out = model(cobalt_batch())
    assert out["logits"].shape == (8, 2)
    loss, _ = model.loss(out, cobalt_batch())
    loss.backward()
    assert torch.isfinite(loss)


def test_cobalt_evidence_is_a_distribution_and_bandit_updates():
    model = small_cobalt()
    batch = cobalt_batch()
    out = model(batch)
    for a in out["assign"].values():
        assert torch.allclose(a.sum(-1), torch.ones(a.shape[:2]), atol=1e-5)
    assert torch.allclose(out["w"].sum(), torch.tensor(1.0))
    before = model.Q.clone()
    model.after_step(batch, out)
    assert not torch.equal(before, model.Q)
    assert not model.Q.requires_grad


# ---------------------------------------------------------------- ORBIT
def orbit_batch(b=8):
    return {"audio": torch.randn(b, 12, 6, generator=g()), "text": torch.randn(b, 7, 5, generator=g(1)),
            "label": torch.tensor([0, 1] * (b // 2)), "language": torch.tensor([0, 1, 2, 0] * (b // 4))}


def small_orbit(**kw):
    return Orbit(OrbitConfig(audio_dim=6, text_dim=5, num_languages=3, hidden=16, geo_dim=8, **kw))


@pytest.mark.parametrize("variant", [{}, {"cross_attention": False}, {"use_grl": False},
                                     {"geometries": ["hyperbolic"]}, {"geometries": ["sphere"]},
                                     {"geometries": ["hyperbolic", "euclidean"]}])
def test_orbit_variants_forward_backward(variant):
    model = small_orbit(**variant)
    out = model(orbit_batch())
    assert torch.allclose(out["p_vote"].sum(-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(out["q_c"].sum(-1), torch.ones(8), atol=1e-5)
    loss, _ = model.loss(out, orbit_batch())
    loss.backward()
    assert torch.isfinite(loss)


def test_orbit_sphere_points_have_radius_r():
    model = small_orbit(sphere_radius=2.0)
    out = model(orbit_batch())
    assert torch.allclose(out["geo"]["sphere"]["x"].norm(dim=-1), torch.full((8,), 2.0), atol=1e-4)


def test_orbit_language_adversaries_push_encoder_away():
    """With GRL on, the adversarial term's gradient on f is the negative of a plain classifier's."""
    batch = orbit_batch()
    grads = {}
    for coeff in (1.0, -1.0):  # -1 turns the reversal layer into an ordinary identity
        set_seed(0)
        model = small_orbit(grl_coeff=coeff)
        out = model(batch)
        out["f"].retain_grad()
        sum(torch.nn.functional.cross_entropy(v, batch["language"]) for v in out["lang_logits"].values()).backward()
        grads[coeff] = out["f"].grad.clone()
    assert torch.allclose(grads[1.0], -grads[-1.0], atol=1e-5)


# ---------------------------------------------------------------- NOVA-ARC
def nova_batch(b=10):
    return {"source": torch.randn(b, 6, 8, generator=g()), "target": torch.randn(b, 6, 8, generator=g(1)),
            "label": torch.arange(b) % 5}


def small_nova(**kw):
    return NovaArc(NovaArcConfig(input_dim=8, latent_dim=12, bottleneck_dim=6, codebook_size=16, **kw))


@pytest.mark.parametrize("variant", [{}, {"geometry": "euclidean"}, {"tokens": "continuous"},
                                     {"tokens": "discrete"}, {"fusion": "concat"}, {"use_hel": False},
                                     {"ot_geometry": "euclidean"}, {"adaptation": "none"}])
def test_nova_arc_variants_forward_backward(variant):
    model = small_nova(**variant)
    batch = nova_batch()
    with torch.no_grad():
        enc = model.encode(batch["source"])
    model.refresh_prototypes(model.ot_embedding(enc), batch["label"])
    loss, parts = model.loss(model(batch), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert ("ot_ce" in parts) == (variant.get("adaptation", "ot") == "ot")


def test_nova_arc_prototypes_and_prior():
    model = small_nova()
    labels = torch.tensor([0, 0, 1, 2, 3, 4, 4, 4])
    emb = model._exp(torch.randn(8, 6) * 0.3)
    model.refresh_prototypes(emb, labels)
    assert torch.allclose(model.class_prior, torch.tensor([2, 1, 1, 1, 3]) / 8.0)
    assert (model.prototypes.norm(dim=-1) < 1).all()
    assert torch.allclose(model.prototypes[1], emb[2], atol=1e-4)   # a single point is its own mean
