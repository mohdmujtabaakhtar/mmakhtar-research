"""Unit tests for the fusion, geometry and audio-language-model papers."""
import math

import numpy as np
import pytest
import torch

from common.alm import PrefixLMHead, apply_lora
from common.divergences import (bhattacharyya, cca_objective, chernoff, js_divergence, renyi,
                                to_distribution)
from common.geometry import (sphere_expmap_north, sphere_logmap_north, stereographic_to_ball)
from common.layers import EmbeddingCNN
from common.ot import ot_exchange
from common.runner import classification_scores, eer_one_vs_rest
from papers.coffe.model import Coffe, CoffeConfig
from papers.garuda.model import Garuda, GarudaConfig
from papers.mata.model import Mata, MataConfig
from papers.micunet.model import MiCuNet, MiCuNetConfig
from papers.parrot.model import Parrot, ParrotConfig
from papers.reno.model import Reno, RenoConfig
from papers.rhyme.model import Rhyme, RhymeConfig
from papers.satyam.model import Satyam, SatyamConfig
from papers.signal.model import Signal, SignalConfig
from papers.snifr.model import Snifr, SnifrConfig, SnifrUnimodal
from papers.trio.model import Trio, TrioConfig


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def two_stream_batch(b=8, d1=48, d2=64, classes=4):
    return {"fm1": torch.randn(b, 1, d1, generator=g()), "fm2": torch.randn(b, 1, d2, generator=g(1)),
            "label": torch.arange(b) % classes}


# ---------------------------------------------------------------- shared pieces
def test_divergences_zero_for_identical_and_positive_otherwise():
    p = to_distribution(torch.randn(4, 10, generator=g()))
    q = to_distribution(torch.randn(4, 10, generator=g(1)))
    for fn in (bhattacharyya, js_divergence, chernoff):
        assert fn(p, p).abs() < 1e-5 and fn(p, q) > 0
    assert js_divergence(p, q) <= math.log(2) + 1e-6
    assert renyi(p, q) > renyi(p, p) - 1e-6


def test_cca_objective_is_high_for_correlated_views():
    x = torch.randn(64, 4, generator=g())
    assert cca_objective(x, x + 0.01 * torch.randn(64, 4, generator=g(1))) > 3.5
    assert cca_objective(x, torch.randn(64, 4, generator=g(2))).abs() < 1.5


def test_ot_exchange_shapes_and_marginals():
    x1, x2 = torch.randn(6, 5, generator=g()), torch.randn(6, 5, generator=g(1))
    a, b, gamma = ot_exchange(x1, x2)
    assert a.shape == b.shape == (6, 5)
    assert torch.allclose(gamma.sum(1), torch.full((6,), 1 / 6), atol=1e-4)


def test_embedding_cnn_output_size():
    cnn = EmbeddingCNN(100, (8, 16))
    assert cnn(torch.randn(3, 100)).shape == (3, 16, 25) and cnn.flat_dim == 16 * 25


def test_sphere_maps_are_inverse_and_stereographic_inside_ball():
    v = torch.randn(5, 6, generator=g())
    v[:, 0] = 0
    v = v / v.norm(dim=-1, keepdim=True) * 1.2
    assert torch.allclose(sphere_logmap_north(sphere_expmap_north(v)), v, atol=1e-4)
    x = 0.9 * v / v.norm(dim=-1, keepdim=True)
    assert (stereographic_to_ball(x).norm(dim=-1) < 1).all()


def test_eer_one_vs_rest_perfect_scores():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert eer_one_vs_rest(y, np.eye(3)[y]) == 0.0
    assert classification_scores(y, np.eye(3)[y])["acc"] == 100.0


def test_prefix_lm_head_scores_answers_and_lora_trains():
    head = PrefixLMHead("toy", prompt="is it real or fake ?", answers=("real", "fake"), lora_rank=4)
    prefix = torch.randn(3, 2, head.hidden_size, requires_grad=True)
    out = head(prefix, torch.tensor([0, 1, 1]))
    assert out["logits"].shape == (3, 2) and (out["logits"] <= 0).all()
    out["lm_loss"].backward()
    assert prefix.grad is not None
    trainable = [n for n, p in head.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)


def test_apply_lora_counts_targets():
    model = torch.nn.Sequential()
    model.q_proj, model.v_proj, model.k_proj = (torch.nn.Linear(4, 4) for _ in range(3))
    assert apply_lora(model, ("q_proj", "v_proj"), rank=2) == 2


# ---------------------------------------------------------------- two-stream fusion papers
FUSION = [
    (Coffe, CoffeConfig, {"filters": (4, 8), "proj_dim": 16}, [{}, {"lam": 0.0}]),
    (Trio, TrioConfig, {"filters": (8, 4), "cca_dim": 4}, [{}, {"fusion": "concat"}]),
    (Parrot, ParrotConfig, {"filters": (4, 8), "proj_dim": 12},
     [{}, {"fusion": "concat"}, {"fusion": "hadamard"}, {"fusion": "ot"}]),
    (Mata, MataConfig, {"filters": (4, 8), "proj_dim": 16, "heads": 4},
     [{}, {"use_mha": False}, {"fusion": "concat"}]),
    (Reno, RenoConfig, {"filters": (4, 8), "align_dim": 16}, [{}, {"fusion": "concat"}]),
]


@pytest.mark.parametrize("cls,cfg_cls,small,variant",
                         [(c, k, s, v) for c, k, s, vs in FUSION for v in vs])
def test_fusion_models_forward_backward(cls, cfg_cls, small, variant):
    model = cls(cfg_cls(input_dims={"fm1": 48, "fm2": 64}, num_classes=4, **small, **variant))
    batch = two_stream_batch()
    out = model(batch)
    assert out["logits"].shape == (8, 4)
    loss, parts = model.loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss) and "ce" in parts


@pytest.mark.parametrize("fusion", ["snifr", "ct", "ec", "lc", "ea", "ep"])
def test_snifr_fusions(fusion):
    cfg = SnifrConfig(input_dims={"audio": 48, "video": 48}, d_model=16, num_tokens=4, heads=2, ff=32,
                      fusion=fusion)
    batch = {"audio": torch.randn(4, 1, 48), "video": torch.randn(4, 1, 48), "label": torch.tensor([0, 1, 2, 3])}
    model = Snifr(cfg)
    loss, _ = model.loss(model(batch), batch)
    loss.backward()
    assert len(model.cross) == (2 if fusion == "snifr" else 1)
    uni = SnifrUnimodal(cfg, "video")
    assert uni(batch)["logits"].shape == (4, 4)


# ---------------------------------------------------------------- SIGNAL
def test_signal_gnn_knn_ensemble_and_open_set():
    model = Signal(SignalConfig(input_dim=40, num_classes=3, filters=(4, 8), embed_dim=16, k=3))
    batch = {"features": torch.randn(6, 1, 40), "label": torch.tensor([0, 1, 2, 0, 1, 3])}   # 3 = unseen
    out = model(batch)
    loss, _ = model.loss(out, batch)
    loss.backward()
    model.eval()
    with torch.no_grad():
        model.fit_knn(model(batch)["z"], torch.tensor([0, 1, 2, 0, 1, 2]))
        out = model(batch)
    assert torch.allclose(out["p_ens"].sum(-1), torch.ones(6), atol=1e-5)
    assert (out["entropy"] >= 0).all()


# ---------------------------------------------------------------- RHYME
@pytest.mark.parametrize("variant", [{}, {"gating": "fixed"}, {"branches": "hyperbolic"},
                                     {"branches": "spherical"}, {"fusion": "euclidean"}])
def test_rhyme_variants(variant):
    model = Rhyme(RhymeConfig(input_dim=12, conv_channels=(16, 8), hidden=8, **variant))
    batch = {"features": torch.randn(4, 10, 12), "label": torch.tensor([0, 1, 0, 1])}
    out = model(batch)
    loss, _ = model.loss(out, batch)
    loss.backward()
    assert out["logits"].shape == (4, 2)
    if variant.get("fusion") != "euclidean":
        assert model.raw_c.grad is not None          # the curvature is learned


# ---------------------------------------------------------------- MiCuNet
@pytest.mark.parametrize("variant", [{}, {"spaces": ("euclidean",)}, {"spaces": ("hyperbolic", "euclidean")},
                                     {"gating": "uniform"}, {"fusion": "concat"}])
def test_micunet_variants(variant):
    cfg = MiCuNetConfig(ptm_dim=32, num_classes={"oe": 5, "ce": 5, "source": 7}, ptm_filters=(4, 8),
                        spec_channels=(4, 6, 8), heads=2, manifold_dim=16, head_hidden=8, **variant)
    model = MiCuNet(cfg)
    batch = {"ptm": torch.randn(4, 1, 32), "spec": torch.randn(4, 32, 16),
             "oe": torch.tensor([0, 1, 2, 3]), "ce": torch.tensor([4, 3, 2, 1]), "source": torch.tensor([0, 6, 2, 3])}
    out = model(batch)
    loss, parts = model.loss(out, batch)
    loss.backward()
    assert set(parts) == {"oe", "ce", "source"} and out["logits_source"].shape == (4, 7)
    if "gate" in out:
        assert torch.allclose(out["gate"].sum(-1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------- SATYAM and GARUDA (toy LM)
@pytest.mark.parametrize("variant", [{}, {"geometry": "euclidean"}, {"bd_st": False},
                                     {"bd_ss": False, "bd_st": False},
                                     {"fusion": "concat", "bd_ss": False, "bd_st": False}])
def test_satyam_variants(variant):
    model = Satyam(SatyamConfig(input_dims={"whisper": 32, "trillsson": 48}, conv_filters=4, shared_dim=16,
                                lm={"backbone": "toy", "toy_hidden": 16}, **variant))
    batch = {"whisper": torch.randn(4, 1, 32), "trillsson": torch.randn(4, 1, 48), "label": torch.tensor([0, 1, 0, 1])}
    out = model(batch)
    loss, parts = model.loss(out, batch)
    loss.backward()
    assert out["logits"].shape == (4, 2) and "lm" in parts
    assert not any(p.requires_grad for p in model.lm.lm.parameters())
    assert model.W_g[1].weight.grad is not None


@pytest.mark.parametrize("variant", [{}, {"fusion": "concat"}, {"align": "kl"}])
def test_garuda_variants(variant):
    model = Garuda(GarudaConfig(input_dims={"whisper": 32, "xvector": 32}, conv_filters=4, hidden=8,
                                lm={"backbone": "toy", "toy_hidden": 16}, **variant))
    batch = {"whisper": torch.randn(4, 1, 32), "xvector": torch.randn(4, 1, 32), "label": torch.tensor([0, 1, 0, 1])}
    out = model(batch)
    loss, parts = model.loss(out, batch)
    loss.backward()
    assert out["logits"].shape == (4, 2)
    assert ("js" in parts or "kl" in parts) == (variant.get("fusion") != "concat")


def test_satyam_with_a_hugging_face_decoder(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests.tiny_hf import make_tiny_qwen2
    from papers.satyam.model import CONDITION_PROMPT, DECISION_PROMPT
    model_dir, tok_dir = make_tiny_qwen2(str(tmp_path), [DECISION_PROMPT, CONDITION_PROMPT, "Real Fake"])
    model = Satyam(SatyamConfig(input_dims={"whisper": 32, "trillsson": 48}, conv_filters=4, shared_dim=16,
                                lm={"backbone": model_dir, "tokenizer": tok_dir, "lora_rank": 2}))
    batch = {"whisper": torch.randn(2, 32), "trillsson": torch.randn(2, 48), "label": torch.tensor([0, 1])}
    loss, _ = model.loss(model(batch), batch)
    loss.backward()
    assert torch.isfinite(loss)
