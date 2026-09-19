import numpy as np
import pytest
import torch

from common import geometry, metrics
from common.baselines import MultiTaskProbe
from common.data import FeatureDataset, fix_length, make_synthetic_manifest, subject_kfold


def test_expmap_logmap_roundtrip():
    v = torch.randn(32, 16) * 0.2  # well inside the ball; larger norms are clipped by design
    for c in (0.5, 1.0, 2.0):
        assert torch.allclose(geometry.logmap0(geometry.expmap0(v, c), c), v, atol=1e-4)


def test_points_stay_in_ball():
    x = geometry.expmap0(torch.randn(64, 8) * 50, c=1.0)
    assert (x.norm(dim=-1) < 1.0).all()


def test_poincare_distance_properties():
    x = geometry.expmap0(torch.randn(10, 4) * 0.3)
    y = geometry.expmap0(torch.randn(10, 4) * 0.3)
    d = geometry.poincare_distance(x, y)
    assert torch.allclose(d, geometry.poincare_distance(y, x), atol=1e-5)
    assert torch.allclose(geometry.poincare_distance(x, x), torch.zeros(10), atol=1e-3)
    assert (d >= 0).all()


def test_spherical_distance_range():
    d = geometry.spherical_distance(torch.randn(20, 6), torch.randn(20, 6))
    assert ((d >= 0) & (d <= torch.pi)).all()


def test_metrics():
    assert metrics.accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(200 / 3)
    assert metrics.mae([1, 2], [2, 4]) == pytest.approx(1.5)
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert metrics.eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_fix_length():
    assert fix_length(np.ones((10, 3)), 4).shape == (4, 3)
    padded = fix_length(np.ones((2, 3)), 5)
    assert padded.shape == (5, 3) and padded[2:].sum() == 0
    assert fix_length(np.ones(3), 2).shape == (2, 3)


def test_subject_kfold_has_no_leakage():
    subjects = np.repeat([f"S{i}" for i in range(10)], 3)
    seen_test = set()
    for train, val, test in subject_kfold(subjects, k=5, seed=1):
        s_tr, s_va, s_te = (set(subjects[i]) for i in (train, val, test))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        assert len(train) + len(val) + len(test) == len(subjects)
        seen_test |= s_te
    assert seen_test == set(subjects)  # every subject is tested exactly once


def test_dataset_and_baseline(tmp_path):
    manifest = make_synthetic_manifest(tmp_path, {"video": (8, 12), "audio": (10, 6)},
                                       n_subjects=6, samples_per_subject=2)
    ds = FeatureDataset(manifest, {"video": 8, "audio": 10}, ["diagnosis", "severity"])
    item = ds[0]
    assert item["video"].shape == (8, 12) and item["audio"].shape == (10, 6)
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in item}
    for head in ("cnn", "fcn"):
        model = MultiTaskProbe({"video": (8, 12), "audio": (10, 6)}, {"diagnosis": 3, "severity": 3}, head)
        loss, _ = model.loss(model(batch), batch)
        loss.backward()
        assert torch.isfinite(loss)
