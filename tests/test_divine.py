import torch

from common.training import set_seed
from papers.divine.model import Divine, DivineConfig

DIMS = {"video": 24, "audio": 16}


def make_batch(bsz=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "video": torch.randn(bsz, 16, DIMS["video"], generator=g),
        "audio": torch.randn(bsz, 16, DIMS["audio"], generator=g),
        "diagnosis": torch.randint(0, 3, (bsz,), generator=g),
        "severity": torch.randint(0, 3, (bsz,), generator=g),
    }


def small_model(**kw):
    cfg = DivineConfig(input_dims=DIMS, refine_dim=64, window_latent_dim=32, shared_dim=32,
                       num_tokens=4, **kw)
    return Divine(cfg)


def test_forward_shapes_and_losses():
    model = small_model()
    out = model(make_batch())
    assert out["logits_diagnosis"].shape == (8, 3)
    assert out["logits_severity"].shape == (8, 3)
    for key in ("L_cycle", "L_sparse", "L_token", "L_vae"):
        assert torch.isfinite(out["aux"][key]), key
    loss, parts = model.loss(out, make_batch())
    loss.backward()
    assert torch.isfinite(loss) and "monitor" in parts
    assert model.tokens.grad is not None  # tokens receive gradient


def test_missing_modality_zeroes_its_gate():
    model = small_model().eval()
    batch = make_batch()
    batch["mask_audio"] = torch.zeros(8)
    with torch.no_grad():
        out = model(batch)
    assert torch.all(out["gates"]["audio"] == 0)
    assert torch.all(out["gates"]["video"] > 0)
    assert out["aux"]["L_cycle"] == 0  # alignment skipped without both modalities


def test_eval_is_deterministic():
    model = small_model().eval()
    batch = make_batch()
    with torch.no_grad():
        a, b = model(batch)["logits_diagnosis"], model(batch)["logits_diagnosis"]
    assert torch.equal(a, b)


def test_kl_warmup_counts_training_steps_only():
    model = small_model(kl_warmup_steps=10)
    model(make_batch())
    model.eval()
    with torch.no_grad():
        model(make_batch())
    assert model.step.item() == 1


def test_learns_a_separable_signal():
    set_seed(0)
    model = small_model(dropout=0.0)
    batch = make_batch(bsz=24, seed=3)
    offsets = torch.tensor([-2.0, 0.0, 2.0]).view(3, 1, 1)  # one offset per diagnosis class
    batch["video"] = batch["video"] + offsets[batch["diagnosis"]]
    batch["audio"] = batch["audio"] + offsets[batch["diagnosis"]]
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = model.loss(model(batch), batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        pred = model(batch)["logits_diagnosis"].argmax(-1)
    assert (pred == batch["diagnosis"]).float().mean() >= 0.9
