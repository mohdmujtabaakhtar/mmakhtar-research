"""DIVINE: DIsentangled Variational INformation NEtwork.

PyTorch implementation of DIVINE from Akhtar, Girish and Singh, "DIVINE:
Coordinating Multimodal Disentangled Representations for Oro-Facial
Neurological Disorder Assessment" (EACL 2026).

Pipeline for each modality m in {video, audio}:

    X_m (frozen FM features, T x d_m)
      -> local temporal refinement   [Conv1d + BN + ReLU + MaxPool] x2  X'_m
      -> local VAE (per time step)   z_sig(t), reconstruct X'_m[t]       L_w
      -> global average pooling      z_bar_m
      -> utterance VAE               shared (weight-tied) + private      L_u
Then across modalities:
      -> cross-modal alignment       D_a(z_shared^v) ~ z_shared^a        L_cycle
      -> sparse gated fusion         g_m = sigmoid(W_m z_priv^m)          L_sparse
                                     h = g_v * z_sh^v + g_a * z_sh^a
      -> symptom-token injection     [T_1..T_K, h] -> Dense -> H_out     L_token
      -> heads on H_out[K+1]         diagnosis and severity

    L = L_cls + alpha L_sev + eps (L_cycle + L_sparse + eps lambda L_token)
        + sum_m (L_w^m + L_u^m)

Loss reductions: every reconstruction, KL, cycle and sparsity term is averaged
over feature dimensions, which keeps the unsupervised terms on the scale of the
cross-entropy losses and prevents the shared latents from collapsing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.losses import kl_standard_normal, l1_sparsity, reparameterise

MODALITIES = ("video", "audio")


@dataclass
class DivineConfig:
    input_dims: dict = field(default_factory=lambda: {"video": 768, "audio": 768})
    refine_dim: int = 128          # d'_m, channels of the local temporal CNN
    refine_blocks: int = 2         # conv blocks in the local temporal CNN (Figure 2: "2x")
    window_latent_dim: int = 64    # d_w, local VAE latent size
    shared_dim: int = 64           # d_s, shared / private latent size
    num_tokens: int = 8            # K learnable clinical symptom tokens
    num_diagnosis: int = 3         # HC, ALS, Stroke
    num_severity: int = 3          # Mild, Moderate, Severe
    alpha: float = 2.0             # severity loss weight (paper)
    epsilon: float = 0.1           # regulariser weight (paper)
    lam: float = 0.4               # token loss weight (paper)
    beta_shared: float = 1.0       # beta_s
    beta_private: float = 1.0      # beta_p
    dropout: float = 0.3
    modality_dropout: float = 0.0  # prob. of dropping one modality per training sample
    kl_warmup_steps: int = 300     # linear KL annealing to avoid posterior collapse
    # Ablation switches (paper Table 5): drop one regulariser at a time.
    use_cycle: bool = True
    use_sparse: bool = True
    use_token_loss: bool = True


class LocalRefine(nn.Module):
    """X'_m = CNN_m(X_m): [Conv1d -> BatchNorm -> ReLU -> MaxPool(2)] x num_blocks.

    Figure 2 of the paper marks this block "2x"; the text describes one block.
    """

    def __init__(self, in_dim: int, out_dim: int, num_blocks: int = 2):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            layers += [nn.Conv1d(in_dim if i == 0 else out_dim, out_dim, 3, padding=1),
                       nn.BatchNorm1d(out_dim), nn.ReLU(), nn.MaxPool1d(2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, T // 2^blocks, d')
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class LocalVAE(nn.Module):
    """VAE_window: encodes every refined time step independently."""

    def __init__(self, dim: int, latent: int):
        super().__init__()
        self.enc = nn.Linear(dim, 2 * latent)
        self.dec = nn.Sequential(nn.Linear(latent, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        z = reparameterise(mu, logvar, self.training)
        return z, F.mse_loss(self.dec(z), x), kl_standard_normal(mu, logvar)


class Gaussian(nn.Module):
    """Maps a vector to the mean and log-variance of a diagonal Gaussian."""

    def __init__(self, in_dim: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 2 * latent))

    def forward(self, x: torch.Tensor):
        return self.net(x).chunk(2, dim=-1)


class Divine(nn.Module):
    def __init__(self, cfg: DivineConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        d_r, d_w, d_s, k = cfg.refine_dim, cfg.window_latent_dim, cfg.shared_dim, cfg.num_tokens

        self.refine = nn.ModuleDict({m: LocalRefine(cfg.input_dims[m], d_r, cfg.refine_blocks)
                                     for m in MODALITIES})
        self.local_vae = nn.ModuleDict({m: LocalVAE(d_r, d_w) for m in MODALITIES})

        # Utterance-level VAE: one shared encoder (weight-tied across modalities),
        # one private encoder per modality, and a per-modality decoder that
        # reconstructs z_bar from [z_shared, z_priv]  [impl: decoder form].
        self.shared_enc = Gaussian(d_w, d_s)
        self.private_enc = nn.ModuleDict({m: Gaussian(d_w, d_s) for m in MODALITIES})
        self.utt_dec = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(2 * d_s, d_w), nn.ReLU(), nn.Linear(d_w, d_w)) for m in MODALITIES
        })

        # Cross-modal alignment: decode video-shared into the audio-shared space.
        self.cross_dec = nn.Sequential(nn.Linear(d_s, d_s), nn.ReLU(), nn.Linear(d_s, d_s))

        # Sparse gates computed from the private latents.
        self.gate = nn.ModuleDict({m: nn.Linear(d_s, d_s) for m in MODALITIES})

        # Clinical symptom tokens and the dense layer over [T_1..T_K, h_fused].
        # "Dense(S)" is applied to the flattened sequence so that tokens and
        # h_fused interact; a per-row dense layer would leave h unaffected by tokens.
        # The LayerNorm in front rescales the (initially tiny) fused latent, which
        # removes a long flat start to training.
        self.tokens = nn.Parameter(torch.randn(k, d_s) * 0.02)
        self.token_dense = nn.Sequential(nn.LayerNorm((k + 1) * d_s),
                                         nn.Linear((k + 1) * d_s, (k + 1) * d_s), nn.ReLU(),
                                         nn.Dropout(cfg.dropout))
        # L_token ("token reconstruction" in the ablation): reconstruct
        # h_fused from the pooled token outputs, plus a decorrelation term that
        # keeps the K symptom tokens specialised (distinct from each other).
        self.token_recon = nn.Linear(d_s, d_s)

        self.head_cls = nn.Linear(d_s, cfg.num_diagnosis)
        self.head_sev = nn.Linear(d_s, cfg.num_severity)

    # ------------------------------------------------------------------ utilities
    def _masks(self, batch: dict, bsz: int, device) -> dict[str, torch.Tensor]:
        """Per-sample presence masks (1 = modality available)."""
        masks = {m: batch.get(f"mask_{m}", torch.ones(bsz, device=device)).float() for m in MODALITIES}
        if self.training and self.cfg.modality_dropout > 0:
            drop = torch.rand(bsz, device=device) < self.cfg.modality_dropout
            drop_video = torch.rand(bsz, device=device) < 0.5
            masks["video"] = masks["video"] * ~(drop & drop_video)
            masks["audio"] = masks["audio"] * ~(drop & ~drop_video)
        return masks

    # --------------------------------------------------------------------- forward
    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        bsz, device = batch["video"].shape[0], batch["video"].device
        masks = self._masks(batch, bsz, device)
        out: dict = {"aux": {}}
        shared, private = {}, {}
        aux_losses = []
        if self.training:
            self.step += 1
        kl_w = min(1.0, self.step.item() / max(cfg.kl_warmup_steps, 1))

        for m in MODALITIES:
            x = batch[m] * masks[m][:, None, None]
            x_ref = self.refine[m](x)                                   # X'_m
            z_sig, rec_w, kl_loc = self.local_vae[m](x_ref)
            loss_w = rec_w + kl_w * kl_loc                              # L_w
            z_bar = z_sig.mean(dim=1)                                   # global average pooling

            mu_s, lv_s = self.shared_enc(z_bar)
            mu_p, lv_p = self.private_enc[m](z_bar)
            shared[m] = reparameterise(mu_s, lv_s, self.training)
            private[m] = reparameterise(mu_p, lv_p, self.training)
            recon = F.mse_loss(self.utt_dec[m](torch.cat([shared[m], private[m]], -1)), z_bar)
            loss_u = recon + kl_w * (cfg.beta_shared * kl_standard_normal(mu_s, lv_s)
                                     + cfg.beta_private * kl_standard_normal(mu_p, lv_p))  # L_u
            aux_losses += [loss_w, loss_u]
            out["aux"][f"L_w_{m}"], out["aux"][f"L_u_{m}"] = loss_w, loss_u

        # Cross-modal alignment, only where both modalities are present.
        both = masks["video"] * masks["audio"]
        err = (self.cross_dec(shared["video"]) - shared["audio"]).pow(2).mean(-1)
        out["aux"]["L_cycle"] = (err * both).sum() / both.sum().clamp_min(1.0)

        # Sparse gated fusion; a missing modality gets a zero gate.
        gates = {m: torch.sigmoid(self.gate[m](private[m])) * masks[m][:, None] for m in MODALITIES}
        h_fused = gates["video"] * shared["video"] + gates["audio"] * shared["audio"]
        out["aux"]["L_sparse"] = l1_sparsity(gates["video"], gates["audio"])
        out["gates"] = gates

        # Token injection and dense layer.
        k, d_s = cfg.num_tokens, cfg.shared_dim
        seq = torch.cat([self.tokens.expand(bsz, k, d_s), h_fused[:, None]], dim=1)   # S
        h_out = self.token_dense(seq.flatten(1)).view(bsz, k + 1, d_s)                 # H_out
        h = h_out[:, -1]                                                               # H_out[K+1]

        token_rec = F.mse_loss(self.token_recon(h_out[:, :k].mean(1)), h_fused.detach())
        t = F.normalize(self.tokens, dim=-1)
        decorrelation = (t @ t.T - torch.eye(k, device=device)).pow(2).mean()
        out["aux"]["L_token"] = token_rec + decorrelation

        out["logits_diagnosis"] = self.head_cls(h)
        out["logits_severity"] = self.head_sev(h)
        out["aux"]["L_vae"] = sum(aux_losses)
        return out

    def loss(self, out: dict, batch: dict):
        cfg, a = self.cfg, out["aux"]
        l_cls = F.cross_entropy(out["logits_diagnosis"], batch["diagnosis"])
        l_sev = F.cross_entropy(out["logits_severity"], batch["severity"])
        reg = (cfg.use_cycle * a["L_cycle"] + cfg.use_sparse * a["L_sparse"]
               + cfg.use_token_loss * cfg.epsilon * cfg.lam * a["L_token"])
        total = l_cls + cfg.alpha * l_sev + cfg.epsilon * reg + a["L_vae"]
        parts = {"cls": l_cls, "sev": l_sev, "cycle": a["L_cycle"], "sparse": a["L_sparse"],
                 "token": a["L_token"], "vae": a["L_vae"],
                 # early stopping watches the supervised terms only, because the
                 # KL warm-up makes the unsupervised terms rise during early training.
                 "monitor": l_cls + cfg.alpha * l_sev}
        return total, {k: v.item() for k, v in parts.items()}
