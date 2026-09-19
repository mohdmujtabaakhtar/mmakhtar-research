"""Generic training loop with early stopping.

Models plug in by implementing two methods:

* ``forward(batch) -> dict`` returning at least the tensors the loss needs;
* ``loss(outputs, batch) -> (total_loss, dict_of_named_parts)``.

Optional hook: ``after_step(batch, outputs)`` is called after every optimiser
step (used, e.g., for non-gradient updates such as COBALT's bandit scores).

Early stopping tracks the validation value of ``parts["monitor"]`` when the
model provides it (e.g. only the supervised terms), otherwise the total loss.
Pass ``val_loader=None`` to train for a fixed number of epochs.
"""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def batch_size_of(batch: dict) -> int:
    return next(v for v in batch.values() if torch.is_tensor(v)).shape[0]


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        loss, parts = model.loss(model(batch), batch)
        size = batch_size_of(batch)
        total += parts.get("monitor", loss.item()) * size
        n += size
    return total / max(n, 1)


def make_optimizer(params, name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0,
                   betas=(0.9, 0.999), eps: float = 1e-8):
    cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
    return cls(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)


def warmup_cosine(optim, total_steps: int, warmup_frac: float = 0.1):
    """Linear warm-up for ``warmup_frac`` of training, then cosine decay to zero."""
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optim, factor)


def fit(model, train_loader: DataLoader, val_loader: DataLoader | None, *, epochs: int = 50,
        lr: float = 1e-3, patience: int = 10, device: str | torch.device = "cpu",
        optimizer: str = "adam", weight_decay: float = 0.0, betas=(0.9, 0.999),
        grad_clip: float | None = None, schedule: str | None = None, warmup_frac: float = 0.1,
        verbose: bool = True):
    """Train the model; with a validation loader, keep the best weights and stop early."""
    device = torch.device(device)
    model.to(device)
    optim = make_optimizer(model.parameters(), optimizer, lr, weight_decay, betas)
    sched = (warmup_cosine(optim, epochs * len(train_loader), warmup_frac)
             if schedule == "cosine" else None)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move(batch, device)
            out = model(batch)
            loss, _ = model.loss(out, batch)
            optim.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if sched is not None:
                sched.step()
            if hasattr(model, "after_step"):
                model.after_step(batch, out)
        if val_loader is None:
            continue
        val_loss = evaluate_loss(model, val_loader, device)
        if verbose:
            print(f"  epoch {epoch:3d}  val_loss {val_loss:.4f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
