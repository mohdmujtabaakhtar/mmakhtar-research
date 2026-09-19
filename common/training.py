"""Generic training loop with early stopping.

Models plug in by implementing two methods:

* ``forward(batch) -> dict`` returning at least the tensors the loss needs;
* ``loss(outputs, batch) -> (total_loss, dict_of_named_parts)``.

Early stopping tracks the validation value of ``parts["monitor"]`` when the
model provides it (e.g. only the supervised terms), otherwise the total loss.
"""
from __future__ import annotations

import copy
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


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        loss, parts = model.loss(model(batch), batch)
        size = next(iter(batch.values())).shape[0]
        total += parts.get("monitor", loss.item()) * size
        n += size
    return total / max(n, 1)


def fit(model, train_loader: DataLoader, val_loader: DataLoader, *, epochs: int = 50,
        lr: float = 1e-3, patience: int = 10, device: str | torch.device = "cpu",
        verbose: bool = True):
    """Train with Adam and keep the weights with the lowest validation loss."""
    device = torch.device(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move(batch, device)
            loss, _ = model.loss(model(batch), batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
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
