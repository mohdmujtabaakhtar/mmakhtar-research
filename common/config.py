"""YAML configs with command-line overrides.

    cfg = load_config("configs/x.yaml", ["model.geometry=euclidean", "training.epochs=5"])

Override values are parsed as YAML, so numbers, booleans and lists work as expected.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
