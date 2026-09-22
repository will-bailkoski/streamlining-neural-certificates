"""Certificate parameters on disk: flat npz columns (layer{i}_W / layer{i}_b), no pickle."""

from __future__ import annotations
from pathlib import Path

import numpy as np


def params_to_npz(params) -> dict:
    """[(W0, b0), ...] -> {"layer0_W": W0, "layer0_b": b0, ...} for np.savez."""
    out = {}
    for i, (W, b) in enumerate(params):
        out[f"layer{i}_W"] = np.asarray(W)
        out[f"layer{i}_b"] = np.asarray(b)
    return out


def load_params(path: Path):
    """Inverse of the npz layout -> [(W0, b0), ...] as numpy arrays."""
    arrays = np.load(path)
    n = sum(1 for k in arrays.files if k.endswith("_W"))
    return [(arrays[f"layer{i}_W"], arrays[f"layer{i}_b"]) for i in range(n)]
