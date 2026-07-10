"""
Certificate fixture store.

A named, on-disk library of neural certificates (supermartingale candidates) so
experiments can reuse the SAME known-valid / known-invalid certs instead of
re-training every run. Layout:

    certs/manifest.json                                  # list of CertEntry dicts
    certs/<env>/<struct>_<sizes>_s<seed>_<validity>.npz  # params (recorder npz layout)

A `valid` cert is a trained supermartingale (max MC drift on domain\\eq <= tol).
An `invalid` cert has a genuine counterexample; its `known_cex` is the worst
(highest-drift) point found, so the refuter/verifier CE-finding experiments have
a ground-truth target. Params use the same flat npz columns as `Recorder`
(`layer{i}_W` / `layer{i}_b`), so `load_params` round-trips them.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import json

import numpy as np

from src.results.recorder import _params_to_npz, load_params
from src.certificates.structures import get_spec

DEFAULT_ROOT = "certs"


@dataclass
class CertEntry:
    cert_id: str
    env: str
    structure: str
    hidden: list           # hidden layer widths, e.g. [8, 8]
    seed: int
    epsilon: float
    validity: str          # "valid" | "invalid"
    max_drift: float       # MC-estimated max drift on domain \ equilibrium
    path: str              # npz path relative to the store root
    known_cex: Optional[list] = None  # worst-drift point (invalid certs only)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CertEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def make_cert_id(env: str, structure: str, hidden: list, seed: int, validity: str) -> str:
    size = "x".join(str(h) for h in hidden) if hidden else "lin"
    return f"{env}__{structure}__{size}__s{seed}__{validity}"


def _manifest_path(root) -> Path:
    return Path(root) / "manifest.json"


def load_manifest(root=DEFAULT_ROOT) -> list[CertEntry]:
    p = _manifest_path(root)
    if not p.exists():
        return []
    with open(p) as f:
        return [CertEntry.from_dict(d) for d in json.load(f)]


def _write_manifest(root, entries: list[CertEntry]) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    with open(_manifest_path(root), "w") as f:
        json.dump([e.to_dict() for e in entries], f, indent=2)


def has_cert(cert_id: str, root=DEFAULT_ROOT) -> bool:
    return any(e.cert_id == cert_id for e in load_manifest(root))


def save_cert(entry: CertEntry, params, root=DEFAULT_ROOT) -> CertEntry:
    """Persist params to `entry.path` and upsert the manifest entry."""
    npz_path = Path(root) / entry.path
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **_params_to_npz(params))
    entries = [e for e in load_manifest(root) if e.cert_id != entry.cert_id]
    entries.append(entry)
    _write_manifest(root, entries)
    return entry


def get_entry(cert_id: str, root=DEFAULT_ROOT) -> CertEntry:
    for e in load_manifest(root):
        if e.cert_id == cert_id:
            return e
    raise KeyError(f"no cert '{cert_id}' in store '{root}'. "
                   f"Available: {[e.cert_id for e in load_manifest(root)]}")


def load_cert(cert_id: str, root=DEFAULT_ROOT):
    """Return (spec, params, entry) for a stored certificate."""
    entry = get_entry(cert_id, root)
    params = load_params(Path(root) / entry.path)
    spec = get_spec(entry.structure)
    return spec, params, entry


def query(
    root=DEFAULT_ROOT,
    *,
    env: str | None = None,
    validity: str | None = None,
    structure: str | None = None,
    hidden: list | None = None,
) -> list[CertEntry]:
    """Filter manifest entries by any of env / validity / structure / hidden."""
    out = []
    for e in load_manifest(root):
        if env is not None and e.env != env:
            continue
        if validity is not None and e.validity != validity:
            continue
        if structure is not None and e.structure != structure:
            continue
        if hidden is not None and list(e.hidden) != list(hidden):
            continue
        out.append(e)
    return out
