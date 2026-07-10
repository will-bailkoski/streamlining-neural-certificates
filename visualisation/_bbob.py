"""
Shared data plumbing for the BBOB refuter figures.

Paired experiment: experiments/runners/refute_bbob.py driven by the
experiments/refute_bbob/<refuter>.py launchers (campaigns "refute_bbob_<r>").
Data read:         results/refute_bbob_<refuter>/<tag>/compiled.csv

Each compiled row is one search: (refuter, function, seed, hp columns...,
final_value, time). The suite maximises with f_opt = 0, so raw regret is
-final_value. Regret is NORMALISED per function by the spread of f over the
domain (f_opt - 5th percentile of f on a fixed uniform sample), so functions
with wildly different value scales are comparable, and floored at 1e-7 for the
log plots.

"Best HPs" for a refuter = the hp combination with the lowest mean normalised
regret across all functions x seeds — one global setting per refuter, exactly
what a user of the refuter would deploy (no per-function tuning).
"""

from __future__ import annotations
import numpy as np

from visualisation.common import glob_campaigns, load_compiled, fnum

# columns that are not hyperparameters (budget/batch_size are fairness-fixed;
# nregret is derived by normalised_regret, not a column)
_META = {"experiment", "run_id", "status", "dim", "function", "refuter", "seed",
         "final_value", "time", "budget", "batch_size", "campaign", "tag",
         "source_root", "source_csv", "nregret"}

REGRET_FLOOR = 1e-7


def load_rows(tag: str | None = None) -> list[dict]:
    """All refuters' rows. With no explicit tag, each campaign contributes its
    LARGEST tag (row count) — the real sweep, never a later smoke test."""
    rows = []
    for title in glob_campaigns("refute_bbob_"):
        if tag is not None:
            rows.extend(load_compiled(title, tag))
            continue
        from visualisation.common import campaign_dir
        root = campaign_dir(title)
        tags = [d.name for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
        best: list[dict] = []
        for t in tags:
            got = load_compiled(title, t)
            if len(got) > len(best):
                best = got
        rows.extend(best)
    return [r for r in rows if r.get("status") in (None, "", "completed")]


def function_scales(functions: list[str], dim: int, n: int = 4096, seed: int = 0) -> dict:
    """f_opt - q05(f(Uniform(domain))) per function: a robust value-range scale."""
    import jax.numpy as jnp
    import jax.random as jrn
    from experiments.bbob_functions import make_suite

    suite = {f.name: f for f in make_suite(dim)}
    scales = {}
    for name in functions:
        fn = suite[name]
        lo, hi = np.array(fn.domain)[:, 0], np.array(fn.domain)[:, 1]
        u = jrn.uniform(jrn.key(seed), (n, dim), minval=jnp.array(lo), maxval=jnp.array(hi))
        vals = np.asarray(fn.f(u))
        scales[name] = float(fn.f_opt - np.percentile(vals, 5))
    return scales


def normalised_regret(rows: list[dict]) -> None:
    """Attach r['nregret'] in place."""
    functions = sorted({r["function"] for r in rows})
    dim = int(fnum(rows[0], "dim", 2)) if rows else 2
    scales = function_scales(functions, dim)
    for r in rows:
        raw = -fnum(r, "final_value")  # f_opt = 0, maximisation
        r["nregret"] = max(raw / scales[r["function"]], REGRET_FLOOR)


def hp_label(rows: list[dict]) -> str:
    keys = sorted(k for k in rows[0] if k not in _META and rows[0].get(k) not in (None, ""))
    return ", ".join(f"{k}={rows[0][k]}" for k in keys) or "(none)"


def best_hp_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """refuter -> its rows at the best (global mean nregret) hp combination."""
    out = {}
    for refuter in sorted({r["refuter"] for r in rows}):
        mine = [r for r in rows if r["refuter"] == refuter]
        hp_keys = sorted(k for k in mine[0] if k not in _META and mine[0].get(k) not in (None, ""))
        combos: dict[tuple, list[dict]] = {}
        for r in mine:
            combos.setdefault(tuple(r.get(k) for k in hp_keys), []).append(r)
        best = min(combos.values(), key=lambda rs: float(np.mean([x["nregret"] for x in rs])))
        out[refuter] = best
    return out
