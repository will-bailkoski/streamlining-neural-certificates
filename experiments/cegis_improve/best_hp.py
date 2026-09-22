"""
Hyperparameters for experiment 4 (refuter-first CEGIS, Chapter 7).

Engine hyperparameters are IMPORTED from the experiment-1 launchers so the two
campaigns mirror each other by construction: a cegis_improve run differs from its
cegis_verify twin ONLY by the refuter pre-screen. Refuter hyperparameters are the
BBOB tuning winners (Table tab:bbob-best: best mean normalised regret over the
five BBOB functions x three seeds).
"""

from __future__ import annotations

from experiments.cegis_verify.alpha_crown import ENGINE_HP as _ALPHA_HP
from experiments.cegis_verify.crown import ENGINE_HP as _CROWN_HP
from experiments.cegis_verify.ibp import ENGINE_HP as _IBP_HP

# Per-engine loop-closer hyperparameters (bare flags -> engine).
ENGINE = {
    "ibp": dict(_IBP_HP),
    "crown": dict(_CROWN_HP),
    "alpha-crown": dict(_ALPHA_HP),
}

# Per-refuter pre-screen hyperparameters. budget/batch_size become the runner's
# core --budget/--batch_size; the rest are routed to the refuter (r_ prefix).
REFUTER = {
    "random": dict(budget=8192, batch_size=256),
    "grid": dict(budget=8192, batch_size=256),
    "gradient": dict(budget=8192, batch_size=256, lr=0.5, decay=0.95),
    "whale": dict(budget=8192, batch_size=64, a_max=1.5, b=0.5, p=0.75),
    "adalip": dict(budget=8192, batch_size=64, pool_mult=16, alpha=0.2),
    "direct": dict(budget=8192, batch_size=64, po_eps=1e-3),
}

# The pre-screen arms of the Chapter 7 campaign: random (the uninformed extreme)
# and whale (the BBOB winner). The verifier-only baseline is the cegis_verify
# campaign itself (same runner, engine HP and history_len); cegis_pipelines pools
# both prefixes and keys the pipeline off the per-row `refuter` column.
ACTIVE_REFUTERS = ["random", "whale"]

# Counterexamples fed to the trainer per round, matched to the exp-1 baseline
# (config.CEGIS history_len=1) so refuter-first vs verifier-only is sample-fair.
HISTORY_LEN = [1]
