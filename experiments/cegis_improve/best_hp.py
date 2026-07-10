"""
Best hyperparameters for experiment 4 (refuter-first CEGIS), chosen from the
experiment 1 (per-engine verification) and experiment 3 (per-refuter refutation)
results.

Engine hyperparameters are IMPORTED from the experiment-1 launchers so the two
campaigns mirror each other by construction: a cegis_improve run differs from its
cegis_verify twin ONLY by the refuter pre-screen (and the compared history_len).
Refuter hyperparameters are the version-controlled record of the BBOB hpsweep
winners (results/analysis/bbob_hpsweep, 2026-07-03).
"""

from __future__ import annotations

from experiments.cegis_verify.alpha_crown import ENGINE_HP as _ALPHA_HP
from experiments.cegis_verify.crown import ENGINE_HP as _CROWN_HP
from experiments.cegis_verify.ibp import ENGINE_HP as _IBP_HP
from experiments.cegis_verify.mab import ENGINE_HP as _MAB_HP
from experiments.cegis_verify.mc import ENGINE_HP as _MC_HP
from experiments.cegis_verify.milp import ENGINE_HP as _MILP_HP
from experiments.cegis_verify.smt import ENGINE_HP as _SMT_HP

# Per-engine loop-closer hyperparameters (bare flags -> engine) — the exp-1
# bigtest configs, verbatim. noise_disc is NOT here: it is an env-family axis
# (config.noise_groups), mirrored from each verify launcher's NOISE_GROUPS.
ENGINE = {
    "smt": dict(_SMT_HP),
    "milp": dict(_MILP_HP),      # full Gurobi license -> runs --local (laptop)
    "mab": dict(_MAB_HP),
    "mc": dict(_MC_HP),
    "ibp": dict(_IBP_HP),
    "crown": dict(_CROWN_HP),
    "alpha-crown": dict(_ALPHA_HP),
}

# Per-refuter pre-screen hyperparameters. budget/batch_size become the runner's
# core --budget/--batch_size; the rest are routed to the refuter (r_ prefix).
# Tuned values = the BBOB hpsweep winners (results/analysis/bbob_hpsweep,
# 2026-07-03: best mean normalised regret over the 5 BBOB functions x 3 seeds).
REFUTER = {
    "random": dict(budget=8192, batch_size=256),
    "grid": dict(budget=8192, batch_size=256),
    "gradient": dict(budget=8192, batch_size=256, lr=0.5, decay=0.95),
    "whale": dict(budget=8192, batch_size=64, a_max=1.5, b=0.5, p=0.75),
    "adalip": dict(budget=8192, batch_size=64, pool_mult=16, alpha=0.2),
    "direct": dict(budget=8192, batch_size=64, po_eps=1e-3),
}

# Which refuters to pre-screen with. The bigtest runs the whale arm only: the
# verifier-only baseline is the cegis_verify campaign itself (same runner, same
# engine HP, same history_len — cegis_pipelines pools both prefixes and keys the
# pipeline off the per-row `refuter` column, so no duplicate 'none' arm needed).
ACTIVE_REFUTERS = ["whale"]

# Counterexamples fed to the trainer per round — MATCHED to the exp-1 baseline
# (config.CEGIS history_len=1; the mab launcher pins its own 8) so refuter-first
# vs verifier-only compares sample-fair. Make it a list to sweep instead.
HISTORY_LEN = [1]
