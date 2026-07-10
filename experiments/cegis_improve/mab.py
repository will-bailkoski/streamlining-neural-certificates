"""
Experiment 4 — refuter-first CEGIS, MAB (sound box-UCB) engine.

Mirror of experiments.cegis_verify.mab with the whale pre-screen: everything —
bounded_pwl cert (REQUIRED for MAB's statistical soundness; the shared
config.CERT_STRUCTURE relu_pwl is unbounded and would be unsound here), the
per-env width/train_epsilon GROUPS, the cranked budgets, history_len=8, and
SLURM_MAB — is imported from the exp-1 twin, so the whale arm is the only
difference. (This launcher previously inherited relu_pwl + SLURM_CPU from the
generic groups() path — both wrong for MAB.)

    python -m experiments.cegis_improve.mab --tag bigtest
    python -m experiments.cegis_improve.mab --local 1
"""

from __future__ import annotations
from experiments.cegis_improve import best_hp
from experiments.cegis_verify.mab import CEGIS, ENGINE_HP, GROUPS, SEEDS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_MAB

CAMPAIGN_TITLE = "cegis_improve_mab"

_whale = dict(best_hp.REFUTER["whale"])
_core = {k: _whale.pop(k) for k in ("budget", "batch_size")}  # runner core flags
_whale_flags = {**_core, **{f"r_{k}": v for k, v in _whale.items()}}

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_MAB,
        args=[
            dict(
                **g,
                seed=SEEDS,
                **CEGIS,
                engine="mab",
                refuter="whale",
                **ENGINE_HP,
                **_whale_flags,
            )
            for g in GROUPS
        ],
    )
