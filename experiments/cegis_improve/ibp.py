"""
Experiment 4 — refuter-first CEGIS, autoLiRPA IBP engine (GPU).

Mirror of experiments.cegis_verify.ibp with the whale pre-screen: same engine HP
and per-env-family noise_disc groups (imported from the exp-1 twin), so the two
campaigns differ only by the refuter arm.

    python -m experiments.cegis_improve.ibp --tag bigtest
    python -m experiments.cegis_improve.ibp --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.ibp import NOISE_GROUPS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_improve_ibp"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_GPU_LONG,
           args=groups("ibp", noise_groups=NOISE_GROUPS))
