"""
Experiment 4 — refuter-first CEGIS, autoLiRPA CROWN engine (GPU).

Mirror of experiments.cegis_verify.crown with the whale pre-screen (same engine
HP + noise groups, imported from the exp-1 twin).

    python -m experiments.cegis_improve.crown --tag bigtest
    python -m experiments.cegis_improve.crown --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.crown import NOISE_GROUPS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_improve_crown"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_GPU_LONG,
           args=groups("crown", noise_groups=NOISE_GROUPS))
