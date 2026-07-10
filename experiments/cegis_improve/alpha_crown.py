"""
Experiment 4 — refuter-first CEGIS, autoLiRPA alpha-CROWN engine (GPU).

Mirror of experiments.cegis_verify.alpha_crown with the whale pre-screen (same
engine HP + noise groups, imported from the exp-1 twin).

    python -m experiments.cegis_improve.alpha_crown --tag bigtest
    python -m experiments.cegis_improve.alpha_crown --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.alpha_crown import NOISE_GROUPS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_improve_alpha_crown"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_GPU_LONG,
           args=groups("alpha-crown", noise_groups=NOISE_GROUPS))
