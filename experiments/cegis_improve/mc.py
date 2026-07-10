"""
Experiment 4 — refuter-first CEGIS, Monte-Carlo (statistical) engine.

Mirror of experiments.cegis_verify.mc with the whale pre-screen. Like the exp-1
twin it must run on the BOUNDED certificate (the Lipschitz mean->max bound needs
V in [0, range_bound]) — the override is imported from there.

    python -m experiments.cegis_improve.mc --tag bigtest
    python -m experiments.cegis_improve.mc --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.mc import CEGIS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "cegis_improve_mc"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_CPU,
           args=groups("mc", cegis=CEGIS))
