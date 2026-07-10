"""
Experiment 4 — refuter-first CEGIS, SMT (Z3) engine.

Mirror of experiments.cegis_verify.smt with the whale pre-screen (same engine HP
+ noise groups, imported from the exp-1 twin). If Z3 stays undecidable here, the
whale arm shows how far a cheap refuter alone carries the loop before the
verifier stalls it.

    python -m experiments.cegis_improve.smt --tag bigtest
    python -m experiments.cegis_improve.smt --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.smt import NOISE_GROUPS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU_LONG

CAMPAIGN_TITLE = "cegis_improve_smt"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_CPU_LONG,
           args=groups("smt", noise_groups=NOISE_GROUPS))
