"""
Experiment 4 — refuter-first CEGIS, MILP (Gurobi big-M) engine.

Mirror of experiments.cegis_verify.milp with the whale pre-screen (same engine
HP + noise groups, imported from the exp-1 twin). Like its twin this RUNS
LOCALLY — the Gurobi WLS license lives on the laptop and the cluster token
server is login-node-only (experiments/common/slurm.py):

    python -m experiments.cegis_improve.milp --local --tag bigtest
    python -m experiments.cegis_improve.milp --local 2 --tag bigtest
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.cegis_verify.milp import NOISE_GROUPS
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "cegis_improve_milp"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_CPU,
           args=groups("milp", noise_groups=NOISE_GROUPS))
