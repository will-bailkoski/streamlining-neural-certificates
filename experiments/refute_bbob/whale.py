"""
Experiment 2 — refuter benchmark on BBOB functions: Whale Optimisation (WOA).

    python -m experiments.refute_bbob.whale
    python -m experiments.refute_bbob.whale --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_whale"

# HP GRID (a list = a sweep axis): the aggregate finds the best combo per
# function; the winner feeds the refuter-first CEGIS experiments.
REFUTER_HP = dict(
    a_max=[1.5, 2.0, 2.5],   # exploration->exploitation schedule ceiling
    b=[0.5, 1.0, 2.0],       # spiral tightness
    p=[0.25, 0.5, 0.75],     # spiral vs encircle mix
)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="refute_bbob",
        slurm=SLURM_CPU,
        args=dict(
            function=config.BBOB_FUNCTIONS,
            seed=config.SEEDS,
            dim=config.BBOB_DIM,
            refuter="whale",
            budget=8192,
            batch_size=64,
            **REFUTER_HP,
        ),
    )
