"""
Experiment 2 — refuter benchmark on BBOB functions: AdaLIPO.

    python -m experiments.refute_bbob.adalip
    python -m experiments.refute_bbob.adalip --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_adalip"

# HP GRID (a list = a sweep axis).
REFUTER_HP = dict(
    pool_mult=[4, 8, 16],
    alpha=[0.01, 0.05, 0.2],
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
            refuter="adalip",
            budget=8192,
            batch_size=64,
            **REFUTER_HP,
        ),
    )
