"""
Experiment 2 — refuter benchmark on BBOB functions: grid search.

    python -m experiments.refute_bbob.grid
    python -m experiments.refute_bbob.grid --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_grid"

REFUTER_HP: dict = {}

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="refute_bbob",
        slurm=SLURM_CPU,
        args=dict(
            function=config.BBOB_FUNCTIONS,
            seed=config.SEEDS,
            dim=config.BBOB_DIM,
            refuter="grid",
            budget=8192,
            batch_size=256,
            **REFUTER_HP,
        ),
    )
