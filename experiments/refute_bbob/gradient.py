"""
Experiment 2 — refuter benchmark on BBOB functions: multi-start gradient ascent.

    python -m experiments.refute_bbob.gradient
    python -m experiments.refute_bbob.gradient --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_gradient"

# HP GRID (a list = a sweep axis).
REFUTER_HP = dict(
    lr=[0.01, 0.1, 0.5],
    decay=[0.95, 0.99, 1.0],
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
            refuter="gradient",
            budget=8192,
            batch_size=256,
            **REFUTER_HP,
        ),
    )
