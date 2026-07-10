"""
Experiment 2 — refuter benchmark on BBOB functions: random search.

Sweeps the shared BBOB functions x seeds against this refuter, over its
hyperparameter grid (a list = a sweep axis). Supporting evidence that the
refuters are strong black-box searchers, which motivates using them to pre-screen
the CEGIS loop (experiment 4).

    python -m experiments.refute_bbob.random
    python -m experiments.refute_bbob.random --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_random"

REFUTER_HP: dict = {}  # random search has no extra hyperparameters

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="refute_bbob",
        slurm=SLURM_CPU,
        args=dict(
            function=config.BBOB_FUNCTIONS,
            seed=config.SEEDS,
            dim=config.BBOB_DIM,
            refuter="random",
            budget=8192,
            batch_size=256,
            **REFUTER_HP,
        ),
    )
