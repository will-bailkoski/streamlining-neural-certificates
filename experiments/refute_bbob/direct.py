"""
Experiment 2 — refuter benchmark on BBOB functions: DIRECT (dividing rectangles).

    python -m experiments.refute_bbob.direct
    python -m experiments.refute_bbob.direct --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "refute_bbob_direct"

# HP GRID (a list = a sweep axis).
REFUTER_HP = dict(
    po_eps=[1e-5, 1e-4, 1e-3],
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
            refuter="direct",
            budget=8192,
            batch_size=64,
            **REFUTER_HP,
        ),
    )
