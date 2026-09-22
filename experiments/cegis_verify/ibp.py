"""
Experiment 1 — CEGIS verifier-only timing, autoLiRPA IBP engine (GPU).

The verifier-only baseline arm of the Chapter 7 campaign (config.STREAMLINE_*):
LinSwitch, DoubleWell and Pendulum over five seeds, at the engine's default noise
discretisation.

    python -m experiments.cegis_verify.ibp --tag final
    python -m experiments.cegis_verify.ibp --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_verify_ibp"

# timeout fires INSIDE the SLURM wall so the verdict is always recorded (a wall
# kill leaves status=running, which the figures silently drop).
ENGINE_HP = dict(
    split="longest",     # longest-edge region splitting
    max_depth=60,
    max_boxes=2_000_000,
    timeout=39_600,      # 11h per verify call < the 12h SLURM_GPU_LONG wall
    # NOTE: the engine's bound-call batch knob is also called batch_size, but that
    # name is a core runner flag (refuter batch) — passing it here would never
    # reach the engine. The engine default (2048) applies.
)


if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_GPU_LONG,
        args=dict(
            env=config.STREAMLINE_ENVS,
            seed=config.STREAMLINE_SEEDS,
            **config.CEGIS,
            engine="ibp",
            refuter="none",
            **ENGINE_HP,
        ),
    )
