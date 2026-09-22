"""
Experiment 1 — CEGIS verifier-only timing, autoLiRPA CROWN engine (GPU).

CROWN is the middle rung of the tightness/cost ladder (ibp -> crown ->
alpha-crown). Verifier-only baseline arm of the Chapter 7 campaign (see ibp.py).

    python -m experiments.cegis_verify.crown --tag final
    python -m experiments.cegis_verify.crown --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_verify_crown"

ENGINE_HP = dict(
    split="longest",         # longest-edge region splitting
    max_depth=60,
    max_boxes=200_000_000,   # deliberately 100x ibp/alpha: crown's bounds are tight
                             # enough that the population usually stays small; the
                             # timeout is the binding budget, not the box cap
    timeout=39_600,          # 11h per verify call < the 12h SLURM_GPU_LONG wall
    # NOTE: batch_size is a core runner flag (refuter batch) — it cannot be set
    # as an engine hp from here; the engine default (2048) applies.
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
            engine="crown",
            refuter="none",
            **ENGINE_HP,
        ),
    )
