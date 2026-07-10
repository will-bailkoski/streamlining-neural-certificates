"""
Experiment 1 — CEGIS verifier-only timing, autoLiRPA CROWN engine (GPU).

CROWN is the middle rung of the tightness/cost ladder (ibp -> crown ->
alpha-crown). noise_disc is swept per env family (see ibp.py / config.noise_groups
for the rationale).

    python -m experiments.cegis_verify.crown --tag bigtest
    python -m experiments.cegis_verify.crown --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_verify_crown"

ENGINE_HP = dict(
    # split=longest + deep: split=all at depth 18 was the old limiter — with the
    # fix crown VERIFIES the doublewell cert in 204 boxes at depth 10 (2026-07-03),
    # while split=all explodes 2**d children per level and stalls.
    split="longest",
    max_depth=60,
    max_boxes=200_000_000,   # deliberately 100x ibp/alpha: crown's bounds are tight
                             # enough that the population usually stays small; the
                             # timeout is the binding budget, not the box cap
    timeout=39_600,          # 11h per verify call < the 12h SLURM_GPU_LONG wall
    # NOTE: batch_size is a core runner flag (refuter batch) — it cannot be set
    # as an engine hp from here; the engine default (2048) applies.
)

NOISE_GROUPS = config.noise_groups(disc_2d=[1, 4, 8], disc_3d=[1, 3], disc_4d=[1, 2])

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_GPU_LONG,
        args=[
            dict(
                **ng,
                seed=config.SEEDS,
                **config.CEGIS,
                engine="crown",
                refuter="none",
                **ENGINE_HP,
            )
            for ng in NOISE_GROUPS
        ],
    )
