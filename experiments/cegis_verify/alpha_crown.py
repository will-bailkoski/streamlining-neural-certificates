"""
Experiment 1 — CEGIS verifier-only timing, autoLiRPA alpha-CROWN engine (GPU).

alpha-CROWN is the tightest (and most expensive) rung of the ladder. The engine
name has a hyphen (`alpha-crown`); only the module/campaign use the underscore.
noise_disc is swept per env family (see ibp.py / config.noise_groups).

    python -m experiments.cegis_verify.alpha_crown --tag bigtest
    python -m experiments.cegis_verify.alpha_crown --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_verify_alpha_crown"

ENGINE_HP = dict(
    split="longest",     # split=all at depth 18 was the old limiter (2026-07-03)
    max_depth=60,
    max_boxes=2_000_000,
    timeout=39_600,      # 11h per verify call < the 12h SLURM_GPU_LONG wall
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
                engine="alpha-crown",
                refuter="none",
                **ENGINE_HP,
            )
            for ng in NOISE_GROUPS
        ],
    )
