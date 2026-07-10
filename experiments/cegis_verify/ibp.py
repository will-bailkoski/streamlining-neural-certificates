"""
Experiment 1 — CEGIS verifier-only timing, autoLiRPA IBP engine (GPU).

The autoLiRPA bound methods bin continuous noise into noise_disc cells per
dimension (a sound over-approximation of E[V]); noise_disc is this experiment's
"discretisation error" axis. It is swept PER ENV FAMILY (config.noise_groups):
the linear* envs ignore it (discrete noise, exact), and the cell count is
disc^dim so the ceiling drops with dimension — 8 on 2D is 64 cells per bound
call, already the practical edge.

    python -m experiments.cegis_verify.ibp --tag bigtest
    python -m experiments.cegis_verify.ibp --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_verify_ibp"

# Cranked budgets (bigtest): the old 200k box cap fired by depth 5-7 even on
# linear 3D/4D (verify_matrix, split=all era). timeout fires INSIDE the SLURM
# wall so the verdict is always recorded (a wall kill leaves status=running,
# which the figures silently drop).
ENGINE_HP = dict(
    split="longest",     # split=all at depth 18 was the old limiter (2026-07-03)
    max_depth=60,
    max_boxes=2_000_000,
    timeout=39_600,      # 11h per verify call < the 12h SLURM_GPU_LONG wall
    # NOTE: the engine's bound-call batch knob is also called batch_size, but that
    # name is a core runner flag (refuter batch) — passing it here would never
    # reach the engine. The engine default (2048) applies.
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
                engine="ibp",
                refuter="none",
                **ENGINE_HP,
            )
            for ng in NOISE_GROUPS
        ],
    )
