"""
Experiment 1 — CEGIS verifier-only timing, MILP (Gurobi big-M) engine.

Runs wherever a Gurobi licence is available; in the thesis this was a local
desktop, so the combos run in-process instead of via sbatch:

    python -m experiments.cegis_verify.milp --local --tag final
    python -m experiments.cegis_verify.milp --local 2 --tag final   # first 2 only

Runs are resumable (finished run_ids are skipped on relaunch), so the campaign
can be chipped away at across sessions.

noise_disc grouping: [4, 6, 8] brackets the linstoch2D verification window
measured by experiments.milp_noise_disc (phantom counterexamples below b = 4).
3D/4D cells grow as disc^dim, so those groups carry a few affordable values.
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "cegis_verify_milp"

ENGINE_HP = dict(
    time_limit=600.0,    # Gurobi per-call wall-clock cap (seconds); the early-exit
                         # callback usually decides long before it
)

NOISE_GROUPS = config.noise_groups(disc_2d=[4, 6, 8], disc_3d=4, disc_4d=[2, 4],
                                   disc_discrete=4)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_CPU,
        args=[
            dict(
                **ng,
                seed=config.SEEDS,
                **config.CEGIS,
                engine="milp",
                refuter="none",
                **ENGINE_HP,
            )
            for ng in NOISE_GROUPS
        ],
    )
