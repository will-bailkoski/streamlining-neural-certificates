"""
Experiment 1 — CEGIS verifier-only timing, MILP (Gurobi big-M) engine.

RUNS LOCALLY (laptop / login node): the cluster's Gurobi token server is only
reachable from login nodes (see experiments/common/slurm.py), and the WLS
academic license lives on the laptop (~/gurobi.lic, auto-found). So instead of
sbatch, run the combos in-process:

    python -m experiments.cegis_verify.milp --local --tag bigtest
    python -m experiments.cegis_verify.milp --local 2 --tag bigtest   # first 2 only

Runs are resumable (finished run_ids are skipped on relaunch), so the campaign
can be chipped away at across sessions.

noise_disc grouping: the measured linstoch2D window is nd={4,5,6} verified,
nd=7 timing out at 600s on a frozen valid cert (results/milp_noise_disc) —
[4, 6, 8] brackets the window and re-probes 8 at the cranked time_limit as the
upper edge. 3D/4D cells grow as disc^dim; the window there was measured EMPTY
(nd<=3 phantom CEs, nd=4 timeout), so those groups carry the honest attempts.
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
