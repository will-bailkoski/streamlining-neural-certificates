"""
Experiment 1 — CEGIS verifier-only timing, SMT (Z3) engine.

Sweeps the shared envs x seeds (experiments.common.config) against the SMT engine
as the loop closer. noise_disc is grouped per env family (config.noise_groups):
the discrete-noise linear* envs ignore it entirely, so they get a single value.
Prior evidence (results/verify_matrix): Z3 timed out on EVERYTHING at 120s, even
discrete-noise 2D — the bigtest crank to 600s is expected to stay a negative
result, but makes it an honest one.

    python -m experiments.cegis_verify.smt --tag bigtest
    python -m experiments.cegis_verify.smt --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU_LONG

CAMPAIGN_TITLE = "cegis_verify_smt"  # preset, not editable

ENGINE_HP = dict(
    timeout_ms=600_000,   # Z3 per-call cap: 120s decided nothing, give it 10 min
)

# 3D/4D cells grow as disc^dim (512 at disc=8 on 3D) — Z3 already times out at
# the small discs, so the high-dim groups keep one moderate value.
NOISE_GROUPS = config.noise_groups(disc_2d=[4, 8], disc_3d=4, disc_4d=2,
                                   disc_discrete=4)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_CPU_LONG,
        args=[
            dict(
                **ng,
                seed=config.SEEDS,
                **config.CEGIS,
                engine="smt",
                refuter="none",
                **ENGINE_HP,
            )
            for ng in NOISE_GROUPS
        ],
    )
