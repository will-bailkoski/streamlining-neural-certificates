"""
Experiment 1 — CEGIS verifier-only timing, Monte-Carlo (statistical) engine.

`mc` is the SOUND non-adaptive statistical verifier (registered as `montecarlo`;
the runner aliases mc -> montecarlo): a global uniform-sample mean of the drift
exceedance, converted to a worst-case ceiling via the drift-Lipschitz constant
(method "A"). Same 1-significance guarantee as mab, but non-adaptive — the
mean->max conversion pays a d-dimensional volume->height penalty, so its
certified ceiling shrinks only like n^{-1/(2(d+1))} and it CANNOT certify at a
zero tolerance. It is the sound-but-expensive baseline that motivates mab's
adaptivity; `stats["certified_sup_drift"]` (-> v_certified_sup_drift in
compiled.csv) records the ceiling per round regardless of verdict.

    python -m experiments.cegis_verify.mc
    python -m experiments.cegis_verify.mc --local 2
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_CPU

CAMPAIGN_TITLE = "cegis_verify_mc"

ENGINE_HP = dict(
    n_states=400_000,  # the certified ceiling shrinks only like
                       # n^{-1/(2(d+1))}, so this buys tightness, not miracles
    significance=0.05,
    range_bound=1.0,   # V in [0, 1] for bounded_pwl -> drift exceedance range
    verify_tol=0.0,    # tolerance for a verified verdict (0 -> report the ceiling)
)

# The Lipschitz mean->max bound needs V bounded to [0, range_bound]; the shared
# config uses the UNBOUNDED relu_pwl, so override to bounded_pwl (exactly as mab
# does — the same soundness requirement).
CEGIS = {**config.CEGIS, "cert_structure": "bounded_pwl"}

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_CPU,
        args=dict(
            env=config.ENVS,
            seed=config.SEEDS,
            **CEGIS,
            engine="mc",
            refuter="none",
            **ENGINE_HP,
        ),
    )
