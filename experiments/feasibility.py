"""
Feasibility probe: find the EASIEST verifiable config per environment.

Not a timing experiment — a reassurance/frontier sweep. For each env it searches
the easy direction of every knob and caps the verifier so it CONCLUDES fast (a
quick "inconclusive" beats a 30-minute grind), so you learn empirically:

  * the LARGEST verify epsilon that still verifies   (bigger margin = easier),
  * the FEWEST neurons that still verify              (smaller net = fewer ReLU
                                                       branches to bound),
  * the SMALLEST noise_disc that still verifies       (just-tight-enough bound).

train_epsilon is fixed >= every verify epsilon, so the cert is trained to a real
margin and the verifier checks an easier threshold (the train-hard / verify-easy
trick that helps bound propagation). Single seed — this is existence, not stats.

    python -m experiments.feasibility                 # write the array + print sbatch
    python -m experiments.feasibility --local 3        # try the first 3 locally
    python -m experiments.feasibility --aggregate      # per-env frontier summary

Run it on the H100 partition (crown). Once one config per env verifies, the hard
settings are merely slow, not impossible. Edit the ladders / caps below to taste.
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "feasibility"

# Easy -> hard ladders (a list = a swept axis).
LADDER = dict(
    epsilon=[1e-2, 3e-3, 1e-3],          # verify threshold — find the MAX that verifies
    hidden_layers=["4,4", "8,8"],         # find the MIN neurons that verify
    noise_disc=[1, 3],                    # find the MIN discretisation that verifies
)
TRAIN_EPSILON = 2e-2                       # >= every verify epsilon (train a real margin).
# Empirically (local CPU): with TRAIN_EPSILON=2e-2 and enough rounds, mc verified an
# 8x8 cert on ALL FOUR envs at verify eps=1e-2 (linstoch2D 7 rnds, linear3D 13,
# vanderpol 16, pendulum_lqr 8). So the easy regime is feasible everywhere; if a
# crown run here is INCONCLUSIVE it's the bound's slack, not an impossible problem —
# swap engine="mc" for a fast statistical second opinion.

# Verifier caps tuned to CONCLUDE quickly, not to grind toward a huge box budget.
CAPS = dict(
    max_depth=20,
    max_boxes=2_000_000,
    max_rounds=60,
    sample_count=2000,
    mc_samples=16,
)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_GPU,
        args=dict(
            env=config.ENVS,
            seed=0,                        # existence probe, not a statistics run
            engine="crown",               # covers every env; switch per your focus
            refuter="none",
            cert_structure=config.CERT_STRUCTURE,
            history_len=1,
            train_epsilon=TRAIN_EPSILON,
            **LADDER,
            **CAPS,
        ),
    )
