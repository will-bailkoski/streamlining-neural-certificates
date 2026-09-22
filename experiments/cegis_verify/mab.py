"""
Experiment 1 — CEGIS verifier-only, MAB (sound box-UCB) sampling engine.

    python -m experiments.cegis_verify.mab                 # write array + print sbatch
    python -m experiments.cegis_verify.mab --local 1       # smoke-test the first combo
    python -m experiments.cegis_verify.mab --aggregate     # collate -> compiled.csv

This is the MAB-ONLY end-to-end CEGIS test: MAB is BOTH the refuter (each round it
returns the offending box) AND the sound final verifier (no black-box refuter in the
loop — refuter="none"). The goal is one MAB-VERIFIED certificate per env.

Design choices that make every combo terminate:
  * cert_structure = "bounded_pwl" (ReLU hidden + hard-sigmoid out): MAB's Hoeffding
    statistical bound is only sound when V is bounded to [0,1] — the shared
    config.CERT_STRUCTURE ("relu_pwl") is UNBOUNDED and would be unsound here.
  * small hidden width (4,4 where trainable, see GROUPS): a smoother certificate
    has a smaller L_V, hence a smaller drift-Lipschitz reward_lip = L_V*(L_f+1);
    box count ~ (domain*reward_lip/margin)^d, so a smaller reward_lip is directly
    fewer boxes. train_epsilon is per-env: it MUST be a margin the seed training
    actually reaches (loss == 0), or CEGIS can never terminate.
  * MAB-FEASIBLE env variants (see src/benchmarks/utils.py): the nonlinear defaults
    place the equilibrium / domain where the achievable margin -> 0. The *_mab
    variants size the eq to cover the noise floor and shrink the domain so a bounded
    cert holds a workable margin. 2D linear/stochastic-linear verify quickly; the
    nonlinear ones are sound-but-slow (millions of live boxes -> the big budget + the
    24h SLURM_MAB wall-clock). 3D/4D and the L_f=38 NN-controller pendulum are
    deliberately excluded — MAB's box count is not confidently finite there.

Speedups relied on (already in src/verifiers): the per-iteration sample+bound kernel
is one jit-compiled vmap, the checker/split are pure NumPy, and k=1024 makes each
iteration one fat batch that maps straight onto a GPU (SLURM_MAB requests one; see
its jaxlib-cuda note).
"""

from __future__ import annotations
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_MAB

CAMPAIGN_TITLE = "cegis_verify_mab"

SEEDS = [0, 1, 2]

# Cranked budgets: the nonlinear envs tile to millions of live boxes. max_samples is
# a LIVE-sample (memory) cap recycled on prune, so cumulative samples run far higher.
ENGINE_HP = dict(
    grid_per_dim=32,           # 32x32 initial cells = k lanes stay full through the
                               # bulk phase (a single root box trickles ~10 boxes/iter)
    significance=0.05,
    range_bound=1.0,
    k=1024,                    # fat batch per iteration (GPU-friendly)
    sample_per_select=128,     # draws per selected box per iteration
    max_boxes=10_000_000,
    max_samples=1_000_000_000,
    timeout=43_200,            # 12h per verify call: fires INSIDE the 24h SLURM_MAB
                               # wall so the verdict is recorded (a wall kill leaves
                               # status=running, which the figures silently drop)
    progress_interval=1000,    # heartbeat to stdout — see where a timed-out job was
)

CEGIS = dict(
    cert_structure="bounded_pwl",  # MUST be bounded for MAB soundness (not relu_pwl)
    epsilon=1e-3,
    max_rounds=300,
    sample_count=2000,
    mc_samples=128,                # refuter-objective draws (unused when refuter='none')
    history_len=8,                 # counterexamples fed to the trainer per round
)

# Per-env training margin + width: train_epsilon must be a margin the trainer can
# actually reach (seed loss == 0), else the loss is pinned > 0 and CEGIS can never
# terminate.
#   * linear2D / pendulum_lqr: 4,4 reaches 2e-2 cleanly.
#   * linstoch2D: 2e-2 is NOT reachable (points near the eq/noise-floor boundary
#     cap the margin); 1e-2 trains to zero.
#   * doublewell: 4,4 is pinned at EVERY margin (the ~1% of seeds hugging
#     the eq boundary need a steeper V than 4 hidden units can shape — checked
#     down to eps=2e-3 at 3x epochs); 8,8 trains 2e-2 to zero. The wider net's
#     larger L_V costs boxes, but LipBaB keeps the bound tight.
GROUPS = [
    # 3D/4D are DIMENSION-SCALING probes: MAB's box count grows ~(domain*L/M)^d,
    # so expect the 4D combos to exhaust budget — that boundary is the datapoint.
    dict(env=["linear2D", "linear3D", "linstoch4D", "pendulum_lqr"],
         hidden_layers="4,4", train_epsilon=2e-2),
    dict(env=["linstoch2D"],
         hidden_layers="4,4", train_epsilon=1e-2),
    dict(env=["doublewell", "linear4D", "linstoch3D"],
         hidden_layers="8,8", train_epsilon=2e-2),
]

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_MAB,
        args=[
            dict(
                **g,
                seed=SEEDS,
                **CEGIS,
                engine="mab",
                refuter="none",
                **ENGINE_HP,
            )
            for g in GROUPS
        ],
    )
