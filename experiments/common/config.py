"""
Single source of truth for the axes shared across every experiment launcher.

Edit HERE to change what the whole campaign suite runs on — the per-engine and
per-refuter launchers import these so the env list, seed count and base CEGIS
settings stay consistent across experiments. A list value is a sweep axis; a
scalar is fixed.
"""

from __future__ import annotations

# The one certificate structure the experiments enforce (the soundness bridge in
# src/certificates/structures.py). relu_pwl = ReLU hidden + linear output, which
# every engine — including the autoLiRPA bound methods — can encode.
CERT_STRUCTURE = "relu_pwl"

# Systems under test (keys into src/benchmarks/utils.py::DIRECTORY). Spans the
# assumption axes: switched-linear with discrete noise (exact everywhere) and
# additive-noise linear (exact with noise binning) each at 2D/3D/4D (dimension
# scaling), a polynomial nonlinear (doublewell — exact in smt/milp), and a
# transcendental pendulum (sin -> smt/milp abstain; sampling + lirpa handle it).
ENVS = [
    "linear2D", "linear3D", "linear4D",
    "linstoch2D", "linstoch3D", "linstoch4D",
    "doublewell", "pendulum_lqr",
]

# Seeds each combination is repeated over (verification campaigns and the BBOB /
# faux-certificate refuter benchmarks).
SEEDS = [0, 1, 2]

# Chapter 7 (Streamlining CEGIS, Table tab:cegis-speedup): the end-to-end campaign
# runs the three LiRPA engines on the three systems they certify, comparing
# verifier-only, random pre-screen and whale pre-screen over five seeds
# (3 engines x 3 systems x 3 strategies x 5 seeds = 135 arms).
STREAMLINE_ENVS = ["linear2D", "doublewell", "pendulum_lqr"]
STREAMLINE_SEEDS = [0, 1, 2, 3, 4]

# Noise-model split of ENVS, for per-env-family noise_disc axes. The linear*
# envs use a discrete Bernoulli mode switch — exact regardless of noise_disc
# (the flag is ignored), so sweeping disc there only reruns identical combos.
# For the continuous-noise envs the cell count grows as disc^dim, so the
# feasible disc ceiling drops with dimension.
ENVS_DISCRETE = ["linear2D", "linear3D", "linear4D"]
ENVS_CONT_2D = ["linstoch2D", "doublewell", "pendulum_lqr"]
ENVS_CONT_3D = ["linstoch3D"]
ENVS_CONT_4D = ["linstoch4D"]


def noise_groups(disc_2d, disc_3d, disc_4d, disc_discrete=1):
    """Arg groups giving each env family its own noise_disc axis (list = sweep).

    Together the groups cover exactly ENVS, but each dimension only sweeps the
    noise binning it can afford (disc^dim cells per drift encoding/bound call).
    """
    return [
        dict(env=ENVS_DISCRETE, noise_disc=disc_discrete),
        dict(env=ENVS_CONT_2D, noise_disc=disc_2d),
        dict(env=ENVS_CONT_3D, noise_disc=disc_3d),
        dict(env=ENVS_CONT_4D, noise_disc=disc_4d),
    ]

# Black-box optimisation benchmarks for the refuter experiments — all five are
# BBOB/COCO noiseless-suite families (f3, f8, f17, f20, f21; core formulas, see
# experiments/bbob_functions.py): regular-multimodal, curved valley, irregular
# ripples, deceptive-boundary, needle-among-distractors.
BBOB_FUNCTIONS = ["rastrigin", "rosenbrock", "schaffers_f7", "schwefel", "gallagher"]
BBOB_DIM = 2

# Base CEGIS settings shared by experiment 1 (cegis_verify) and experiment 4
# (cegis_improve). A list on any key makes it a sweep axis.
CEGIS = dict(
    cert_structure=CERT_STRUCTURE,
    hidden_layers="8,8",
    epsilon=1e-3,
    train_epsilon=2e-2,  # train a bigger margin than we verify
    max_rounds=300,
    sample_count=2000,
    mc_samples=16,   # successor draws per refuter-objective drift estimate (m = 16);
                     # a refuter counterexample is re-checked with a fresh n=4096
                     # estimate (--refuter_recheck_n) before it is accepted.
    history_len=1,  # counterexamples fed to the trainer per round (fairness control)
)
