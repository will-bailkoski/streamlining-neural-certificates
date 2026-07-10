"""
Phase 1 of experiments.pendulum_anytime — generate the VERIFIED pendulum_lqr
certificates the anytime experiment refutes, one per seed, in PARALLEL on the GPU
partition (CEGIS with the CROWN verifier).

The anytime figure needs a valid supermartingale per seed; those certs are produced
ONCE here (a SLURM GPU array, one seed per task) and then SCRAPED by the trace step,
which only recomputes the cheap bound-prop traces. Splitting cert-gen out is what
makes the experiment cluster-friendly: the heavy, GPU-bound CEGIS runs fan across
GPUs instead of serialising inside one process, and each task gets a DEDICATED GPU so
JAX (env + training) and torch (CROWN bound propagation) don't fight over VRAM — the
exact contention that OOM'd the old inline-generation path on a shared node.

    python -m experiments.pendulum_certs                 # write the array + print sbatch
    python -m experiments.pendulum_certs --local 1        # generate just seed 0 locally
    python -m experiments.pendulum_certs --aggregate      # collate -> compiled.csv

Certs land under results/pendulum_certs/<tag>/runs/; the trace step scrapes them with
its default --cert-campaign pendulum_certs (its scan is recursive, so the launch tag
is found automatically). Then run phase 2:

    python -m experiments.pendulum_anytime --emit-sbatch  # writes its GPU submit.sbatch
    sbatch --dependency=afterok:<this array's jobid> results/pendulum_anytime/submit.sbatch
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "pendulum_certs"

# The exact certificate the anytime experiment expects: relu_pwl 8x8 (lirpa-encodable,
# linear output), verified by CROWN with the noise discretisation fixed at 8. These
# mirror experiments.pendulum_anytime's CERTSPEC / WIDTH / NOISE_DISC / EPS / TRAIN_EPS
# so a scraped cert is identical to one the inline fallback would have produced.
# split="longest" (not the engine default "all") — the proven setting that keeps the
# per-round CROWN verify from exploding 2**d children per depth (see cegis_verify/crown).
CERT_HP = dict(
    cert_structure="relu_pwl",
    hidden_layers="8,8",
    epsilon=1e-3,
    train_epsilon=2e-2,
    noise_disc=8,
    split="longest",
    max_rounds=40,
    max_depth=40,
    max_boxes=200_000,
)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cegis",
        slurm=SLURM_GPU,
        args=dict(
            env="pendulum_lqr",
            seed=config.SEEDS,     # [0, 1, 2] -> one array task per seed, one GPU each
            engine="crown",
            refuter="none",
            **CERT_HP,
        ),
    )
