"""
autoLiRPA bound-method x split benchmark — GPU SLURM array.

Standalone verifier characterisation (NOT CEGIS): each task trains one cert for
an (env, hidden_layers, seed) and benchmarks EVERY (bound method x split) on it,
so those two axes live INSIDE the job and don't multiply the array. The array
axes are the ones worth parallelising: env (=> dimension) and network size.

    python -m experiments.lirpa_bench
    python -m experiments.lirpa_bench --local 2
    python -m experiments.lirpa_bench --aggregate     # one row per method x split

torch runs on the GPU (SLURM_GPU); JAX stays on CPU for the one-off cert training.
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "lirpa_bench"

BENCH = dict(
    hidden_layers=["8,8", "16,16", "32,32", "8,8,8"],  # network-size axis
    cert_structure=config.CERT_STRUCTURE,
    epsilon=1e-3,
    sample_count=2000,
    noise_disc=1,
    batch_size=4096,        # boxes per compute_bounds — the GPU throughput knob
    max_depth=20,
    max_boxes=500_000,
)

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="lirpa_bench",
        slurm=SLURM_GPU,
        iters_aggregate=True,   # collate one row per (method, split)
        args=dict(
            env=config.ENVS,
            seed=config.SEEDS,
            **BENCH,
        ),
    )
