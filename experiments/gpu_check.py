"""
GPU pre-flight SLURM array — submit this FIRST, before the real experiments.

Same launcher flow as everything else (it just uses the GPU runner). Each array
task lands on an allocated GPU and verifies JAX (jit) and torch actually run on it
(device placement + a CPU-vs-GPU speedup). Set REPLICAS to at least your GPU
throttle so one wave probes every GPU you'll use.

    python -m experiments.gpu_check                 # write the array + print sbatch
    python -m experiments.gpu_check --local 1        # run one probe locally
    python -m experiments.gpu_check --aggregate      # collate -> compiled.csv + PASS summary

Workflow on the cluster:
    1. edit experiments/common/slurm.py (partition/qos/account/gres for your cluster)
       and the venv/module-load lines in the generated submit.sbatch,
    2. sbatch the printed gpu_check submit.sbatch,
    3. --aggregate and confirm every row passed (jax_gpu=1, torch_cuda=1),
    4. THEN sbatch a real experiment's submit.sbatch (cegis_verify/* etc.).
"""

from __future__ import annotations
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "gpu_check"

# One probe per array slot. Make this >= your GPU throttle so a single wave covers
# every GPU you'll actually use for the experiments.
REPLICAS = list(range(4))

if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="gpu_check",
        slurm=SLURM_GPU,
        args=dict(replica=REPLICAS, size=4096, iters=200),
    )
