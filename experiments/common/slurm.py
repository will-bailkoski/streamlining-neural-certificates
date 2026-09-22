"""
SLURM resource profiles, in one place so every experiment requests the same.

Two profiles: a CPU default for the symbolic / sampling engines and refuters,
and a GPU profile for the autoLiRPA bound methods (torch runs on the GPU). A key
left as None omits that `#SBATCH` line. Edit a profile here and every launcher
that uses it picks up the change.
"""

from __future__ import annotations

# `setup` lines are injected into submit.sbatch verbatim (after the cd into the
# repo, before the python call). EDIT the venv path below for your cluster.
#
# MILP needs a Gurobi licence visible to the process. In the thesis the MILP
# campaigns ran locally (`--local`) because the licence was not reachable from
# the cluster's compute nodes; a WLS licence (GRB_LICENSE_FILE) also works in a job.
#
# GPU (ibp/crown/alpha-crown): do NOT load a CUDA module — jax[cuda12] and torch
# bundle their own CUDA/CuDNN; a system CuDNN shadows jax's via LD_LIBRARY_PATH and
# breaks the GPU backend. Unload it defensively.
_VENV = "source venv/bin/activate"  # relative to the repo root
_SETUP_CPU = [
    _VENV,
]
_SETUP_GPU = [
    "module unload cuda 2>/dev/null || true",
    _VENV,
    # JAX and torch/autoLiRPA share the one allocated GPU (the env + cert training run
    # on JAX; the bound propagation on torch). Stop JAX preallocating ~75% of VRAM on
    # first use (its default) — that starves torch and OOMs, and collides with a co-tenant
    # on a shared node; let it grow on demand instead. Harmless for the pure-JAX mab job
    # (it just grows to what it needs). An explicit export in the environment still wins.
    "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
]

SLURM_CPU = dict(
    job_name="exp",
    mem="16G",
    time="08:00:00",
    cpus=2,
    throttle=16,  # max array tasks running at once (the %N in array=0-N%N)
    partition=None,   # set to your cluster's CPU partition (None = default)
    account=None,
    qos=None,
    gres=None,
    setup=_SETUP_CPU,
)

# autoLiRPA (ibp / crown / alpha-crown) propagate bounds on torch; give them a GPU.
# qos is left unset (some clusters forbid specifying it); set it only if yours needs it.
SLURM_GPU = dict(
    job_name="exp_gpu",
    mem="16G",
    time="04:00:00",
    cpus=4,
    throttle=16,
    partition="gpu",  # EDIT: a GPU partition on your cluster
    account=None,
    qos=None,
    gres="gpu:1",
    setup=_SETUP_GPU,
)

# Long variants for the verification campaigns, whose per-call budgets (hours per
# LiRPA/MAB call, 3600s per Z3 call) need a matching wall. Same resources otherwise.
SLURM_CPU_LONG = {**SLURM_CPU, "time": "24:00:00"}
SLURM_GPU_LONG = {**SLURM_GPU, "time": "12:00:00"}

# MAB (sound box-UCB sampling) is a *pure-JAX* verifier: one big jit-compiled,
# vmapped sample+bound kernel per iteration (k boxes x sample_per_select draws).
# That batches perfectly onto a GPU — with k=1024 each iteration is one fat
# matmul batch — so this profile requests a GPU. It also asks for a lot of RAM
# and a long wall-clock because the sound tiling of the harder (nonlinear) envs
# runs to millions of live boxes and can take hours.
#
# GPU CAVEAT: MAB only uses the GPU if the venv has a CUDA jaxlib
# (`pip install -U "jax[cuda12]"`); with the CPU jaxlib it still runs correctly,
# just without GPU acceleration. To force CPU set gres=None + partition=<cpu>.
SLURM_MAB = dict(
    job_name="cegis_mab",
    mem="64G",
    time="24:00:00",
    cpus=8,
    throttle=8,
    partition="gpu",      # EDIT: GPU partition (see SLURM_GPU); a CPU one if gres=None
    account=None,
    qos=None,
    gres="gpu:1",
    setup=_SETUP_GPU,
)
