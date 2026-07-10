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
# NOTE (this cluster): the Gurobi floating-license token server (login2:41954) is
# only reachable from LOGIN nodes, not compute nodes — so milp cannot run in a
# SLURM job here (SLURM runs on compute nodes). Use `smt` (Z3, no license) for
# symbolic verification instead; it covers every env milp would, just slower. To
# enable milp on compute nodes you'd need either a Gurobi academic WLS license
# (cloud-based, no token server — set GRB_LICENSE_FILE to the WLS gurobi.lic) or
# the admins to open the token-server port from compute nodes.
#
# GPU (ibp/crown/alpha-crown): do NOT load a CUDA module — jax[cuda12] and torch
# bundle their own CUDA/CuDNN; a system CuDNN shadows jax's via LD_LIBRARY_PATH and
# breaks the GPU backend. Unload it defensively.
_VENV = "source /nfs/scistore16/tomgrp/wbailkos/masters_thesis/venv/bin/activate"
_SETUP_CPU = [
    _VENV,
    # "module load gurobi/952",   # milp only: needs WLS license / token-server access
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
    partition=None,   # set to a CPU partition that can reach the license token server
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
    partition="gpu100",  # a GPU partition MUST be named (default partition has no GPUs)
    account=None,
    qos=None,             # do NOT set qos here (the cluster forbids specifying it)
    gres="gpu:1",
    setup=_SETUP_GPU,
)

# Long variants for the cranked-budget "bigtest" campaigns: the verify calls are
# given 5-10x the old per-call budgets, so the arrays need the wall to match
# (lirpa noise_disc=8 on 2D is 64 V-compositions per bound call; smt gets a
# 600s Z3 timeout per call). Same resources otherwise.
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
# (`pip install -U "jax[cuda12]"`). gpu_check found the cluster venv shipping the
# CPU jaxlib (jax_gpu=0) — with that build MAB still runs correctly on this GPU
# node's CPU cores (the jit/vmap speedups make CPU viable), just without GPU
# acceleration. To force CPU (free the GPU) set gres=None + partition=<cpu>.
SLURM_MAB = dict(
    job_name="cegis_mab",
    mem="64G",
    time="24:00:00",
    cpus=8,
    throttle=8,
    partition="gpu100",   # GPU partition (see SLURM_GPU note); set a CPU one if gres=None
    account=None,
    qos=None,
    gres="gpu:1",
    setup=_SETUP_GPU,
)
