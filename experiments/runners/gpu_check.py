"""
GPU pre-flight: confirm JAX (jit) and torch actually run on the GPU.

One array task = one probe of one allocated GPU. For each framework it checks the
device the work LANDS on (not just that a GPU is visible) and times a CPU-vs-GPU
matmul so a real speedup proves the GPU is doing the work:

  JAX   : default backend + that a jitted op's output device is a GPU device,
          and gpu vs cpu time for the same jit.
  torch : cuda availability + that the device's compute capability is in this
          build's arch list (the SAME check src/verifiers/lirpa uses to decide
          cuda vs cpu — so a PASS here means the lirpa engines will use the GPU),
          and gpu vs cpu time.

`passed` is jax-on-gpu AND torch-cuda AND torch-arch-ok. The per-task log prints a
PASS/FAIL banner; stats.json (-> compiled.csv) holds the booleans + speedups so you
can confirm every task on every node used its GPU before launching the real arrays.

    python -m experiments.runners.gpu_check --replica 0
"""

from __future__ import annotations
import argparse
import os
import socket
import time
import warnings

warnings.filterwarnings("ignore")

from src.results.run_dir import Run


def jax_check(n: int, iters: int) -> dict:
    import jax
    import jax.numpy as jnp

    res = {"jax_backend": jax.default_backend(),
           "jax_devices": ";".join(str(d) for d in jax.devices())}
    f = jax.jit(lambda a: a @ a + a)
    gpu_devs = [d for d in jax.devices() if d.platform != "cpu"]

    if gpu_devs:
        x = jax.device_put(jnp.ones((n, n), jnp.float32), gpu_devs[0])
        y = f(x); y.block_until_ready()                       # compile, warm up
        t0 = time.perf_counter()
        for _ in range(iters):
            y = f(x)
        y.block_until_ready()
        gpu_t = time.perf_counter() - t0
        landed = next(iter(y.devices()))
        res.update(jax_gpu=int(landed.platform != "cpu"),
                   jax_gpu_device=str(landed), jax_gpu_time=round(gpu_t, 4))
    else:
        res["jax_gpu"] = 0
        gpu_t = None

    cpu = jax.devices("cpu")[0]
    xc = jax.device_put(jnp.ones((n, n), jnp.float32), cpu)
    yc = f(xc); yc.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        yc = f(xc)
    yc.block_until_ready()
    cpu_t = time.perf_counter() - t0
    res["jax_cpu_time"] = round(cpu_t, 4)
    if gpu_t and gpu_t > 0:
        res["gpu_speedup_jax"] = round(cpu_t / gpu_t, 2)
    return res


def torch_check(n: int, iters: int) -> dict:
    import torch

    res = {"torch_cuda": int(torch.cuda.is_available())}
    if not torch.cuda.is_available():
        res["lirpa_device"] = "cpu"
        return res

    cap = torch.cuda.get_device_capability()
    arch = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_ok = int(sm in arch)
    res.update(torch_device=torch.cuda.get_device_name(0), torch_capability=sm,
               torch_arch_ok=arch_ok, torch_arch_list=";".join(arch),
               lirpa_device="cuda" if arch_ok else "cpu")

    if not arch_ok:
        # torch has no precompiled kernel for this GPU's compute capability;
        # running one would crash with "no kernel image". lirpa falls back to cpu
        # here too, so report and skip the GPU matmul (don't crash the pre-flight).
        return res

    dev = torch.device("cuda")
    a = torch.ones((n, n), device=dev)
    b = a @ a; torch.cuda.synchronize()                       # warm up
    t0 = time.perf_counter()
    for _ in range(iters):
        b = a @ a
    torch.cuda.synchronize()
    gpu_t = time.perf_counter() - t0
    res["torch_gpu_time"] = round(gpu_t, 4)

    ac = torch.ones((n, n))
    bc = ac @ ac
    t0 = time.perf_counter()
    for _ in range(iters):
        bc = ac @ ac
    cpu_t = time.perf_counter() - t0
    res["torch_cpu_time"] = round(cpu_t, 4)
    if gpu_t > 0:
        res["gpu_speedup_torch"] = round(cpu_t / gpu_t, 2)
    return res


def main():
    p = argparse.ArgumentParser(description="GPU pre-flight for JAX + torch")
    p.add_argument("--replica", type=int, default=0, help="probe index (one per array slot)")
    p.add_argument("--size", type=int, default=4096, help="matmul side length")
    p.add_argument("--iters", type=int, default=200, help="matmuls per timing loop")
    p.add_argument("--campaign", default="gpu_check")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    env = {"host": socket.gethostname(),
           "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    run = Run("gpu_check", vars(args), campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    stats = {"replica": args.replica, "size": args.size, "iters": args.iters, **env}
    stats.update(jax_check(args.size, args.iters))
    stats.update(torch_check(args.size, args.iters))
    # report the two frameworks separately. torch_pass is the GPU-CRITICAL path
    # (autoLiRPA runs on torch); jax is only one-off cert training + light refuters,
    # so jax-on-CPU is acceptable here (install a CUDA jaxlib only to accelerate it).
    torch_pass = int(stats.get("torch_cuda") == 1 and stats.get("torch_arch_ok", 0) == 1)
    jax_pass = int(stats.get("jax_gpu") == 1)
    stats.update(torch_pass=torch_pass, jax_pass=jax_pass, passed=int(torch_pass and jax_pass))
    run.stats(stats)
    run.finish()

    bar = "=" * 70
    print(bar)
    print(f"  GPU CHECK  replica={args.replica}  host={env['host']}  "
          f"CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}")
    print(f"  JAX    backend={stats.get('jax_backend')}  on_gpu={bool(jax_pass)}"
          f"  speedup={stats.get('gpu_speedup_jax', 'n/a')}x  ({stats.get('jax_gpu_device','-')})")
    print(f"  torch  cuda={bool(stats.get('torch_cuda'))}  arch_ok={bool(stats.get('torch_arch_ok'))}"
          f"  speedup={stats.get('gpu_speedup_torch', 'n/a')}x  ({stats.get('torch_device','-')})")
    print(f"  ==> torch(GPU-critical): {'PASS' if torch_pass else 'FAIL'}"
          f"   |   jax: {'GPU' if jax_pass else 'CPU-only (ok; CUDA jaxlib to accelerate)'}"
          f"   |   lirpa uses: {stats.get('lirpa_device','?')}")
    print(bar)


if __name__ == "__main__":
    main()
