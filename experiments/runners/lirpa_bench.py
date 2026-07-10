"""
Benchmark the autoLiRPA bound methods x splitting strategies on ONE problem.

Trains a single certificate for (env, hidden_layers, seed), then verifies that
SAME certificate with every (bound method x split strategy) combination so the
comparison is apples-to-apples. Per combination it records the work done and the
cost:

    total_boxes    cumulative boxes bounded (total work)
    peak_boxes     max boxes held at once (memory pressure; 2**d vs 2x growth)
    n_bound_calls  compute_bounds invocations (GPU batches)
    time           wall-clock seconds
    boxes_per_sec  throughput (a GPU-utilisation proxy)
    verdict, depth, hit  (verified / counterexample / inconclusive + why)

    python -m experiments.runners.lirpa_bench --env linstoch2D --hidden_layers 8,8 --seed 0

One row per (method, split) -> iterations.csv; the shared per-problem metadata
(dim, n_params, lipschitz, device, ...) -> stats.json. Swept via the GPU launcher
experiments.lirpa_bench (collated with --aggregate, one row per method x split).
"""

from __future__ import annotations
import argparse
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)

import numpy as np
import jax.random as jrn

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec, glorot_init, get_spectral_norm_product
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401  (register engines)
from src.verifiers.lirpa import LIRPA_METHODS
from src.training.training_cycles import train_fixed_dataset
from experiments.capabilities import supports
from src.results.run_dir import Run

SPLITS = ("all", "longest")


def _n_params(params) -> int:
    return int(sum(np.asarray(W).size + np.asarray(b).size for W, b in params))


def _train_hp(spec):
    return dict(batch_size=256, lr=1e-2, n=16, decay=0.9, max_epochs=200,
               threshold=0.0, certificate=spec.forward, printing=False)


def main():
    p = argparse.ArgumentParser(description="Benchmark autoLiRPA methods x split strategies")
    p.add_argument("--env", required=True)
    p.add_argument("--cert_structure", default="relu_pwl",
                   help="lirpa needs a relu/clip/bottom cert (no hardsigmoid)")
    p.add_argument("--hidden_layers", default="8,8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--sample_count", type=int, default=2000)
    p.add_argument("--methods", default=",".join(LIRPA_METHODS),
                   help="comma list of bound methods to benchmark")
    p.add_argument("--splits", default=",".join(SPLITS),
                   help="comma list of split strategies: all,longest")
    p.add_argument("--noise_disc", type=int, default=1)
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--max_boxes", type=int, default=500_000)
    p.add_argument("--batch_size", type=int, default=4096, help="boxes per compute_bounds (GPU)")
    p.add_argument("--device", default=None, help="cuda | cpu | (default: auto)")
    p.add_argument("--campaign", default="lirpa_bench")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    run = Run("lirpa_bench", vars(args), campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    env = make_env(args.env)
    spec = get_spec(args.cert_structure)
    methods = [m for m in args.methods.split(",") if m]
    splits = [s for s in args.splits.split(",") if s]

    if not all(supports(m, env, spec) for m in methods):
        run.stats({"verdict": "unsupported", "env": args.env, "cert_structure": args.cert_structure})
        run.finish()
        print(f"unsupported: {methods} on {args.env}/{args.cert_structure}")
        return

    hidden = [int(x) for x in args.hidden_layers.split(",") if x != ""]
    init = glorot_init([env.dim] + hidden + [1], jrn.fold_in(jrn.key(args.seed), 99))
    params, _, seed_loss, _ = train_fixed_dataset(
        env, args.epsilon * 10.0, init, N=args.sample_count,
        dataset_key=jrn.fold_in(jrn.key(args.seed), 7), **_train_hp(spec),
    )

    hp = dict(noise_disc=args.noise_disc, max_depth=args.max_depth,
              max_boxes=args.max_boxes, batch_size=args.batch_size, device=args.device)
    device_used = None
    print(f"  {args.env} {args.hidden_layers} (dim={env.dim}, n_params={_n_params(params)}) "
          f"seed_loss={seed_loss:.2e}")
    print(f"  {'method':12s} {'split':8s} {'verdict':14s} {'boxes':>9s} {'peak':>7s} "
          f"{'calls':>6s} {'time':>8s} {'boxes/s':>9s}")
    for method in methods:
        for split in splits:
            r = get_engine(method).find_counterexample(
                env, spec, params, args.epsilon, key=jrn.fold_in(jrn.key(args.seed), 3),
                split=split, **hp,
            )
            s = r.stats
            device_used = s.get("device", device_used)
            row = {
                "method": method, "split": split,
                "verdict": {True: "verified", False: "counterexample", None: "inconclusive"}[r.verified],
                "total_boxes": s.get("total_boxes"), "peak_boxes": s.get("peak_boxes"),
                "n_bound_calls": s.get("n_bound_calls"), "time": s.get("time"),
                "boxes_per_sec": s.get("boxes_per_sec"), "depth": s.get("depth"),
                "hit": s.get("hit", ""),
            }
            run.log_iter(row)
            run.flush()
            print(f"  {method:12s} {split:8s} {row['verdict']:14s} {row['total_boxes']:>9} "
                  f"{row['peak_boxes']:>7} {row['n_bound_calls']:>6} {row['time']:>7.3f}s "
                  f"{row['boxes_per_sec']:>9}")

    run.stats({
        "env": args.env, "dim": env.dim, "hidden_layers": args.hidden_layers,
        "n_params": _n_params(params), "cert_structure": args.cert_structure,
        "epsilon": args.epsilon, "seed": args.seed, "seed_loss": round(float(seed_loss), 6),
        "noise_disc": args.noise_disc, "max_depth": args.max_depth,
        "max_boxes": args.max_boxes, "batch_size": args.batch_size,
        "device": device_used, "lipschitz": float(get_spectral_norm_product(params)),
    })
    run.finish()
    print(f"done {run.run_id} -> {run.dir}")


if __name__ == "__main__":
    main()
