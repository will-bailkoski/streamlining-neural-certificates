"""
One refuter × one BBOB function × one hyperparameter combo.

    python -m experiments.runners.refute_bbob --function rastrigin --refuter whale --seed 0
    python -m experiments.runners.refute_bbob --function ackley --refuter gradient --lr 0.1 --plots

Refuter hyperparameters flow through as plain `--flag value` pairs (`--a_max 2.0`).
Records final value + wall-clock (stats.json), JIT-compiling the search before
timing it (so the compile cost is excluded, exactly as in the CEGIS loop). With
--plots also writes the search-pattern figure.
"""

from __future__ import annotations
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.random as jrn

from experiments.bbob_functions import make_suite
from src.refuters.base import get_refuter
import src.refuters  # noqa: F401  (register refuters)
from experiments.common.runner import parse_extras
from src.results.run_dir import Run


def main():
    # allow_abbrev=False so short refuter hp (e.g. --b, --p) are NOT mistaken for
    # abbreviations of core flags (--budget, --batch_size) — they go to extras.
    p = argparse.ArgumentParser(description="Refute one BBOB function with one refuter",
                                allow_abbrev=False)
    p.add_argument("--function", required=True)
    p.add_argument("--refuter", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--plots", action="store_true", help="also save the search-pattern figure")
    p.add_argument("--campaign", default="refute_bbob")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args, extra = p.parse_known_args()
    e_hp, r_hp = parse_extras(extra)
    hp = {**e_hp, **r_hp}  # this runner has only a refuter; all extras are its hp
    call_hp = {"budget": args.budget, "batch_size": args.batch_size, **hp}

    run = Run("refute_bbob", {**vars(args), **hp}, campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    fn = {f.name: f for f in make_suite(args.dim)}[args.function]
    refuter = get_refuter(args.refuter)

    search = jax.jit(lambda k: refuter(lambda xs, kk: fn.f(xs), fn.domain, k, **call_hp))
    jax.block_until_ready(search(jrn.key(args.seed)))  # warm up (exclude compile)
    t0 = time.perf_counter()
    bx, by, _, _ = search(jrn.key(args.seed + 1))
    jax.block_until_ready((bx, by))
    run.stats({"final_value": float(by), "time": round(time.perf_counter() - t0, 4)})

    if args.plots and args.dim == 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.plotting.refuter_search import draw_search
        from src.plotting.systems import PUB_RC
        with plt.rc_context(PUB_RC):
            f, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
            draw_search(ax, args.refuter, fn, budget=call_hp["budget"],
                        batch_size=call_hp["batch_size"], seed=args.seed,
                        hp={k: v for k, v in call_hp.items()
                            if k not in ("budget", "batch_size")})
            ax.set_title(f"{args.refuter} - {args.function}")
            run.figure(f, f"{args.refuter}_{args.function}")
            plt.close(f)

    run.finish()
    print(f"done {run.run_id} -> {run.dir}")


if __name__ == "__main__":
    main()
