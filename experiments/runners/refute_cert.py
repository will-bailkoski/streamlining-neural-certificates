"""
Replay ONE harvested invalid certificate with ONE refuter — is the refuter faster
than the verifier was, on the exact same refutation task?

A snapshot is an invalid cert that a verifier refuted during experiment 1, saved
at <cegis-run>/objects/invalid/round_<k>.npz with its meta (env, cert structure,
epsilon, the counterexample, and the verifier's wall-time for that call) in the
run's manifest. This runner rebuilds that exact situation — same env, same cert,
same drift objective (same mc_samples) — JIT-compiles the refuter, warms it up to
exclude compilation (as the CEGIS pipeline does), then times the search to a
positive-drift point and compares it to the recorded verifier time.

    python -m experiments.runners.refute_cert --snapshot <run>/objects/invalid/round_0.npz --refuter whale

Refuter hyperparameters flow through as plain `--flag value` pairs.
"""

from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import jax.random as jrn

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec
from src.verifiers.drift import make_drift
from src.refuters.base import get_refuter
import src.refuters  # noqa: F401  (register refuters)
from src.results.params import load_params
from experiments.common.runner import parse_extras
from src.results.run_dir import Run


def _resolve(snapshot: Path) -> tuple[dict, dict, str]:
    """(run args, snapshot meta, cert_id) from a snapshot npz path via its manifest."""
    run_dir = snapshot.parents[2]                 # .../runs/<run_id>
    manifest = json.loads((run_dir / "manifest.json").read_text())
    meta = manifest.get("outputs", {}).get("invalid", {}).get(snapshot.stem, {})
    cert_id = f"{run_dir.name}__{snapshot.stem}"
    return manifest.get("args", {}), meta, cert_id


def main():
    p = argparse.ArgumentParser(description="Replay one invalid cert with one refuter",
                                allow_abbrev=False)
    p.add_argument("--snapshot", required=True, help="path to objects/invalid/round_<k>.npz")
    p.add_argument("--refuter", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--campaign", default="refute_certs")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args, extra = p.parse_known_args()
    e_hp, r_hp = parse_extras(extra)
    hp = {**e_hp, **r_hp}
    call_hp = {"budget": args.budget, "batch_size": args.batch_size, **hp}

    snapshot = Path(args.snapshot)
    cegis_args, meta, cert_id = _resolve(snapshot)
    if not meta:
        raise SystemExit(f"no invalid-cert meta for {snapshot} in its manifest")

    env_name = meta["env"]
    epsilon = float(meta["epsilon"])
    mc_samples = int(cegis_args.get("mc_samples", 16))
    verifier_time = float(meta.get("verifier_time", float("nan")))

    run_args = {"refuter": args.refuter, "seed": args.seed, "cert_id": cert_id,
                "env": env_name, "round": meta.get("round"), **hp}
    run = Run("refute_cert", run_args, campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    env = make_env(env_name)
    spec = get_spec(meta["cert_structure"])
    params = load_params(snapshot)
    domain = jnp.asarray(env.domain.bounds)

    V = lambda z: spec.forward(params, z)
    _, _, mcb = make_drift(env, V, epsilon, n=mc_samples)

    def objective(xs, k):
        return jnp.where(jax.vmap(env.is_valid_domain)(xs), mcb(xs, k), -jnp.inf)

    refuter = get_refuter(args.refuter)
    search = jax.jit(lambda k: refuter(objective, domain, k, **call_hp))
    jax.block_until_ready(search(jrn.key(args.seed)))  # warm up (exclude compile)

    t0 = time.perf_counter()
    bx, by, _, _ = search(jrn.key(args.seed + 1))
    jax.block_until_ready((bx, by))
    refuter_time = time.perf_counter() - t0

    best_obj = float(by)
    found = best_obj > 0.0
    speedup = (verifier_time / refuter_time) if refuter_time > 0 else float("nan")
    run.stats({
        "env": env_name, "engine": meta.get("engine"), "refuter": args.refuter,
        "round": meta.get("round"), "cert_id": cert_id,
        "found": found, "best_obj": best_obj,
        "refuter_time": round(refuter_time, 6),
        "verifier_time": round(verifier_time, 6),
        "speedup": round(speedup, 4) if speedup == speedup else None,
    })
    run.finish()
    print(f"{cert_id}: {args.refuter} found={found} obj={best_obj:+.2e}  "
          f"refuter={refuter_time:.4f}s vs verifier={verifier_time:.4f}s  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
