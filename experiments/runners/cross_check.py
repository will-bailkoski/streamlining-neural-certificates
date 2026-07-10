"""
Re-verify ONE saved certificate with ONE engine — the cross-check unit task.

A certificate that engine A produced (and called verified, or gave up on) is
handed to engine B for a single one-shot `find_counterexample` call — no CEGIS
loop, no retraining. The outcomes that matter:

  * A said verified, B finds a GENUINE counterexample  -> soundness red flag
    (one of the two engines is wrong about the same drift condition);
  * A said verified, B is inconclusive                 -> tightness gap (B looser);
  * A was inconclusive, B decides                      -> B tighter on this cert.

A reported counterexample is always re-checked with the canonical jax MC drift
(src.verifiers.drift.recheck_violation): the sound engines over-approximate, so
a phantom CE — one the recheck clears — is a looseness artefact, not a
soundness conflict. It is exactly the cross-check signal we record, never a crash.

    python -m experiments.runners.cross_check \
        --cert_run results/cegis_verify_mab/<tag>/runs/<run_id> --engine smt

Check-engine hyperparameters flow through as plain `--flag value` pairs
(`--noise_disc 8 --time_limit 600`), same as the cegis runner.
"""

from __future__ import annotations
import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# JAX shares the GPU with torch/autoLiRPA here (crown/ibp/alpha-crown propagate bounds
# on torch). Stop JAX preallocating ~75% of VRAM on import — that starves torch (OOM)
# or collides with a co-tenant on a shared GPU. setdefault: an explicit export still
# wins. MUST precede `import jax` below.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax.random as jrn

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401  (register engines)
from src.verifiers.drift import recheck_violation
from src.results.recorder import load_params
from experiments.capabilities import supports
from experiments.common.runner import parse_extras
from src.results.run_dir import Run

_VERDICT = {True: "verified", False: "counterexample", None: "inconclusive"}
_ENGINE_ALIASES = {"mc": "montecarlo"}  # campaign name -> registered engine


def _load_cert(cert_dir: Path):
    """(env, spec, params, manifest args, stats) for the certificate in a run dir.

    Same mechanics as visualisation.common.load_cert, reimplemented here so the
    runner never imports from visualisation/ (matplotlib has no business on a
    compute node)."""
    manifest = json.loads((cert_dir / "manifest.json").read_text())
    args = manifest.get("args", {})
    env = make_env(args["env"])
    spec = get_spec(args.get("cert_structure", "relu_pwl"))
    params = load_params(cert_dir / "objects" / "certificate.npz")
    sp = cert_dir / "stats.json"
    stats = json.loads(sp.read_text()) if sp.exists() else {}
    return env, spec, params, args, stats


def _source_campaign(cert_dir: Path) -> str:
    """Campaign title from the run-dir layout: results/<title>/runs/<id> (flat
    campaigns like verify_matrix) or results/<title>/<tag>/runs/<id> (launched)."""
    holder = cert_dir.parent.parent  # the dir containing runs/
    return holder.name if holder.parent.name == "results" else holder.parent.name


def _agreement(source: str | None, check: str, genuine: bool | None) -> str:
    """Classify the (source verdict, one-shot check verdict) pair.

    conflict : one side proves the cert, the other has a GENUINE counterexample
               on it — a soundness red flag on whichever side proved.
    looser   : the check engine could not reproduce a verified verdict.
    tighter  : the check engine decided (verified or genuinely refuted) a cert
               the source left undecided.
    agree    : same verdict either way.
    """
    if check == "counterexample" and not genuine:
        # phantom CE: the over-approximation flags a point the canonical MC
        # drift clears — a looseness artefact, treated as failing to decide
        check = "inconclusive"
    if source == check:
        return "agree"
    if "verified" in (source, check) and "counterexample" in (source, check):
        return "conflict"
    return "looser" if source == "verified" else "tighter"


def main():
    # allow_abbrev=False so short engine hp are never mistaken for abbreviations
    # of the core flags — every unknown flag flows to extras.
    p = argparse.ArgumentParser(description="Re-verify one saved certificate with one engine",
                                allow_abbrev=False)
    p.add_argument("--cert_run", required=True,
                   help="cegis run dir holding objects/certificate.npz + manifest.json")
    p.add_argument("--engine", required=True, help="the CHECK engine")
    p.add_argument("--seed", type=int, default=0, help="seed for the check engine's key")
    p.add_argument("--campaign", default="cross_check")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args, extra = p.parse_known_args()
    engine_hp, _ = parse_extras(extra)  # bare flags -> check-engine hyperparameters

    cert_dir = Path(args.cert_run)
    source_run_id = cert_dir.name

    # keyed by the source run's IDENTITY (its run_id slug), not its path, so the
    # same cross-check dedupes/resumes wherever the source campaign is mounted
    run_args = {"cert_run": source_run_id, "engine": args.engine,
                "seed": args.seed, **engine_hp}
    run = Run("cross_check", run_args, campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    env, spec, params, src_args, src_stats = _load_cert(cert_dir)
    epsilon = float(src_args.get("epsilon", 1e-3))
    source = {
        "source_campaign": _source_campaign(cert_dir),
        "source_run_id": source_run_id,
        "source_engine": src_args.get("engine"),
        "source_verdict": src_stats.get("verdict"),
        "env": src_args.get("env"),
        "seed": src_args.get("seed"),
        "epsilon": epsilon,
        "check_engine": args.engine,
    }

    engine_name = _ENGINE_ALIASES.get(args.engine, args.engine)
    if not supports(engine_name, env, spec):
        run.stats({**source, "check_verdict": "unsupported", "verdict": "unsupported"})
        run.finish()
        print(f"unsupported: {args.engine} on {source['env']}")
        return

    engine = get_engine(engine_name)
    print(f"cross-check  {source['source_engine']} -> {args.engine}   env={source['env']}"
          f"   eps={epsilon:g}   cert={source_run_id}")
    t0 = time.perf_counter()
    try:
        r = engine.find_counterexample(env, spec, params, epsilon,
                                       key=jrn.key(args.seed), **engine_hp)
    except Exception as e:
        # the lirpa engines have no internal try/except (a CUDA OOM propagates) —
        # record the failure so the cell shows up in the matrix instead of dying
        # as a status=running ghost
        run.stats({**source, "check_verdict": "error", "error": str(e),
                   "time": round(time.perf_counter() - t0, 6)})
        run.finish()
        print(f"  -> ERROR after {time.perf_counter() - t0:.1f}s: {e}")
        return
    wall = time.perf_counter() - t0
    check_verdict = _VERDICT[r.verified]

    genuine = None
    recheck: dict = {}
    if r.violation is not None:
        genuine, drift = recheck_violation(env, spec, params, epsilon,
                                           np.asarray(r.violation).reshape(-1),
                                           jrn.fold_in(jrn.key(args.seed), 1), n=4096)
        recheck = {"recheck_genuine": bool(genuine), "recheck_drift": float(drift)}

    agreement = _agreement(source["source_verdict"], check_verdict, genuine)
    stats = r.stats or {}
    run.stats({
        **source,
        "check_verdict": check_verdict,
        "agreement": agreement,
        "time": round(float(stats.get("time", wall)), 6),
        **recheck,
        # the check engine's full stats, v_-prefixed like the cegis runner
        **{f"v_{k}": v for k, v in stats.items()
           if isinstance(v, (int, float, str)) and k not in ("frames", "trace")},
    })
    run.finish()
    print(f"  -> {check_verdict}  (source said {source['source_verdict']})  "
          f"agreement={agreement}  {wall:.1f}s"
          + (f"  recheck_genuine={genuine}  drift={recheck['recheck_drift']:+.2e}"
             if recheck else ""))


if __name__ == "__main__":
    main()
