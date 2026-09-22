"""
One CEGIS run: train a certificate, then close the loop with a sound verifier,
optionally pre-screening each round with a cheap refuter.

    # verifier-only (experiment 1)
    python -m experiments.runners.cegis --env linstoch2D --engine smt --refuter none --seed 0
    # refuter-first (experiment 4)
    python -m experiments.runners.cegis --env linstoch2D --engine smt --refuter whale --seed 0

Hyperparameters beyond the core flags flow through as plain `--flag value` pairs:
bare flags are engine hyperparameters (`--noise_disc 2`, `--timeout_ms 60000`),
`--r_<key>` flags are refuter hyperparameters (`--r_a_max 2.0`). This keeps a
combo line identical to the local command.

Produces the summary stats (rounds, verdict, time split), the per-round series
(iterations.csv), the final certificate (objects/certificate.npz), and — for
every round a verifier refuted — an invalid-cert snapshot under objects/invalid/
(params + counterexample + that verifier call's wall-time) for the refute-certs
experiment to replay.
"""

from __future__ import annotations
import argparse
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
import jax
import jax.numpy as jnp
import jax.random as jrn

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec, glorot_init, get_spectral_norm_product
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401
from src.verifiers.lirpa import LIRPA_METHODS
from src.verifiers.drift import make_drift, recheck_violation
from src.refuters.base import get_refuter
import src.refuters  # noqa: F401
from src.training.training_cycles import train_fixed_dataset, cegis_trainer
from experiments.capabilities import supports
from experiments.common.runner import parse_extras, expand_cex
from src.results.run_dir import Run

_VERDICT = {True: "verified", False: "counterexample", None: "inconclusive"}
_ENGINE_ALIASES = {"mc": "montecarlo"}  # campaign name -> registered engine
_WIDTH = 74


def _rule(char="="):
    return char * _WIDTH


def _banner(args, hidden, engine_hp):
    print()
    print(_rule("="))
    print(f"  CEGIS   |   env={args.env}   engine={args.engine}   refuter={args.refuter}")
    print(f"  seed={args.seed}   epsilon={args.epsilon:g}   cert={args.cert_structure} {hidden}"
          f"   history_len={args.history_len}")
    if "noise_disc" in engine_hp:
        print(f"  noise_disc={engine_hp['noise_disc']}")
    print(_rule("="))


def _fmt_loss(loss):
    return "-" if loss is None else f"{loss:.2e}"


def _round_gap(stats, verified):
    """Closeness in drift units, so you can watch the loop converge toward 0. For a
    counterexample: the size of the violation found. For verified/inconclusive: the
    tightest SOUND residual drift upper bound the verifier reached (<= 0 == verified)."""
    stats = stats or {}
    if verified is False:                       # counterexample: how big a violation
        for k in ("max_drift", "drift_lo"):
            if isinstance(stats.get(k), (int, float)):
                return stats[k]
    for k in ("obj_bound", "resid_bound", "max_ucb", "certified_sup_drift", "max_drift"):
        if isinstance(stats.get(k), (int, float)):
            return stats[k]
    return None


def _round_effort(stats):
    """Compact indicator of what the verifier spent (boxes / samples / branches)."""
    stats = stats or {}
    if isinstance(stats.get("total_boxes"), (int, float)):
        return f"{int(stats['total_boxes'])//1000}k bx"
    if isinstance(stats.get("samples"), (int, float)):
        return f"{int(stats['samples'])//1000}k sm"
    if isinstance(stats.get("n_branches"), (int, float)):
        return f"{int(stats['n_branches'])} br"
    return ""


def _inconclusive_reason(stats):
    """Why the verifier could not decide: the specific budget/limit that stopped it."""
    stats = stats or {}
    if stats.get("hit"):                     # lirpa/mab: max_boxes | max_depth | max_samples | timeout
        return stats["hit"]
    if stats.get("reason"):                  # Z3: 'timeout' | 'unknown'
        return str(stats["reason"])
    if stats.get("status") == 9:             # gurobi TIME_LIMIT (bound still > 0)
        return "timeout"
    if isinstance(stats.get("certified_sup_drift"), (int, float)):  # mc: only a positive ceiling
        return "positive-ceiling"
    if stats.get("error"):                   # engine died mid-call; full message in v_error
        return "error"
    return "no-progress"


def _print_table_header():
    print()
    print(f"  {'rnd':>3}  {'source':<8}  {'outcome':<20}  {'loss':>9}  {'buffer':>7}  "
          f"{'gap(drift)':>11}  {'effort':>8}  {'elapsed':>8}")
    print("  " + "-" * (_WIDTH + 22))


def _print_round(rnd, source, outcome, loss, buffer, elapsed, gap=None, effort=""):
    gtxt = "" if not isinstance(gap, (int, float)) else f"{gap:+.4f}"
    print(f"  {rnd:>3}  {source:<8}  {outcome:<20}  {_fmt_loss(loss):>9}  {buffer:>7}  "
          f"{gtxt:>11}  {effort:>8}  {elapsed:>7.1f}s")


def _print_summary(verdict, rounds, total, verifier_t, refuter_t, n_calls, lip, note=""):
    print(_rule("="))
    print(f"  result     : {verdict}" + (f"   ({note})" if note else ""))
    print(f"  rounds     : {rounds}")
    print(f"  time       : {total:.1f}s   (verifier {verifier_t:.1f}s | refuter {refuter_t:.1f}s)")
    print(f"  verifier   : {n_calls} call(s)")
    print(f"  lipschitz  : {lip:.4g}")
    print(_rule("="))


def _lipbab_lv(params, boundaries, timeout=30.0):
    """LipBaB Lipschitz upper bound for V's raw ReLU net (sound, tightens to timeout)."""
    from src.verifiers.lipbab import compute_lipschitz

    weights = [None] + [np.asarray(W, dtype=float) for W, _ in params]
    biases = [None] + [np.asarray(b, dtype=float) for _, b in params]
    L_V, _jac, elapsed = compute_lipschitz(
        weights, biases, boundaries, pnorm=2, timeout=timeout, verbose=False
    )
    return float(L_V), float(elapsed)


def compute_lipschitz_report(env, params, spec, lipbab_timeout=30.0) -> dict:
    """Lipschitz constants of the trained certificate, for the stats record."""
    L_f = float(env.lip_f)
    L_V_spec = float(get_spectral_norm_product(params))
    out = {
        "lip_f": L_f,
        "lip_v_spectral": L_V_spec,
        "lip_d_spectral": L_V_spec * (L_f + 1.0),
    }
    if spec.hidden_activation == "relu":
        boundaries = np.asarray(env.domain.bounds, dtype=float).tolist()
        L_V_lbb, secs = _lipbab_lv(params, boundaries, timeout=lipbab_timeout)
        out["lip_v_lipbab"] = L_V_lbb
        out["lip_d_lipbab"] = L_V_lbb * (L_f + 1.0)
        out["lipbab_secs"] = secs
    return out


def print_lipschitz_report(lip: dict, spec) -> None:
    print()
    print(_rule("="))
    print(f"  lipschitz  |  L_V of trained certificate (p=2),  L_f = {lip['lip_f']:.4g}")
    print(f"  method 1  spectral product : L_V = {lip['lip_v_spectral']:11.4g}"
          f"   L_D = {lip['lip_d_spectral']:.4g}")
    if "lip_v_lipbab" not in lip:
        print(f"  method 2  LipBaB           : n/a (needs ReLU hidden, "
              f"got {spec.hidden_activation})")
        print(_rule("="))
        return
    print(f"  method 2  LipBaB ({lip['lipbab_secs']:4.1f}s)    : L_V = {lip['lip_v_lipbab']:11.4g}"
          f"   L_D = {lip['lip_d_lipbab']:.4g}")
    if lip["lip_v_lipbab"] > 0:
        print(f"  ratio  L_V(spectral) / L_V(LipBaB) = "
              f"{lip['lip_v_spectral'] / lip['lip_v_lipbab']:.3f}"
              f"   (LipBaB tighter by this factor)")
    print(_rule("="))


def _train_hp(spec):
    return dict(batch_size=256, lr=1e-2, n=16, decay=0.9, max_epochs=200,
               threshold=0.0, certificate=spec.forward, printing=False)


def main():
    # allow_abbrev=False so short engine/refuter hp are never mistaken for
    # abbreviations of the core flags — every unknown flag flows to extras.
    p = argparse.ArgumentParser(description="One CEGIS run (verifier-only or refuter-first)",
                                allow_abbrev=False)
    p.add_argument("--env", required=True)
    p.add_argument("--verify_env", default=None,
                   help="env variant the VERIFIER checks (default: --env), e.g. "
                        "the same dynamics/domain with a larger equilibrium")
    p.add_argument("--engine", default="smt", help="sound verifier that closes the loop")
    p.add_argument("--refuter", default="none", help="refuter to pre-screen each round, or 'none'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cert_structure", default="relu_pwl")
    p.add_argument("--hidden_layers", default="8,8")
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--train_epsilon", type=float, default=None,
                   help="epsilon used during TRAINING (default: 10x the verify epsilon). "
                        "Set it larger than --epsilon to train a bigger margin while "
                        "verifying an easier threshold — helps bound-propagation verifiers.")
    p.add_argument("--max_rounds", type=int, default=300)
    p.add_argument("--budget", type=int, default=8192, help="refuter evaluation budget")
    p.add_argument("--batch_size", type=int, default=256, help="refuter batch size")
    p.add_argument("--mc_samples", type=int, default=16)
    p.add_argument("--sample_count", type=int, default=2000)
    p.add_argument("--history_len", type=int, default=1,
                   help="counterexamples fed to the trainer per round (fairness control)")
    p.add_argument("--refuter_recheck_n", type=int, default=4096,
                   help="fresh MC draws to confirm a refuter counterexample before "
                        "accepting it (0 disables). The refuter maximises a noisy "
                        "mean-of-mc_samples surface, so its argmax is extreme-value "
                        "biased; this gate kills the spurious positives.")
    p.add_argument("--ball_radius", type=float, default=0.05,
                   help="radius for point-verifiers when history_len > 1")
    p.add_argument("--campaign", default="cegis")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--overwrite", action="store_true")
    args, extra = p.parse_known_args()
    engine_hp, refuter_hp = parse_extras(extra)  # bare -> engine, --r_* -> refuter

    # the run is keyed by what it ACTUALLY ran (core args + resolved hyperparameters)
    run_args = {**vars(args), **engine_hp, **{f"r_{k}": v for k, v in refuter_hp.items()}}
    run = Run("cegis", run_args, campaign=args.campaign,
              results_dir=args.results_dir, overwrite=args.overwrite)
    if run.done:
        print(f"skip {run.run_id} (already done)")
        return

    env = make_env(args.env)
    # verifier checks env_v (defaults to env): training/refuting stay on env, so
    # verifier counterexamples (domain \ eq_v is a subset of domain \ eq) remain
    # valid training points either way
    env_v = make_env(args.verify_env) if args.verify_env else env
    spec = get_spec(args.cert_structure)
    engine_name = _ENGINE_ALIASES.get(args.engine, args.engine)
    if not supports(engine_name, env_v, spec):
        run.stats({"verdict": "unsupported", "env": args.env})
        run.finish()
        print(f"unsupported: {args.engine} on {args.env}")
        return

    hidden = [int(x) for x in args.hidden_layers.split(",") if x != ""]
    training_eps = args.train_epsilon if args.train_epsilon is not None else args.epsilon * 10.0
    thp = _train_hp(spec)
    engine = get_engine(engine_name)

    refuter = None if args.refuter == "none" else get_refuter(args.refuter)
    refuter_call_hp = {"budget": args.budget, "batch_size": args.batch_size, **refuter_hp}
    domain = jnp.asarray(env.domain.bounds)

    _banner(args, hidden, engine_hp)
    if args.engine == "mab":  # confirm whether the sample kernel gets a GPU (see SLURM_MAB)
        print(f"  jax backend: {jax.default_backend()}   devices: {jax.devices()}")

    print(f"  seeding certificate on {args.sample_count} samples "
          f"(train_eps={training_eps:g}, verify_eps={args.epsilon:g}) ...")
    dom_mid = 0.5 * (domain[:, 0] + domain[:, 1])  # zero for origin-centered envs
    init = glorot_init([env.dim] + hidden + [1], jrn.fold_in(jrn.key(args.seed), 99),
                       input_center=dom_mid)
    params, _, seed_loss, full_x = train_fixed_dataset(
        env, training_eps, init, N=args.sample_count,
        dataset_key=jrn.fold_in(jrn.key(args.seed), 7), **thp,
    )
    print(f"  seed loss = {seed_loss:.2e}")
    gen = cegis_trainer(env, training_eps, params, seed_dataset=full_x, **thp)
    params, _ = next(gen)

    def objective(ps, xs, k):
        V = lambda z: spec.forward(ps, z)
        _, _, mcb = make_drift(env, V, args.epsilon, n=args.mc_samples)
        return jnp.where(jax.vmap(env.is_valid_domain)(xs), mcb(xs, k), -jnp.inf)

    # params is the only thing that changes across rounds; make it a traced arg of one
    # jitted runner so the refuter's lax.scan compiles once and is reused every round.
    run_refuter = None
    if refuter is not None:
        def _run_refuter(ps, rkey):
            return refuter(lambda xs, k: objective(ps, xs, k), domain, rkey, **refuter_call_hp)
        run_refuter = jax.jit(_run_refuter)

    key = jrn.fold_in(jrn.key(args.seed), 3)
    refuter_time = verifier_time = 0.0
    n_verifier_calls = n_refuter_ces = n_verifier_ces = n_refuter_spurious = 0
    verdict = "inconclusive"
    verdict_note = ""
    buffer = args.sample_count
    last_vstats: dict = {}  # final verifier call's engine stats -> compiled.csv
    t0 = time.perf_counter()

    _print_table_header()

    for rnd in range(args.max_rounds):
        ce, source, outcome, rgap, reffort = None, "verifier", None, None, ""

        if refuter is not None:
            key, rk = jrn.split(key)
            tr = time.perf_counter()
            bx, by, _, _ = run_refuter(params, rk)
            jax.block_until_ready((bx, by))
            found_ce = float(by) > 0.0
            if found_ce and args.refuter_recheck_n > 0:
                key, ck = jrn.split(key)
                genuine, dchk = recheck_violation(env, spec, params, args.epsilon,
                                                  np.asarray(bx).reshape(-1), ck,
                                                  n=args.refuter_recheck_n)
                if not genuine:
                    found_ce = False
                    n_refuter_spurious += 1
                    run.log_iter({"round": rnd, "source": "refuter", "verdict": "spurious",
                                  "recheck_drift": round(float(dchk), 6),
                                  "elapsed": round(time.perf_counter() - t0, 3)})
            refuter_time += time.perf_counter() - tr
            if found_ce:
                key, ek = jrn.split(key)
                ce = expand_cex(np.asarray(bx)[None, :], args.history_len, ek, env,
                                region=None, ball_radius=args.ball_radius)
                source = "refuter"
                n_refuter_ces += ce.shape[0]
                outcome = f"CE (obj {float(by):+.2e})"
                rgap, reffort = float(by), "refuter"
                run.log_iter({"round": rnd, "source": "refuter", "verdict": "counterexample",
                              "elapsed": round(time.perf_counter() - t0, 3)})
                run.flush()  # survive a SLURM timeout with the rounds so far

        if ce is None:
            key, vk = jrn.split(key)
            tv = time.perf_counter()
            r = engine.find_counterexample(env_v, spec, params, args.epsilon, key=vk, **engine_hp)
            call_dt = time.perf_counter() - tv
            verifier_time += call_dt
            n_verifier_calls += 1
            rgap, reffort = _round_gap(r.stats, r.verified), _round_effort(r.stats)
            call_stats = {k: v for k, v in (r.stats or {}).items()
                          if (isinstance(v, (int, float)) and k != "time")
                          or k in ("hit", "error")}
            # keep the LAST call's full engine stats (boxes/depth/samples/budget
            # 'hit' reason ...) for the summary record — for an inconclusive run
            # this is the line that says WHICH budget stopped it
            last_vstats = {k: v for k, v in (r.stats or {}).items()
                           if isinstance(v, (int, float, str)) and k != "frames"}
            run.log_iter({"round": rnd, "source": "verifier", "verdict": _VERDICT[r.verified],
                          "call_time": round(call_dt, 3),
                          "elapsed": round(time.perf_counter() - t0, 3), **call_stats})
            run.flush()  # survive a SLURM timeout with the rounds so far
            if r.verified:
                verdict = "verified"
                _print_round(rnd, "verifier", "VERIFIED", None, buffer,
                             time.perf_counter() - t0, gap=rgap, effort=reffort)
                break
            if r.violation is None:
                verdict = "inconclusive"
                verdict_note = _inconclusive_reason(r.stats)
                _print_round(rnd, "verifier", f"inconc: {verdict_note}", None, buffer,
                             time.perf_counter() - t0, gap=rgap, effort=reffort)
                break
            # persist this refuted cert so refute-certs can replay the exact task
            # (the single reported point + the verifier's wall-time for this call)
            run.snapshot(f"round_{rnd}", params, meta={
                "round": rnd, "env": args.env, "cert_structure": args.cert_structure,
                "hidden_layers": args.hidden_layers, "epsilon": args.epsilon,
                "seed": args.seed, "engine": args.engine, "verifier_time": round(call_dt, 6),
                "cex": np.asarray(r.violation).reshape(-1).tolist(),
            })
            key, ek = jrn.split(key)
            ce = expand_cex(r.violation, args.history_len, ek, env,
                            region=r.region, ball_radius=args.ball_radius)
            outcome = "counterexample"
            n_verifier_ces += ce.shape[0]

        buffer += int(np.asarray(ce).shape[0])
        params, loss = gen.send(jnp.asarray(ce))
        _print_round(rnd, source, outcome, loss, buffer, time.perf_counter() - t0,
                     gap=rgap, effort=reffort)
        run.log_iter({"round": rnd, "source": "train", "verdict": "retrained",
                      "loss": float(loss), "buffer": buffer,
                      "elapsed": round(time.perf_counter() - t0, 3)})
        run.flush()
    else:
        verdict = "max_rounds"
        verdict_note = "no valid cert trained"

    lipd = compute_lipschitz_report(env, params, spec)
    _print_summary(verdict, rnd + 1, time.perf_counter() - t0,
                   verifier_time, refuter_time, n_verifier_calls, lipd["lip_v_spectral"],
                   note=verdict_note)
    print_lipschitz_report(lipd, spec)
    run.stats({
        "strategy": (f"refuter:{args.refuter}->{args.engine}" if refuter
                     else f"verifier_only:{args.engine}"),
        "verdict": verdict,
        "rounds": rnd + 1,
        "total_time": round(time.perf_counter() - t0, 3),
        "refuter_time": round(refuter_time, 3),
        "verifier_time": round(verifier_time, 3),
        "n_verifier_calls": n_verifier_calls,
        "n_refuter_ces": n_refuter_ces,
        "n_refuter_spurious": n_refuter_spurious,
        "n_verifier_ces": n_verifier_ces,
        "n_counterexamples": n_refuter_ces + n_verifier_ces,
        "final_buffer": buffer,
        "dim": env.dim,
        "lipschitz": lipd["lip_v_spectral"],
        **lipd,
        # final verifier call's engine stats, v_-prefixed (v_total_boxes, v_depth,
        # v_samples, v_iterations, v_hit, v_error ...) — the diagnosis columns
        **{f"v_{k}": v for k, v in last_vstats.items()},
    })
    run.object("certificate", params, meta={
        "verdict": verdict, "env": args.env, "structure": args.cert_structure,
        "hidden_layers": args.hidden_layers, "epsilon": args.epsilon,
        "seed": args.seed, "lipschitz": lipd["lip_v_spectral"],
    })
    run.finish()


if __name__ == "__main__":
    main()
