"""
Verification matrix: which (engine, env) pairs yield a verified certificate, at what cost.

For each (engine, env) it runs the standard CEGIS runner --- which trains, closes the
loop with the sound verifier, and SAVES the resulting certificate --- and records the
verdict plus rounds / total-time / verifier-time. The saved certs under
results/verify_matrix/runs/ are the very certificates to reuse downstream (e.g. the
refuter-speedup comparison). Resumable: an already-completed (engine, env) is skipped.

Each engine uses the cert structure it needs: bounded_pwl (hard-sigmoid) for the
sample-based (mab, mc) and for MILP on stochastic dynamics (the recipe that makes the
noise over-approximation tight enough); relu_pwl for the bound-propagation engines.

    python -m experiments.verify_matrix                       # phase A+B (fast + MILP)
    python -m experiments.verify_matrix --engines mab         # phase C (slow, 2D)
    python -m experiments.verify_matrix --envs linear2D,linstoch2D
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

CAMPAIGN = "verify_matrix"
EPS = "1e-3"
ENVV = dict(os.environ, PYTHONPATH=".", PYTHONUNBUFFERED="1")

# engine -> how to run it (cert, width, train_eps, engine hp, per-call cap minutes)
ENGINES = {
    "milp":  dict(cert="bounded_pwl", width="4,4", teps="1e-2",
                  hp=dict(noise_disc=4, time_limit=300), cap=35),
    "crown": dict(cert="relu_pwl", width="8,8", teps="2e-2",
                  hp=dict(noise_disc=2, max_boxes=200000, max_depth=40), cap=20),
    "ibp":   dict(cert="relu_pwl", width="8,8", teps="2e-2",
                  hp=dict(noise_disc=2, max_boxes=200000, max_depth=40), cap=20),
    "alpha-crown": dict(cert="relu_pwl", width="8,8", teps="2e-2",
                        hp=dict(noise_disc=2, max_boxes=200000, max_depth=40), cap=30),
    "smt":   dict(cert="relu_pwl", width="4,4", teps="2e-2",
                  hp=dict(noise_disc=4, timeout_ms=120000), cap=25),
    "mc":    dict(cert="bounded_pwl", width="4,4", teps="1e-2",
                  hp=dict(), cap=12),
    "mab":   dict(cert="bounded_pwl", width="4,4", teps="1e-2",
                  hp=dict(grid_per_dim=32, k=1024, sample_per_select=128,
                          max_boxes=10_000_000, max_samples=1_000_000_000,
                          significance=0.05, range_bound=1.0, progress_interval=2000),
                  cap=150),
}
ENVS = ["linear2D", "linear3D", "linear4D", "linstoch2D", "linstoch3D",
        "linstoch4D", "doublewell", "pendulum_lqr"]


def _find_run(env, engine, seed, rdir):
    for d in sorted(glob.glob(f"{rdir}/{CAMPAIGN}/runs/*")):
        mf = os.path.join(d, "manifest.json")
        if not os.path.exists(mf):
            continue
        a = json.load(open(mf)).get("args", {})
        if a.get("env") == env and a.get("engine") == engine and int(a.get("seed", -1)) == seed:
            sp = os.path.join(d, "stats.json")
            return json.load(open(sp)) if os.path.exists(sp) else {}
    return None


def run_cell(engine, env, seed, rdir):
    cfg = ENGINES[engine]
    cached = _find_run(env, engine, seed, rdir)
    if cached is not None:
        return cached, True
    cmd = [sys.executable, "-m", "experiments.runners.cegis",
           "--env", env, "--engine", engine, "--refuter", "none", "--seed", str(seed),
           "--cert_structure", cfg["cert"], "--hidden_layers", cfg["width"],
           "--epsilon", EPS, "--train_epsilon", cfg["teps"], "--max_rounds", "60",
           "--campaign", CAMPAIGN, "--results_dir", rdir]
    for k, v in cfg["hp"].items():
        cmd += [f"--{k}", str(v)]
    t = time.time()
    try:
        subprocess.run(cmd, env=ENVV, timeout=cfg["cap"] * 60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    stats = _find_run(env, engine, seed, rdir)
    if stats is None:
        stats = {"verdict": "killed", "note": f"no stats after {cfg['cap']}m cap"}
    stats["_wall"] = round(time.time() - t, 1)
    return stats, False


def cell_summary(s):
    """verdict + the REASON it stopped + a closeness GAP: the residual drift UPPER bound
    the method still couldn't push <= 0 (<= 0 would verify), in drift units. MILP exposes
    it as obj_bound, lirpa as resid_bound; SMT (Z3) gives no progress signal at all."""
    v = s.get("verdict")
    if not v:
        return "-"
    if v == "verified":
        return f"OK {float(s.get('total_time', 0)):.0f}s"
    if v == "counterexample":
        return "cex"
    if v == "max_rounds":
        return "no-converge"        # trainer never produced a valid cert (not the verifier)
    if v == "unsupported":
        return "n/a"
    gap = next((s[k] for k in ("v_obj_bound", "obj_bound", "v_resid_bound")
                if isinstance(s.get(k), (int, float))), None)
    g = f" {gap:+.3f}" if gap is not None else ""
    hit, reason, status = s.get("v_hit"), s.get("v_reason"), s.get("v_status")
    if hit == "max_boxes":
        return f"maxboxes{g}"
    if hit == "max_depth":
        return f"maxdepth{g}"
    if hit == "max_samples":
        return f"maxsamples{g}"
    if reason == "timeout":
        return "Z3-timeout"         # Z3 unknown: no closeness signal
    if status == 9:
        return f"timeout{g}"        # milp hit its time limit, bound still > 0
    if v == "killed":
        return "wall-cap"
    return f"inconc{g}"


def print_matrix(rows, engines, envs):
    W = 18
    width = 14 + W * len(engines)
    print("\n" + "=" * width, flush=True)
    print(f"{'env':<14}" + "".join(f"{e:>{W}}" for e in engines), flush=True)
    print("-" * width, flush=True)
    for env in envs:
        print(f"{env:<14}" + "".join(f"{cell_summary(rows.get((e, env), {})):>{W}}"
                                     for e in engines), flush=True)
    print("=" * width, flush=True)


def main():
    p = argparse.ArgumentParser(description="CEGIS verification matrix (engine x env)")
    p.add_argument("--engines", default="milp,crown,ibp,smt,mc")
    p.add_argument("--envs", default=",".join(ENVS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    engines = [e for e in args.engines.split(",") if e]
    envs = [e for e in args.envs.split(",") if e]

    rows, outdir = {}, Path(args.results_dir) / CAMPAIGN
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "matrix.json"          # merge prior cells so refocusing engines never loses data
    if mp.exists():
        try:
            rows = {tuple(k.split("|")): v for k, v in json.load(open(mp)).items()}
        except Exception:
            pass
    for env in envs:
        for engine in engines:
            stats, was_cached = run_cell(engine, env, args.seed, args.results_dir)
            rows[(engine, env)] = stats
            print(f"[{engine:>6} x {env:<12}] verdict={stats.get('verdict','?'):<14} "
                  f"rounds={stats.get('rounds','-')} total={stats.get('total_time','-')}s "
                  f"verifier={stats.get('verifier_time','-')}s"
                  + ("  (cached)" if was_cached else ""), flush=True)
            (outdir / "matrix.json").write_text(json.dumps(
                {f"{e}|{v}": s for (e, v), s in rows.items()}, indent=2, default=str))
            print_matrix(rows, engines, envs)
    print("\nDONE verify_matrix  ->  results/verify_matrix/matrix.json", flush=True)


if __name__ == "__main__":
    main()
