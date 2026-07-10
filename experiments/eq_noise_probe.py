"""
Local probe: does (a) enlarging the VERIFY equilibrium after training and/or
(b) shrinking the noise variance rescue the sound verifiers on linstoch2D?

Motivated by the cluster campaign grinding to a halt: linear2D/linstoch2D MAB
runs verified the vast majority of the domain then stalled on a few boxes at
the equilibrium rim — exactly where the trainer never places samples
(env.sample rejection-samples domain \\ eq) and where the hinge loss leaves the
margin at its worst. Two knobs, tested on a FIXED refuter-hardened certificate
per setting so every engine faces the same problem:

  * eq trick   : train with eq_margin m_t, verify with eq_margin m_v >= m_t.
                 m_v > m_t excludes the thin worst-margin shell from the check.
                 (m_t < 1.5 additionally trains INSIDE the default equilibrium.)
  * variance   : noise_scale sweep. eq_radius auto-derives from the noise floor
                 (∝ scale) and the known-V rim margin scales ∝ scale², so this
                 is NOT obviously monotone — that is the measurement.

    python -m experiments.eq_noise_probe --phase a          # eq trick, ns=0.1
    python -m experiments.eq_noise_probe --phase b          # variance sweep
    python -m experiments.eq_noise_probe --list             # print the grid

Rows append to results/eq_noise_probe/probe.csv as they finish (resumable:
existing rows are skipped). Certificates cache under certs/ keyed by
(noise_scale, train margin, seed) so re-runs and both phases share them.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

from src.benchmarks.linear_stochastic import LinearStochasticEnv
from src.certificates.structures import get_spec, glorot_init
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401  (register engines)
from src.verifiers.drift import make_drift, recheck_violation
from src.refuters.base import get_refuter
import src.refuters  # noqa: F401
from src.training.training_cycles import train_fixed_dataset, cegis_trainer
from src.results.recorder import _params_to_npz, load_params
from experiments.common.runner import expand_cex

ROOT = Path("results") / "eq_noise_probe"
CSV_PATH = ROOT / "probe.csv"
CERT_DIR = ROOT / "certs"

HIDDEN = [4, 4]                 # linstoch2D width from the mab campaign probe
CERT_STRUCTURE = "bounded_pwl"  # bounded output — sound for mab, exact in milp
EPSILON = 1e-3                  # verify threshold (campaign value)
TRAIN_EPS_LADDER = [1e-2, 5e-3, 2.5e-3, 1.25e-3]  # first margin that seeds to 0
MC_SAMPLES_OBJ = 128            # refuter-objective draws (campaign value)
N_TRAIN = 4000
HARDEN_MAX_ROUNDS = 40
HARDEN_CLEAN_PASSES = 2

# eq-margin (train, verify) pairs — phase A, noise_scale fixed at 0.1
EQ_PAIRS = [
    (1.5, 1.5),   # baseline: campaign geometry
    (1.5, 2.0),   # verify-fatter-eq only (no retrain needed in principle)
    (1.5, 3.0),
    (1.1, 1.5),   # train inside the default eq, verify the default
    (1.1, 2.0),
]

# variance sweep — phase B, eq pair fixed at baseline (1.5, 1.5)
NOISE_SCALES_B = [0.05, 0.2]    # 0.1 baseline is covered by phase A row 1

# dose-response refinement — phase C fills the 2.0 < m_v < 3.0 gap (seed 0),
# phase D replicates the milp/crown dose curve on fresh seeds
EQ_PAIRS_C = [(1.5, 2.5), (1.5, 3.5)]
EQ_PAIRS_D = [(1.5, 1.5), (1.5, 2.0), (1.5, 2.5), (1.5, 3.0)]

MAB_HP = dict(
    grid_per_dim=16, significance=0.05, range_bound=1.0,
    k=256, sample_per_select=64, max_boxes=2**21, max_samples=200_000_000,
    timeout=900.0, progress_interval=4000, mc_samples=1,
)
MILP_HP = dict(time_limit=600.0)
LIRPA_HP = dict(split="longest", max_depth=60, max_boxes=500_000, timeout=300.0)

# (engine, per-phase noise_disc lists). mab samples the noise directly (nd n/a).
ENGINES_A = [("milp", [4]), ("mab", [None]),
             ("ibp", [4]), ("crown", [4]), ("alpha-crown", [4])]
ENGINES_B = [("milp", [2, 4, 6]), ("mab", [None]),
             ("ibp", [1, 2, 4]), ("crown", [1, 2, 4]), ("alpha-crown", [4])]
ENGINES_C = [("milp", [4]), ("crown", [4]), ("mab", [None])]
ENGINES_D = [("milp", [4]), ("crown", [4])]

FIELDS = [
    "phase", "noise_scale", "eq_margin_train", "eq_margin_verify",
    "eq_radius_train", "eq_radius_verify", "seed", "train_eps", "seed_loss",
    "harden_rounds", "harden_clean", "engine", "noise_disc", "verdict",
    "genuine_cex", "time", "v_hit", "v_samples", "v_iterations", "v_boxes",
    "v_live_at_stop", "v_obj_bound", "v_lip", "error",
    "scan_worst_drift", "scan_worst_radius", "scan_hot_frac", "scan_bulk_worst",
]


def make_ls_env(noise_scale, eq_margin, noise_disc=2):
    return LinearStochasticEnv(noise_scale=noise_scale, eq_margin=eq_margin,
                               noise_disc=noise_disc)


# ----------------------------------------------------------------------------
# Certificate: seed-train to zero loss, then harden with the whale refuter
# ----------------------------------------------------------------------------


def _train_hp(spec):
    return dict(batch_size=256, lr=1e-2, n=16, decay=0.9, max_epochs=300,
                threshold=0.0, certificate=spec.forward, printing=False)


def get_certificate(noise_scale, eq_margin_train, seed, spec):
    """Train (or load cached) a refuter-hardened cert for this env setting."""
    tag = f"ns{noise_scale:g}_eqm{eq_margin_train:g}_s{seed}"
    npz, meta_p = CERT_DIR / f"{tag}.npz", CERT_DIR / f"{tag}.json"
    if npz.exists() and meta_p.exists():
        return load_params(npz), json.loads(meta_p.read_text())

    env = make_ls_env(noise_scale, eq_margin_train)
    thp = _train_hp(spec)
    dom_mid = 0.5 * (env.domain.bounds[:, 0] + env.domain.bounds[:, 1])

    params = seed_loss = train_eps = full_x = None
    for eps_t in TRAIN_EPS_LADDER:
        init = glorot_init([env.dim] + HIDDEN + [1],
                           jrn.fold_in(jrn.key(seed), 99), input_center=dom_mid)
        params, _, seed_loss, full_x = train_fixed_dataset(
            env, eps_t, init, N=N_TRAIN,
            dataset_key=jrn.fold_in(jrn.key(seed), 7), **thp)
        train_eps = eps_t
        print(f"    [{tag}] train_eps={eps_t:g}  seed_loss={seed_loss:.2e}")
        if seed_loss <= 0.0:
            break

    # refuter-hardening: whale (BBOB winner) until 2 consecutive clean passes,
    # so the engines below time out on near-valid certs instead of instantly
    # returning the counterexample the trainer just never saw.
    gen = cegis_trainer(env, train_eps, params, seed_dataset=full_x, **thp)
    params, _ = next(gen)
    refuter = get_refuter("whale")
    rhp = dict(budget=8192, batch_size=64, a_max=1.5, b=0.5, p=0.75)
    domain = jnp.asarray(env.domain.bounds)

    def objective(ps, xs, k):
        V = lambda z: spec.forward(ps, z)
        _, _, mcb = make_drift(env, V, EPSILON, n=MC_SAMPLES_OBJ)
        return jnp.where(jax.vmap(env.is_valid_domain)(xs), mcb(xs, k), -jnp.inf)

    run_refuter = jax.jit(lambda ps, rk: refuter(
        lambda xs, k: objective(ps, xs, k), domain, rk, **rhp))

    key = jrn.fold_in(jrn.key(seed), 3)
    clean, rounds = 0, 0
    for rounds in range(1, HARDEN_MAX_ROUNDS + 1):
        key, rk = jrn.split(key)
        bx, by, _, _ = run_refuter(params, rk)
        jax.block_until_ready((bx, by))
        found = float(by) > 0.0
        if found:
            key, ck = jrn.split(key)
            genuine, _ = recheck_violation(env, spec, params, EPSILON,
                                           np.asarray(bx).reshape(-1), ck, n=4096)
            found = bool(genuine)
        if not found:
            clean += 1
            if clean >= HARDEN_CLEAN_PASSES:
                break
            continue
        clean = 0
        key, ek = jrn.split(key)
        ce = expand_cex(np.asarray(bx)[None, :], 8, ek, env,
                        region=None, ball_radius=0.05)
        params, loss = gen.send(jnp.asarray(ce))
        print(f"    [{tag}] harden round {rounds}: CE at obj {float(by):+.2e}, "
              f"retrained to loss {loss:.2e}")

    meta = dict(noise_scale=noise_scale, eq_margin_train=eq_margin_train,
                seed=seed, train_eps=train_eps, seed_loss=float(seed_loss),
                harden_rounds=rounds, harden_clean=clean >= HARDEN_CLEAN_PASSES,
                eq_radius_train=float(env.eq_radius))
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **_params_to_npz(params))
    meta_p.write_text(json.dumps(meta, indent=1))
    return load_params(npz), meta  # reload: uniform numpy types either path


# ----------------------------------------------------------------------------
# Empirical margin scan — the WHY for each timing row
# ----------------------------------------------------------------------------


def margin_scan(env, spec, params, seed, n_points=32768, n_mc=256):
    """MC drift over valid samples: worst margin, where it sits, how hot the rim is."""
    V = lambda z: spec.forward(params, z)
    _, _, mcb = make_drift(env, V, EPSILON, n=n_mc)
    key = jrn.fold_in(jrn.key(seed), 11)
    key, sk = jrn.split(key)
    xs, _ = env.sample(sk, n_points)
    drifts = []
    for i in range(0, n_points, 4096):
        key, dk = jrn.split(key)
        drifts.append(np.asarray(mcb(xs[i:i + 4096], dk)))
    d = np.concatenate(drifts)
    r = np.linalg.norm(np.asarray(xs), axis=1)
    i_worst = int(np.argmax(d))  # drift closest to (or past) the 0 threshold
    bulk = r > 2.0 * float(env.eq_radius)
    return dict(
        scan_worst_drift=float(d.max()),         # > 0 == empirically violated
        scan_worst_radius=float(r[i_worst]),
        scan_hot_frac=float(np.mean(d > -EPSILON)),
        scan_bulk_worst=float(d[bulk].max()) if bulk.any() else np.nan,
    )


# ----------------------------------------------------------------------------
# One verify call
# ----------------------------------------------------------------------------

_MAB_ERR = re.compile(r"after (\d+) iterations, (\d+) samples, (\d+) live boxes")


def run_engine(engine_name, env, spec, params, seed, noise_disc):
    hp = {"milp": dict(MILP_HP), "mab": dict(MAB_HP)}.get(
        engine_name, dict(LIRPA_HP))
    if noise_disc is not None:
        hp["noise_disc"] = noise_disc
    engine = get_engine(engine_name)
    t0 = time.perf_counter()
    try:
        r = engine.find_counterexample(env, spec, params, EPSILON,
                                       key=jrn.key(seed), **hp)
    except Exception as e:  # lirpa OOM / gurobi license etc: record, keep going
        return dict(verdict="error", error=str(e)[:300],
                    time=round(time.perf_counter() - t0, 3))
    wall = time.perf_counter() - t0
    stats = r.stats or {}
    row = dict(
        verdict={True: "verified", False: "counterexample", None: "inconclusive"}[r.verified],
        time=round(float(stats.get("time", wall)), 3),
        v_hit=stats.get("hit", ""),
        v_samples=stats.get("samples", ""),
        v_iterations=stats.get("iterations", ""),
        v_boxes=stats.get("total_boxes", stats.get("n_boxes", "")),
        v_obj_bound=stats.get("obj_bound", stats.get("resid_bound", "")),
        v_lip=stats.get("reward_lipschitz", stats.get("lip_v", "")),
        error=str(stats.get("error", ""))[:300],
    )
    # mab's budget path buries its progress in the RuntimeError message — parse it
    m = _MAB_ERR.search(row["error"])
    if m:
        row["v_iterations"], row["v_samples"], row["v_live_at_stop"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if r.violation is not None:
        genuine, _ = recheck_violation(env, spec, params, EPSILON,
                                       np.asarray(r.violation).reshape(-1),
                                       jrn.fold_in(jrn.key(seed), 1), n=4096)
        row["genuine_cex"] = bool(genuine)
    return row


# ----------------------------------------------------------------------------
# Grid driver
# ----------------------------------------------------------------------------


def build_grid(phase, seed):
    grid = []
    if phase in ("a", "all"):
        for m_t, m_v in EQ_PAIRS:
            for eng, nds in ENGINES_A:
                for nd in nds:
                    grid.append(dict(phase="a", noise_scale=0.1,
                                     eq_margin_train=m_t, eq_margin_verify=m_v,
                                     seed=seed, engine=eng, noise_disc=nd))
    if phase in ("b", "all"):
        for ns in NOISE_SCALES_B:
            for eng, nds in ENGINES_B:
                for nd in nds:
                    grid.append(dict(phase="b", noise_scale=ns,
                                     eq_margin_train=1.5, eq_margin_verify=1.5,
                                     seed=seed, engine=eng, noise_disc=nd))
    if phase == "c":  # dose-response gap + a bigger-budget mab retry at 2.0
        MAB_HP.update(max_samples=500_000_000, timeout=1800.0)
        for m_t, m_v in EQ_PAIRS_C:
            for eng, nds in ENGINES_C:
                for nd in nds:
                    grid.append(dict(phase="c", noise_scale=0.1,
                                     eq_margin_train=m_t, eq_margin_verify=m_v,
                                     seed=seed, engine=eng, noise_disc=nd))
        grid.append(dict(phase="c", noise_scale=0.1, eq_margin_train=1.5,
                         eq_margin_verify=2.0, seed=seed, engine="mab",
                         noise_disc=None))
    if phase == "d":  # seed replication of the milp/crown dose curve
        for m_t, m_v in EQ_PAIRS_D:
            for eng, nds in ENGINES_D:
                for nd in nds:
                    grid.append(dict(phase="d", noise_scale=0.1,
                                     eq_margin_train=m_t, eq_margin_verify=m_v,
                                     seed=seed, engine=eng, noise_disc=nd))
    return grid


def row_key(r):
    return (str(r["phase"]), f'{float(r["noise_scale"]):g}',
            f'{float(r["eq_margin_train"]):g}', f'{float(r["eq_margin_verify"]):g}',
            str(r["seed"]), str(r["engine"]), str(r["noise_disc"]))


def existing_keys():
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def append_row(row):
    ROOT.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--phase", choices=["a", "b", "c", "d", "all"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true", help="print grid and exit")
    args = p.parse_args()

    grid = build_grid(args.phase, args.seed)
    done = existing_keys()
    todo = [g for g in grid if row_key(g) not in done]
    print(f"grid: {len(grid)} cells, {len(grid) - len(todo)} done, {len(todo)} to run")
    if args.list:
        for g in todo:
            print("  ", g)
        return

    spec = get_spec(CERT_STRUCTURE)
    scan_cache, cert_cache = {}, {}

    for i, g in enumerate(todo):
        ns, m_t, m_v = g["noise_scale"], g["eq_margin_train"], g["eq_margin_verify"]
        ckey = (ns, m_t, g["seed"])
        print(f"\n[{i+1}/{len(todo)}] ns={ns:g} eq {m_t:g}->{m_v:g} "
              f"engine={g['engine']} nd={g['noise_disc']}")
        if ckey not in cert_cache:
            cert_cache[ckey] = get_certificate(ns, m_t, g["seed"], spec)
        params, meta = cert_cache[ckey]

        nd_env = g["noise_disc"] if g["noise_disc"] else 2
        env_v = make_ls_env(ns, m_v, noise_disc=nd_env)

        skey = (ns, m_t, m_v, g["seed"])
        if skey not in scan_cache:
            scan_cache[skey] = margin_scan(env_v, spec, params, g["seed"])
            s = scan_cache[skey]
            print(f"    scan: worst drift {s['scan_worst_drift']:+.2e} at r="
                  f"{s['scan_worst_radius']:.3f} (eq_v r={env_v.eq_radius:.3f}), "
                  f"bulk worst {s['scan_bulk_worst']:+.2e}, hot frac {s['scan_hot_frac']:.4f}")

        res = run_engine(g["engine"], env_v, spec, params, g["seed"], g["noise_disc"])
        print(f"    -> {res['verdict']}  {res.get('time', float('nan')):.1f}s"
              + (f"  hit={res['v_hit']}" if res.get("v_hit") else "")
              + (f"  genuine={res['genuine_cex']}" if "genuine_cex" in res else ""))

        append_row({**g, **res, **scan_cache[skey],
                    "eq_radius_train": meta["eq_radius_train"],
                    "eq_radius_verify": float(env_v.eq_radius),
                    "train_eps": meta["train_eps"], "seed_loss": meta["seed_loss"],
                    "harden_rounds": meta["harden_rounds"],
                    "harden_clean": meta["harden_clean"]})

    print("\nall cells done ->", CSV_PATH)


if __name__ == "__main__":
    main()
