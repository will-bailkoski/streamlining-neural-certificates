"""
The epsilon-L_V budget and the choice of decrease condition (Lipschitz chapter).

A self-contained analysis (not a CEGIS/refuter sweep) backing the supplementary
Lipschitz chapter's claim that the certifiable drift margin is a *budget*:

    for all x,   V(x) - E_w[V(x')]  <=  L_V * E_w||x'-x||          (Lipschitz budget)
    hence        eps <= L_V * delta_min,   delta_min = inf_x E_w||x'-x||,

so a fixed margin eps is capped by the SLOWEST-moving region (near the
equilibrium), and the *shape* of the demanded margin is a design choice.

Three measurements, all training + sampled/grid evaluation (no sound verify):
  E1  budget curve   : E_w||x'-x|| vs radius -> the budget is a property of the
                       dynamics; extracts delta_min per env.
  E2  eps frontier   : sweep fixed-eps -> (L_V via LipBaB, achieved worst-case
                       margin on a dense grid). Confirms kappa_max <= L_V and that
                       the achieved margin tracks the ceiling delta_min*L_V, not
                       the demanded eps (demanding more eps only inflates L_V).
  E3  decrease conds : the demanded-margin PROFILE of fixed eps vs geometric rho*V
                       vs adaptive gamma*E[d], against the ceiling -> budget
                       matching (fixed is flat/wasteful, adaptive tracks E[d]).
  --diagnostics adds a single-hp cross-env transfer table and a small-init
  non-degeneracy check (both HONEST negatives on same-scale benchmarks; see the
  chapter discussion).

    python -m experiments.lipschitz_budget                 # E1-E3 tables + results.json
    python -m experiments.lipschitz_budget --plot          # + figures into doc/Figures/
    python -m experiments.lipschitz_budget --diagnostics   # + transfer / non-degeneracy
    python -m experiments.lipschitz_budget --seed 1        # existence probe; single seed

Bounded (hard-sigmoid) certificate is REQUIRED: the budget is only a real
constraint when V is normalised to [0,1] (else L_V rescales freely with V), which
is exactly the boundedness the mab / montecarlo engines already need.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap, jit, value_and_grad
import jax.nn as jnn
import optax

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec, glorot_init
from src.verifiers.lipbab import lipschitz_v_bound

CAMPAIGN_TITLE = "lipschitz_budget"

# Bounded certificate (see module docstring: boundedness makes L_V, hence the
# budget, meaningful). NOT config.CERT_STRUCTURE (relu_pwl is unbounded).
CERT = "bounded_pwl"
HIDDEN = [16, 16]
TRAIN_STEPS = 300
N_TRAIN = 4        # successor samples per state during training
BATCH = 256
LR = 1e-2

CURVE_ENVS = ["linstoch2D", "linear2D"]     # E1 (floor vs no-floor contrast)
FRONTIER_ENV = "linstoch2D"                 # E2
EPS_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
PROFILE_ENV = "linstoch2D"                  # E3
# Representative, grid-valid hyperparameters for the three decrease conditions
# (calibrated by --diagnostics; adaptive gamma must stay small — larger gamma
# over-demands the far field, where a bounded V has no decrease capacity left).
PROFILE_HP = dict(fixed=0.02, geometric=0.10, adaptive=0.06)
TRANSFER_ENVS = ["linear2D", "linstoch2D", "linstoch3D"]

SPEC = get_spec(CERT)
V = SPEC.forward


# ----------------------------------------------------------------------
# trainer with a pluggable decrease condition
# ----------------------------------------------------------------------
def _make_update(env, margin_kind, n):
    bstep = vmap(env.step)

    @jit
    def loss_fn(p, x, key, mp):
        keys = jrn.split(key, n * x.shape[0]).reshape(n, x.shape[0], 2)
        xn = vmap(lambda kk: bstep(x, kk)[0])(keys)         # (n, B, d)
        vn = vmap(lambda xx: V(p, xx))(xn).mean(0)          # (B,)
        v = V(p, x)                                         # (B,)
        if margin_kind == "fixed":
            margin = mp
        elif margin_kind == "geometric":
            margin = mp * v
        elif margin_kind == "adaptive":
            margin = mp * jnp.linalg.norm(xn - x[None], axis=-1).mean(0)
        else:
            raise ValueError(margin_kind)
        return jnp.mean(jnn.relu(vn - v + margin))

    opt = optax.adam(LR)

    @jit
    def upd(p, os, x, key, mp):
        l, g = value_and_grad(loss_fn)(p, x, key, mp)
        u, os = opt.update(g, os)
        return optax.apply_updates(p, u), os, l

    return upd, opt


def train(env, params0, pool, margin_kind, mp, seed, steps=TRAIN_STEPS, thresh=1e-4):
    upd, opt = _make_update(env, margin_kind, N_TRAIN)
    params, os = params0, opt.init(params0)
    key = jrn.PRNGKey(seed)
    last, it = float("inf"), steps
    for i in range(steps):
        key, k1, k2 = jrn.split(key, 3)
        idx = jrn.randint(k1, (BATCH,), 0, pool.shape[0])
        params, os, l = upd(params, os, pool[idx], k2, float(mp))
        last = float(l)
        if last <= thresh and i > 30:
            it = i + 1
            break
    return params, last, (last <= thresh), it


def lipV(env, params, timeout=12.0):
    return float(lipschitz_v_bound(SPEC, params, env.domain.bounds,
                                   method="lipbab", timeout=timeout))


# ----------------------------------------------------------------------
# evaluation: per-state E[V(x')] and E||x'-x||
# ----------------------------------------------------------------------
def _fields(env, params, xs, key, n=96, chunk=4000):
    bstep = vmap(env.step)

    @jit
    def one(xc, key):
        keys = jrn.split(key, n * xc.shape[0]).reshape(n, xc.shape[0], 2)
        xn = vmap(lambda kk: bstep(xc, kk)[0])(keys)
        vn = vmap(lambda xx: V(params, xx))(xn).mean(0)
        dd = jnp.linalg.norm(xn - xc[None], axis=-1).mean(0)
        return vn, dd

    N = xs.shape[0]
    EVp, Ed = np.empty(N), np.empty(N)
    for i in range(0, N, chunk):
        key, k = jrn.split(key)
        vn, dd = one(xs[i:i + chunk], k)
        EVp[i:i + chunk] = np.asarray(vn)
        Ed[i:i + chunk] = np.asarray(dd)
    Vx = np.asarray(V(params, xs))
    return dict(V=Vx, EVp=EVp, Ed=Ed, dec=Vx - EVp,
                r=np.linalg.norm(np.asarray(xs), axis=-1)), key


def _dense_grid(env, g, key):
    """Valid (domain\\eq) grid points and their env-only E[d] (V-independent)."""
    b = np.asarray(env.domain.bounds)
    axes = [np.linspace(b[i, 0], b[i, 1], g) for i in range(env.dim)]
    mesh = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], -1)
    xs = jnp.asarray(mesh, jnp.float32)
    xs = xs[np.asarray(vmap(env.is_valid_domain)(xs))]
    return xs


def _radial(r, y, nb=26, lo=None, hi=None):
    lo = r.min() if lo is None else lo
    hi = r.max() if hi is None else hi
    edges = np.linspace(lo, hi, nb + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    w = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
    m = np.array([y[w == b].mean() if np.any(w == b) else np.nan for b in range(nb)])
    return ctr, m


# ----------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------
def e1_budget_curve(seed):
    print("\n=== E1: budget curve  E_w||x'-x|| vs radius ===")
    key = jrn.PRNGKey(seed)
    out = {}
    for name in CURVE_ENVS:
        env = make_env(name)
        xs, key = env.sample(key, 60000)
        f, key = _fields(env, glorot_init([env.dim, 4, 1]), xs, key, n=64)
        ctr, prof = _radial(f["r"], f["Ed"], nb=40)
        out[name] = dict(radius=ctr.tolist(), Ed=prof.tolist(),
                         delta_min=float(f["Ed"].min()), eq_r=float(env.eq_radius),
                         lip_f=float(env.lip_f))
        print(f"  {name:11s} delta_min={f['Ed'].min():.4f} eq_r={env.eq_radius:.3f} "
              f"E[d]@eq~{prof[0]:.3f} E[d]@far~{prof[-1]:.3f}")
    return out


def e2_frontier(seed):
    print(f"\n=== E2: fixed-eps frontier on {FRONTIER_ENV} (grid worst-case) ===")
    key = jrn.PRNGKey(seed)
    env = make_env(FRONTIER_ENV)
    xs = _dense_grid(env, 200, key)
    ed_fields, key = _fields(env, glorot_init([env.dim, 4, 1]), xs, key)
    Ed, r = ed_fields["Ed"], ed_fields["r"]
    dmin = float(Ed.min())
    pool, key = env.sample(key, 60000)
    p0 = glorot_init([env.dim] + HIDDEN + [1], jrn.PRNGKey(seed + 100))
    out = dict(delta_min=dmin, points=[])
    for eps in EPS_SWEEP:
        pt, loss, conv, _ = train(env, p0, pool, "fixed", eps, seed)
        f, key = _fields(env, pt, xs, key)
        Lv = lipV(env, pt)
        kappa = f["dec"] / np.maximum(Ed, 1e-6)
        row = dict(eps=eps, loss=loss, converged=bool(conv), L_V=Lv,
                   kappa_max=float(np.percentile(kappa, 99.5)),
                   achieved=float(np.percentile(f["dec"], 0.5)),
                   ceiling=dmin * Lv)
        out["points"].append(row)
        print(f"  eps={eps:5.3f} conv={conv!s:5s} L_V={Lv:6.3f} "
              f"kappa_max={row['kappa_max']:.3f} achieved={row['achieved']:+.4f} "
              f"ceiling={dmin * Lv:.4f}")
    return out


def e3_profiles(seed):
    print(f"\n=== E3: decrease-condition margin profiles on {PROFILE_ENV} ===")
    key = jrn.PRNGKey(seed)
    env = make_env(PROFILE_ENV)
    xs = _dense_grid(env, 200, key)
    base, key = _fields(env, glorot_init([env.dim, 4, 1]), xs, key)
    Ed, r = base["Ed"], base["r"]
    lo, hi = float(env.eq_radius), float(r.max())
    ctr, Ed_prof = _radial(r, Ed, lo=lo, hi=hi)
    pool, key = env.sample(key, 60000)
    p0 = glorot_init([env.dim] + HIDDEN + [1], jrn.PRNGKey(seed + 100))
    out = dict(radius=ctr.tolist(), Ed=Ed_prof.tolist())
    for method, mp in PROFILE_HP.items():
        pt, loss, conv, _ = train(env, p0, pool, method, mp, seed)
        f, key = _fields(env, pt, xs, key)
        Lv = lipV(env, pt)
        if method == "fixed":
            demand = np.full_like(r, mp)
        elif method == "geometric":
            demand = mp * f["V"]
        else:
            demand = mp * Ed
        _, dem = _radial(r, demand, lo=lo, hi=hi)
        _, dec = _radial(r, f["dec"], lo=lo, hi=hi)
        out[method] = dict(mp=mp, L_V=Lv, loss=loss,
                           demand=dem.tolist(), dec=dec.tolist())
        print(f"  {method:9s} mp={mp:5.3f} L_V={Lv:.3f} "
              f"demand@eq~{dem[0]:.4f} demand@far~{dem[-1]:.3f} "
              f"ratio={dem[-1] / max(dem[0], 1e-9):.1f}x")
    return out


def diagnostics(seed):
    """Honest negatives: single-hp transfer (weak on same-scale envs) and the
    small-init non-degeneracy check (geometric does NOT collapse here)."""
    print("\n=== diag: single-hp transfer (fixed eps vs adaptive gamma) ===")
    key = jrn.PRNGKey(seed)
    eps_g, gam_g = PROFILE_HP["fixed"], PROFILE_HP["adaptive"]
    trans = dict(eps=eps_g, gamma=gam_g, envs={})
    for name in TRANSFER_ENVS:
        env = make_env(name)
        pool, key = env.sample(key, 60000 if env.dim <= 2 else 120000)
        p0 = glorot_init([env.dim] + HIDDEN + [1], jrn.PRNGKey(seed + 100))
        row = {}
        for method, mp in (("fixed", eps_g), ("adaptive", gam_g)):
            pt, loss, conv, _ = train(env, p0, pool, method, mp, seed)
            f, key = _fields(env, pt, pool, key)
            row[method] = dict(loss=loss, converged=bool(conv), L_V=lipV(env, pt),
                               achieved=float(np.percentile(f["dec"], 0.5)),
                               delta_min=float(f["Ed"].min()))
            print(f"  {name:11s} {method:9s} mp={mp:.3f} conv={conv!s:5s} "
                  f"L_V={row[method]['L_V']:6.3f} achieved={row[method]['achieved']:+.4f} "
                  f"delta_min={row[method]['delta_min']:.3f}")
        trans["envs"][name] = row

    print("\n=== diag: non-degeneracy from small init (does geometric collapse?) ===")
    env = make_env(PROFILE_ENV)
    pool, key = env.sample(key, 60000)
    small = [(w * 0.05, b) for w, b in
             glorot_init([env.dim] + HIDDEN + [1], jrn.PRNGKey(seed + 200))]
    nd = {}
    for method, mp in (("geometric", PROFILE_HP["geometric"]),
                       ("adaptive", PROFILE_HP["adaptive"])):
        pt, loss, conv, _ = train(env, small, pool, method, mp, seed)
        Vx = np.asarray(V(pt, pool))
        nd[method] = dict(loss=loss, Vrange=float(Vx.max() - Vx.min()),
                          Vmean=float(Vx.mean()))
        print(f"  {method:9s} loss={loss:.4f} Vrange={Vx.max() - Vx.min():.3f} "
              f"Vmean={Vx.mean():.3f}")
    return dict(transfer=trans, nondegeneracy=nd)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def make_figures(rep, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIOLET, CORAL, GREEN, MUT, INK = "#5D50C6", "#D85A30", "#1D9E75", "#888780", "#2C2C2A"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: budget curve
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for name, col in zip(CURVE_ENVS, (VIOLET, CORAL)):
        d = rep["E1"][name]
        ax.plot(d["radius"], d["Ed"], "-", color=col, lw=2.4,
                label=f"{name}  (Lf={d['lip_f']:.2f})")
        ax.axhline(d["delta_min"], color=col, ls="--", lw=1.0, alpha=0.55)
        ax.axvline(d["eq_r"], color=col, ls=":", lw=1.1, alpha=0.7)
    ax.set_xlabel(r"radius $\|x\|$   (equilibrium $\to$ domain edge)")
    ax.set_ylabel(r"$\mathbb{E}_w\,\|x'-x\|$   (observed drift)")
    ax.set_title("Budget curve: the certifiable margin is set by the dynamics",
                 fontsize=11)
    ax.text(0.98, 0.05, r"dashed $=\delta_{\min}$   dotted $=$ eq radius",
            transform=ax.transAxes, fontsize=8.3, color=MUT, ha="right")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "lipschitz_budget_curve.png"); plt.close(fig)

    # Fig 2: eps frontier
    e2 = rep["E2"]; dm = e2["delta_min"]; pts = e2["points"]
    Lv = np.array([p["L_V"] for p in pts]); eps = np.array([p["eps"] for p in pts])
    ach = np.array([p["achieved"] for p in pts])
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    xx = np.linspace(0, Lv.max() * 1.05, 50)
    ax.plot(xx, dm * xx, "-", color=INK, lw=1.7, label=r"budget ceiling $\delta_{\min}L_V$")
    ax.fill_between(xx, dm * xx, (dm * xx.max()) * 1.4, color=CORAL, alpha=0.05)
    ax.plot(Lv, eps, "o", mfc="none", mec=MUT, ms=7, label=r"demanded $\varepsilon$")
    ax.scatter(Lv, ach, color=GREEN, s=48, zorder=5, label="achieved margin (grid)")
    for p in pts:
        ax.annotate(rf"{p['eps']:g}", (p["L_V"], p["eps"]), fontsize=7.3,
                    textcoords="offset points", xytext=(5, 3), color=MUT)
        if not p["converged"]:
            ax.annotate("cliff", (p["L_V"], p["eps"]), fontsize=8.0, color=CORAL,
                        textcoords="offset points", xytext=(4, -12))
    ax.set_xlabel(r"certificate Lipschitz constant $L_V$  (LipBaB, sound)")
    ax.set_ylabel("decrease margin")
    ax.set_title(r"Demanding more $\varepsilon$ buys only a steeper $V$", fontsize=11)
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "lipschitz_frontier.png"); plt.close(fig)

    # Fig 3: margin-demand profiles (normalised by L_V; ceiling = E[d])
    P = rep["E3"]; r = np.array(P["radius"]); Ed = np.array(P["Ed"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(r, Ed, "-", color=INK, lw=2.2,
            label=r"budget ceiling  $\mathbb{E}[d]$  (per unit $L_V$)")
    styles = dict(fixed=(CORAL, r"fixed $\varepsilon$"),
                  geometric=(VIOLET, r"geometric $\rho V$"),
                  adaptive=(GREEN, r"adaptive $\gamma\,\mathbb{E}[d]$"))
    norm = {}
    for m, (col, lab) in styles.items():
        norm[m] = np.array(P[m]["demand"]) / P[m]["L_V"]
        ax.plot(r, norm[m], "-", color=col, lw=2.3, label=f"{lab} (mp={P[m]['mp']:g})")
    ax.fill_between(r, norm["fixed"], Ed, color=CORAL, alpha=0.06)
    ax.text(0.62, 0.5, "wasted budget", transform=ax.transAxes,
            color="#993C1D", fontsize=9, ha="center")
    ax.set_xlabel(r"radius $\|x\|$   (equilibrium $\to$ domain edge)")
    ax.set_ylabel(r"demanded margin / $L_V$")
    ax.set_title("Budget-matching: fixed is flat, adaptive tracks the dynamics",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "lipschitz_profiles.png"); plt.close(fig)
    print(f"  wrote 3 figures -> {fig_dir.as_posix()}/lipschitz_*.png")


def main():
    p = argparse.ArgumentParser(description="Lipschitz eps-L_V budget experiment")
    p.add_argument("--seed", type=int, default=0, help="single-seed existence probe")
    p.add_argument("--plot", action="store_true", help="render figures into doc/Figures/")
    p.add_argument("--diagnostics", action="store_true",
                   help="also run transfer + non-degeneracy (honest negatives)")
    p.add_argument("--results_dir", default=f"results/{CAMPAIGN_TITLE}")
    p.add_argument("--fig_dir", default="doc/Figures")
    a = p.parse_args()

    rep = dict(seed=a.seed, cert=CERT, hidden=HIDDEN,
               E1=e1_budget_curve(a.seed),
               E2=e2_frontier(a.seed),
               E3=e3_profiles(a.seed))
    if a.diagnostics:
        rep["diagnostics"] = diagnostics(a.seed)

    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_seed{a.seed}.json").write_text(json.dumps(rep, indent=2))
    print(f"\nsaved {(out_dir / f'results_seed{a.seed}.json').as_posix()}")

    if a.plot:
        make_figures(rep, Path(a.fig_dir))


if __name__ == "__main__":
    main()
