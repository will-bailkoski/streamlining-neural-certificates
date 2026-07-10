"""
Thesis figure: grid reconstruction for ANY branch-and-bound verifier
(verification chapter) — the state of the partition at chosen points of the
run, every cell shaded by status:

    green  = verified      (sound upper bound <= 0 on the cell)
    amber  = inconclusive  (still live / will be split)
    red    = violating     (sound lower bound > 0: a counterexample cell)

Works identically for the bound-propagation engines (ibp / crown / alpha-crown,
one frame per B&B depth) and the MAB engine (one frame every `record` search
iterations) because both engines expose the same opt-in `record` hook that
returns {los, his, status} box frames.

Paired experiments: any campaign whose runs store a certificate
(pendulum_certs / cegis_verify_* for lirpa; cegis_verify_mab / milp_ndcerts
for mab).
Data read: <run_dir>/objects/grid_frames_<method>.npz — cached on first use by
re-running the engine on the stored certificate with record=True; later plot
tweaks never recompute.

    python -m visualisation.verifier_grid --method crown          # auto-pick run
    python -m visualisation.verifier_grid --method mab --gif
    python -m visualisation.verifier_grid --method ibp --run <dir> --panels 8
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from visualisation.common import (PUB_RC, STATUS_COLOR, STATUS_NAME,
                                  find_cert_run, load_cert, load_manifest,
                                  save, base_parser)
from src.plotting.utils import add_domain_border, add_equilibrium

LIRPA_METHODS = ("ibp", "crown", "alpha-crown")
# campaigns searched for a certificate when --run is absent, per engine family
LIRPA_CAMPAIGNS = ["pendulum_certs", "cegis_verify_crown", "cegis_verify_ibp"]
MAB_CAMPAIGNS = ["cegis_verify_mab", "milp_ndcerts"]


def pick_run(method: str, env: str | None) -> Path:
    for c in (LIRPA_CAMPAIGNS if method in LIRPA_METHODS else MAB_CAMPAIGNS):
        d = find_cert_run(c, env=env, verdict="verified") or find_cert_run(c, env=env, verdict=None)
        if d is not None:
            return d
    raise SystemExit(f"no certificate run found for method={method} — pass --run")


# ----------------------------------------------------------------------
# frames: cache in the run dir, record on first use
# ----------------------------------------------------------------------
def load_or_record(run_dir: Path, method: str, *, epsilon, noise_disc, max_depth,
                   record_every, max_samples, refresh) -> list[dict]:
    cache = run_dir / "objects" / f"grid_frames_{method}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache)
        n = int(z["n_frames"])
        return [dict(iter=int(z["iters"][i]), los=z[f"f{i}_los"], his=z[f"f{i}_his"],
                     status=z[f"f{i}_status"]) for i in range(n)]

    env, spec, params, args = load_cert(run_dir)
    import jax.random as jrn
    from src.verifiers.base import get_engine
    import src.verifiers  # noqa: F401

    print(f"recording {method} exploration on {args.get('env')} "
          f"(cert from {run_dir.name}) ...")
    if method in LIRPA_METHODS:
        r = get_engine(method).find_counterexample(
            env, spec, params, epsilon, key=jrn.key(0), noise_disc=noise_disc,
            max_depth=max_depth, device="cpu", record=True)
        frames = r.stats.get("frames", [])
        frames = [dict(iter=i, **f) for i, f in enumerate(frames)]
    else:
        r = get_engine("mab").find_counterexample(
            env, spec, params, epsilon, key=jrn.key(0), record=record_every,
            max_samples=max_samples)
        frames = r.stats.get("frames", [])
    if not frames:
        raise SystemExit("engine recorded no frames")
    verdict = {True: "verified", False: "counterexample", None: "inconclusive"}[r.verified]
    print(f"  -> {verdict}, {len(frames)} frames")

    payload = {"n_frames": np.array(len(frames)),
               "iters": np.array([f["iter"] for f in frames])}
    for i, f in enumerate(frames):
        payload[f"f{i}_los"] = np.asarray(f["los"])
        payload[f"f{i}_his"] = np.asarray(f["his"])
        payload[f"f{i}_status"] = np.asarray(f["status"])
    np.savez_compressed(cache, **payload)
    print(f"  cached frames -> {cache.as_posix()}")
    return frames


def accumulate_verified(frames: list[dict]) -> list[dict]:
    """lirpa frames hold only the boxes ALIVE at that depth — cells verified at
    earlier depths would vanish from later panels. Carry them forward so every
    panel shows the full partition. (mab frames already include retired cells.)"""
    out, acc_lo, acc_hi = [], [], []
    for f in frames:
        los = np.concatenate([np.asarray(f["los"])] + acc_lo) if acc_lo else np.asarray(f["los"])
        his = np.concatenate([np.asarray(f["his"])] + acc_hi) if acc_hi else np.asarray(f["his"])
        n_acc = sum(a.shape[0] for a in acc_lo)
        status = np.concatenate([np.asarray(f["status"]),
                                 np.ones(n_acc, dtype=np.int8)]) if n_acc else np.asarray(f["status"])
        out.append(dict(iter=f["iter"], los=los, his=his, status=status))
        done = np.asarray(f["status"]) == 1
        if done.any():
            acc_lo.append(np.asarray(f["los"])[done])
            acc_hi.append(np.asarray(f["his"])[done])
    return out


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def draw_frame(ax, env, frame, bounds, label):
    los, his, status = frame["los"], frame["his"], frame["status"]
    rects = [Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1])
             for lo, hi in zip(los, his)]
    colors = [STATUS_COLOR[int(s)] for s in status]
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="white",
                                      linewidths=0.15, alpha=0.8))
    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    counts = {s: int(np.sum(status == s)) for s in (1, 0, 2)}
    ax.set_title(f"{label}\n({counts[1]} {STATUS_NAME[1]}, {counts[0]} open, "
                 f"{counts[2]} {STATUS_NAME[2]})", fontsize=8.5)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def montage(env, method, frames, bounds, n_show, unit, rows=1):
    idx = sorted(set(np.linspace(0, len(frames) - 1, min(n_show, len(frames))).astype(int)))
    ncols = int(np.ceil(len(idx) / rows))
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(rows, ncols, figsize=(2.9 * ncols, 3.3 * rows),
                                 constrained_layout=True, squeeze=False)
        flat = axes.ravel()
        for ax, i in zip(flat, idx):
            draw_frame(ax, env, frames[i], bounds, f"{unit} {frames[i]['iter']}")
        for ax in flat[len(idx):]:          # blank any unused panels in the grid
            ax.axis("off")
        fig.suptitle(f"{env.name} — {method} partition by verification status",
                     fontsize=13)
    return fig


def gif(env, method, frames, bounds, out_path: Path, fps=2):
    import matplotlib.animation as animation
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)

        def update(i):
            ax.clear()
            draw_frame(ax, env, frames[i], bounds, f"frame {frames[i]['iter']}")
            return []

        anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  -> {out_path.as_posix()}  ({len(frames)} frames)")


def main():
    p = base_parser("Grid reconstruction of a verifier run, cells shaded by status")
    p.add_argument("--method", default="crown",
                   help="ibp | crown | alpha-crown | mab")
    p.add_argument("--run", default=None, help="run dir holding the certificate")
    p.add_argument("--env", default=None, help="env filter for the auto-picked run")
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--noise_disc", type=int, default=8, help="(lirpa) noise cells")
    p.add_argument("--max_depth", type=int, default=12, help="(lirpa) B&B depth cap")
    p.add_argument("--record_every", type=int, default=50, help="(mab) frame cadence")
    p.add_argument("--max_samples", type=int, default=20_000_000, help="(mab) budget")
    p.add_argument("--panels", type=int, default=6, help="montage panel count")
    p.add_argument("--rows", type=int, default=1, help="montage rows (panels wrap across rows)")
    p.add_argument("--gif", action="store_true", help="also write the animation")
    p.add_argument("--refresh", action="store_true", help="re-record the frames")
    a = p.parse_args()

    run_dir = Path(a.run) if a.run else pick_run(a.method, a.env)
    frames = load_or_record(run_dir, a.method, epsilon=a.epsilon,
                            noise_disc=a.noise_disc, max_depth=a.max_depth,
                            record_every=a.record_every, max_samples=a.max_samples,
                            refresh=a.refresh)
    if a.method in LIRPA_METHODS:
        frames = accumulate_verified(frames)

    env, _, _, args = load_cert(run_dir)
    if env.dim != 2:
        raise SystemExit(f"grid rendering supports 2D envs (got dim={env.dim})")
    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    unit = "depth" if a.method in LIRPA_METHODS else "iteration"

    env_name = args.get("env", "env")
    fig = montage(env, a.method, frames, bounds, a.panels, unit, rows=a.rows)
    stem = f"{env_name}_{a.method}_grid"
    save(fig, "verifier_grid", stem, f"{stem}.png" if a.doc else None)
    if a.gif:
        from visualisation.common import FIG_ROOT
        gif(env, a.method, frames, bounds,
            FIG_ROOT / "verifier_grid" / f"{env_name}_{a.method}_evolution.gif")


if __name__ == "__main__":
    main()
