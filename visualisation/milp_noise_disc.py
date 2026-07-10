"""
Thesis figures: the MILP noise-discretisation sweet spot (verification chapter,
Figures/milp_noise_disc.png) and the size of the underlying optimisation
problem (Figures/milp_problem_size.png).

Paired experiment: experiments/milp_noise_disc.py  (campaign "milp_noise_disc";
                   certificates under campaign "milp_ndcerts")
Data read:         results/milp_noise_disc/results.json

Figure A (time): MILP solve time vs noise bins b per dimension, one line per
state dimension D, markers = verdict (verified / phantom counterexample /
timeout). Too few bins -> the noise over-approximation is so loose the solver
"refutes" a valid certificate (phantom); too many -> the model explodes.

Figure B (size): number of Gurobi variables / binaries / constraints vs b.
The encoding duplicates the network once per noise cell (b^D cells), which is
exactly why the sweet spot narrows with dimension. Sizes come from the runs'
recorded n_vars stats when present; for older results they are recomputed by
BUILDING (not solving) each model from the stored campaign certificates, and
cached in results/milp_noise_disc/model_sizes.json.

    python -m visualisation.milp_noise_disc
    python -m visualisation.milp_noise_disc --doc
"""

from __future__ import annotations
import json

import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, campaign_dir, save, base_parser)

CAMPAIGN = "milp_noise_disc"       # == experiments.milp_noise_disc.CAMPAIGN
CERT_CAMPAIGN = "milp_ndcerts"     # where its ensure_cert stores certificates
DOC_NAME_TIME = "milp_noise_disc.png"
DOC_NAME_SIZE = "milp_problem_size.png"

DIM_COLOR = {2: "#2b6cb0", 3: "#c05621", 4: "#6b46c1", 5: "#1D9E75"}
VERDICT_MARKER = {"verified": "o", "phantom_ce": "X", "counterexample": "X",
                  "timeout": "^", "inconclusive": "s"}

# Dimensions shown in the published sweet-spot figure (Figure A). The 3D sweep is
# retained in results.json but dropped from the thesis figure: it never leaves the
# phantom-counterexample regime at any bin count, so it contributes little to the
# sweet-spot story the figure tells. Set to None to plot every dimension present.
DOC_DIMS = [2]


def load_results() -> dict:
    path = campaign_dir(CAMPAIGN) / "results.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m experiments.milp_noise_disc` first")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------
# Figure A: solve time vs noise bins
# ----------------------------------------------------------------------
def figure_time(results: dict):
    tl = results["meta"]["time_limit"]
    dims = sorted({r["D"] for r in results["runs"]})
    if DOC_DIMS is not None:
        dims = [D for D in dims if D in DOC_DIMS]
    plotted_envs = sorted({r["env"] for r in results["runs"] if r["D"] in dims})
    sys_label = plotted_envs[0] if len(plotted_envs) == 1 else r"linstoch$d$D"
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for D in dims:
            runs = [r for r in results["runs"] if r["D"] == D]
            c = DIM_COLOR.get(D, "#444")
            base = runs[0]["sweep"]
            vx = [r["nd"] for r in base if r["verdict"] == "verified"]
            vy = [max(r["time"], 0.05) for r in base if r["verdict"] == "verified"]
            ax.plot(vx, vy, "-", color=c, lw=1.6, alpha=0.85,
                    label=f"$d_w$={D} ({runs[0]['env']})")
            for r in runs:
                for row in r["sweep"]:
                    y = tl if row["verdict"] == "timeout" else max(row["time"], 0.05)
                    ax.scatter(row["nd"], y, marker=VERDICT_MARKER.get(row["verdict"], "o"),
                               s=66, color=c, edgecolor="k", linewidth=0.5, zorder=3)
        ax.axhline(tl, ls="--", color="k", lw=1, alpha=0.6)
        ax.text(ax.get_xlim()[1], tl, f"  timeout ({tl:g}s)", va="center", ha="left", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel(r"noise bins per dimension  $b$   (cells $= b^{\,d_w}$)")
        ax.set_ylabel("MILP solve time  (s, log)")
        ax.set_title(f"MILP noise-discretisation sweet spot on {sys_label}\n"
                     "○ verified    × phantom counterexample (too loose)    △ timeout (too big)")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.set_xticks(sorted({row["nd"] for r in results["runs"] if r["D"] in dims
                              for row in r["sweep"]}))
        ax.grid(True, which="both", alpha=0.25)
    return fig


# ----------------------------------------------------------------------
# Figure B: model size vs noise bins
# ----------------------------------------------------------------------
def _count_by_building(env_name: str, nd: int, seed: int):
    """Build (never solve) the sweep's MILP and read its size. Uses the stored
    campaign certificate, exactly like the sweep did."""
    import gurobipy as gp
    from gurobipy import GRB
    from src.benchmarks.utils import make_env
    from src.certificates.structures import get_spec
    from src.certificates.encoders import encode_gurobi
    from src.verifiers.milp import _successor_bounds
    from visualisation.common import run_dirs, load_manifest
    from src.results.recorder import load_params

    cert = None
    for d in run_dirs(CERT_CAMPAIGN):
        a = load_manifest(d).get("args", {})
        if a.get("env") == env_name and int(a.get("seed", -1)) == seed \
                and (d / "objects" / "certificate.npz").exists():
            cert = d, a
            break
    if cert is None:
        return None
    d, a = cert
    env = make_env(env_name)
    spec = get_spec(a.get("cert_structure", "bounded_pwl"))
    params = load_params(d / "objects" / "certificate.npz")

    domain_bounds = (env.domain.bounds[:, 0], env.domain.bounds[:, 1])
    succ = _successor_bounds(env)
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(env.dim, lb=-GRB.INFINITY)
    xs = list(x.values())
    env.domain.encode_gurobi_inclusion(m, xs)
    env.equilibrium.encode_gurobi_exclusion(m, xs)
    Vx = encode_gurobi(spec, m, params, xs, input_bounds=domain_bounds, name="V", tag="t")
    branches = env.gurobi_step(m, [(1.0, xs)], noise_disc=nd)
    import gurobipy as _gp
    _gp.quicksum(w * encode_gurobi(spec, m, params, xn, input_bounds=succ, name="V", tag=f"n{i}")
                 for i, (w, xn) in enumerate(branches))
    m.update()
    return dict(n_vars=int(m.NumVars), n_bin_vars=int(m.NumBinVars),
                n_constrs=int(m.NumConstrs), cells=len(branches))


def model_sizes(results: dict) -> dict:
    """{(D, nd): size dict} from recorded stats, else rebuilt+cached."""
    cache_path = campaign_dir(CAMPAIGN) / "model_sizes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out, dirty = {}, False
    for r in results["runs"]:
        for row in r["sweep"]:
            key = f"{r['D']}:{row['nd']}"
            if "n_vars" in row:                      # recorded by newer engine runs
                out[key] = {k: row[k] for k in ("n_vars", "n_bin_vars", "n_constrs")}
                out[key]["cells"] = row.get("cells")
                continue
            if key not in cache:
                got = _count_by_building(r["env"], row["nd"], r["seed"])
                if got is None:
                    continue
                cache[key] = got
                dirty = True
            out[key] = cache[key]
    if dirty:
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"  cached model sizes -> {cache_path.as_posix()}")
    return out


def figure_size(results: dict, sizes: dict):
    dims = sorted({r["D"] for r in results["runs"]})
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for D in dims:
            nds = sorted({row["nd"] for r in results["runs"] if r["D"] == D
                          for row in r["sweep"] if f"{D}:{row['nd']}" in sizes})
            if not nds:
                continue
            c = DIM_COLOR.get(D, "#444")
            nv = [sizes[f"{D}:{b}"]["n_vars"] for b in nds]
            nb = [sizes[f"{D}:{b}"]["n_bin_vars"] for b in nds]
            ax.plot(nds, nv, "-o", color=c, lw=1.8, ms=4.5, label=f"$d_w$={D}  variables")
            ax.plot(nds, nb, "--s", color=c, lw=1.4, ms=4, alpha=0.75,
                    label=f"$d_w$={D}  binaries")
        ax.set_yscale("log")
        ax.set_xlabel(r"noise bins per dimension  $b$")
        ax.set_ylabel("Gurobi model size  (log)")
        ax.set_title("The MILP grows as $b^{\\,d_w}$ network copies:\n"
                     "one $V(x')$ encoding per noise cell", fontsize=11)
        ax.set_xticks(sorted({int(k.split(':')[1]) for k in sizes}))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8.5, framealpha=0.9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig


def main():
    p = base_parser("MILP noise-disc sweet spot + problem-size figures")
    a = p.parse_args()
    results = load_results()

    fig = figure_time(results)
    save(fig, "milp_noise_disc", "milp_noise_disc", DOC_NAME_TIME if a.doc else None)

    sizes = model_sizes(results)
    if sizes:
        fig = figure_size(results, sizes)
        save(fig, "milp_noise_disc", "milp_problem_size", DOC_NAME_SIZE if a.doc else None)
    else:
        print("  (no model sizes available — no stats recorded and no local certs to rebuild)")


if __name__ == "__main__":
    main()
