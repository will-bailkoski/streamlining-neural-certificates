# Streamlining Neural Certificate Synthesis

Code accompanying the MSc thesis **"Streamlining Neural Certificate Synthesis"**
(William Bailkoski, 2026). Every verifier, refuter, experiment driver, and thesis
figure is produced by the code in this repository: the `experiments/` launchers
write campaigns to `results/<campaign>/<tag>/`, and the `visualisation/` scripts
render the thesis figures from those results (`--doc` emits them under the exact
names the thesis includes).

## The framework: neural certificate verification, unified

Train and verify **neural certificates** (supermartingale / Lyapunov functions `V`)
for stochastic dynamical systems, and compare verification engines *fairly* on the
same problem.

The design principle is **one definition, many engines**. A single `Env` defines the
dynamics, noise, domain and equilibrium; a single `CertificateSpec` defines the
parametric form of `V`; and every verification engine checks the *same* drift
condition over the *same* system:

```
                 verified  ⇔   E_w[ V(f(x, w)) ] − V(x) + ε  ≤  0
                            for all  x ∈ domain \ equilibrium
```

A counterexample is a state where that quantity is `> 0`. Because all engines
consume one source of truth, a "switched-linear" experiment means the same system
whether it is checked by SMT, MILP, autoLiRPA or sampling — which is what makes
cross-engine comparisons sound and fair.

## Layout

```
src/
  benchmarks/    one Env per system (dynamics + noise + domain + equilibrium)
  certificates/  one CertificateSpec per V structure + z3/gurobi/torch encoders
  verifiers/     the engines behind one interface + the shared drift functional
  training/      CEGIS trainer (counterexample-guided)
  results/       Recorder (run.json + metrics.csv + npz params; no pickle)
  experiment.py  config-driven CEGIS runner (entrypoint)
experiments/     config generation + the fairness harness
tests/           cross-backend consistency / soundness guard rails
```

### Environments (`src/benchmarks/utils.py::DIRECTORY`)

| name | system | noise | engines |
|------|--------|-------|---------|
| `linear2D`   | switched linear, default 2D matrices (Bernoulli) | discrete (exact) | all |
| `linear3D`, `linear4D`, `linear2D_rand`, ... | switched linear, **random** matrices at a target spectral norm, any dimension | discrete (exact) | all |
| `doublewell` | gradient flow, double-well potential | continuous (Uniform) | sampling, MILP, SMT, autoLiRPA |
| `pendulum`   | inverted pendulum, trained NN controller (nonlinear) | internal (Triangular) | sampling, autoLiRPA |

Convention: `_dynamics` is the core map; additive noise lives in `_pre`/`_post`
and is applied identically in the jax rollout and the symbolic encodings.

Build arbitrary-dimension linear benchmarks directly:

```python
from src.benchmarks.linear_system import SwitchedLinearEnv
env = SwitchedLinearEnv.random_construction(ndims=5, lip=0.8)  # two random L2-contractive modes
# known supermartingale: V(x) = ||x||_2  (V(x) = |x|_1 for the default L1 matrices)
```

The inverted pendulum is nonlinear (a `sin` term), so SMT/MILP raise; it is
verified with the sampling and autoLiRPA engines. Its controller is trained by policy gradient
with a fixed seed, so `make_env("pendulum")` rebuilds the same env each time (no
pickle, no weight files) — at the cost of a short training step on construction.

### Certificate structures (`src/certificates/structures.py::DIRECTORY`)

`bounded_pwl` (ReLU + hard-sigmoid, `V∈[0,1]`), `clip_pwl`, `relu_pwl`,
`bottom_pwl`, `tanh`. The jax forward, the z3/gurobi encodings and the torch
module are all derived from one spec, so training and verification never diverge.

### Engines (`src/verifiers/`)

| engine | kind | sound? | notes |
|--------|------|--------|-------|
| `smt`      | Z3 symbolic        | yes | exact for discrete noise; box over-approx for continuous |
| `milp`     | Gurobi big-M       | yes | needs a Gurobi license; ReLU big-M from interval bounds |
| `lirpa`    | autoLiRPA + BaB    | yes | bound propagation; box method, weaker near ball equilibria |
| `sampling` | random + gradient  | refuter | finds counterexamples; cannot prove `verified` |

For continuous noise the SMT, MILP **and autoLiRPA** engines bin the noise into
cells and bound each cell as a **box** (`x' ∈ x + [lo, hi]`), a sound
over-approximation of `E[V]`; discrete noise collapses the box to an exact point.
The cell count is tunable via `--noise_disc` (per dimension) — higher means a
tighter over-approximation at higher cost. Counterexamples from the sound engines
are re-checked with the canonical jax drift (`src/verifiers/drift.py`).

## Running experiments

Experiments are organised as **launchers** (one per engine / refuter) that each
expand a dict of axes into a SLURM array, plus **runners** (the per-combo
executable that one array task runs). Edit the shared axes once in
`experiments/common/config.py` (envs, seeds, base CEGIS settings) and `slurm.py`
(resource profiles); each launcher only adds its own hyperparameter grid (a list
value is a sweep axis).

```
experiments/
  common/        config (envs/seeds/cegis), slurm profiles, array builder, launch()
  runners/       one process = one array task: cegis, refute_bbob, refute_cert, lirpa_bench
  cegis_verify/  experiment 1: per-engine CEGIS verifier-only timing (smt, milp, mab, mc, ibp, crown, alpha_crown)
  refute_bbob/   experiment 2: per-refuter BBOB benchmark (random, grid, gradient, adalip, direct, whale)
  refute_certs/  experiment 3: per-refuter replay of harvested invalid certs
  cegis_improve/ experiment 4: per-engine refuter-first CEGIS (fairness controls on)
```

Run one combination locally (the printed `test one locally` line is exactly a
combo), a whole launcher locally, or submit the array:

```bash
# one CEGIS combo via the runner
./venv/Scripts/python.exe -m experiments.runners.cegis --env linstoch2D --engine smt --refuter none --seed 0
# one refuter on one BBOB function (+ search-pattern figure)
./venv/Scripts/python.exe -m experiments.runners.refute_bbob --function rastrigin --refuter whale --plots

# a launcher: write the array + print the sbatch line
./venv/Scripts/python.exe -m experiments.cegis_verify.smt
./venv/Scripts/python.exe -m experiments.cegis_verify.smt --local 2     # run the first 2 combos now
./venv/Scripts/python.exe -m experiments.cegis_verify.smt --aggregate   # collate latest tag -> compiled.csv

# soundness / consistency tests
./venv/Scripts/python.exe -m pytest tests -q
```

Each combo writes to `results/<campaign_title>/<tag>/runs/<run_id>/` via
`src/results/run_dir.py`: `manifest.json` (args + status), plus whichever of
`stats.json` (single stats), `iterations.csv` (per-iteration), `figures/` (plots)
or `objects/*.npz` (networks; the CEGIS runner also snapshots every refuted cert
under `objects/invalid/`) the experiment produces. `campaign_title` is preset per
launcher; `tag` defaults to the launch datetime (override with `--tag`). The
array's SLURM logs land in `results/<campaign_title>/<tag>/logs/`.

### Loading runs back

Runs are self-describing, so loading needs no pickle — the env is rebuilt from
its registry name and the certificate from its spec name, then the saved params
are attached:

```python
from src.results.loader import list_runs, load_run

for r in list_runs():
    print(r["run_id"], r["status"], r["verified"], r["env"], r["engine"])

run  = load_run("linear2D_smt_2026-06-14_16-20-16")
env  = run.env()              # reconstructed Env
spec = run.spec()            # CertificateSpec
V, p = run.certificate()     # callable V(x) + raw params (final round; pass round=k for another)
rows = run.metrics()         # metrics.csv as a list of dicts
```

This means an experiment's certificate can be re-verified by a *different* engine,
re-plotted, or continued, just from its `results/<run_id>/` directory.

## Refuters

The sampling engine's counterexample search is a *refuter*: a black-box maximiser
of an objective over a box. `src/refuters/` provides them behind one interface,
decoupled from the drift so they can be benchmarked and reused:

```python
refuter(objective, domain, key, *, budget, batch_size) -> (best_x, best_y, evals_trace, best_trace)
# objective(xs, key) -> ys   (xs: (B,d) -> ys: (B,), MAXIMISED)
```

Available (`src/refuters/base.py::DIRECTORY`): `random`, `grid`, `gradient`
(multi-start projected ascent), `adalip` (AdaLIPO), `direct` (Lipschitz DIRECT),
`whale` (WOA). Each is a single `lax.scan`, so the whole search is one JIT call
and emits a convergence trace.

Benchmark / tune them on BBOB-style functions via experiment 2 (one refuter ×
function × hp per run; `--plots` adds the search-pattern figure), one launcher per
refuter:

```bash
./venv/Scripts/python.exe -m experiments.refute_bbob.whale --local
# -> results/refute_bbob_whale/<tag>/compiled.csv  (+ per-run figures with --plots)
```

The test functions in `experiments/bbob_functions.py` are maximisation problems
(the refuters maximise the objective directly). On 2D Rastrigin / Rosenbrock /
Ackley (budget 8192, 10 runs): `whale` is the best quality/speed trade-off
(near-optimal, ~5 ms); `direct` matches its quality but is ~10× slower; `adalip`
is sample-efficient but ~350 ms (its Lipschitz history dominates); `gradient`
stalls in local optima on multimodal landscapes; `random` and `grid` are fast
baselines.

## Visualisation

Publication figures are produced by the drivers in `visualisation/` (they reuse
the renderers in `src/plotting/` and write vector PDF + PNG under
`results/figures/`):

```bash
# 2D environments: sets, phase portrait + trajectories, candidate-cert heatmaps
./venv/Scripts/python.exe -m visualisation.env_sets --env linstoch2D
./venv/Scripts/python.exe -m visualisation.env_sets --cert results/cegis_verify_mc/<tag>/runs/<id> --cex
# BBOB landscapes + per-optimiser evaluation patterns
./venv/Scripts/python.exe -m visualisation.bbob_landscapes
# how an autoLiRPA verifier explored a 2D env: per-cell status montage + animated GIF
./venv/Scripts/python.exe -m visualisation.verifier_exploration --env linstoch2D
# performance comparison from one or more compiled.csv (speed, verdicts, sample efficiency)
./venv/Scripts/python.exe -m visualisation.performance results
```

## Status

Working: the benchmark suite + certificate structures (cross-backend verified by
the consistency tests), the verification engines (`smt`, `milp`, the autoLiRPA
bound methods `ibp`/`crown`/`alpha-crown`, `montecarlo`, the sound box-UCB `mab`),
the four experiments (`cegis_verify`, `refute_bbob`, `refute_certs`,
`cegis_improve`) with their launchers + runners, the `compile.py` analysis, and the
`visualisation/` drivers. autoLiRPA handles continuous noise (binned into cells).
The CEGIS loop recompiles the refuter once (params are a traced argument), but
sound verification on trained ReLU nets can still be slow / inconclusive — which is
itself part of the thesis story.
