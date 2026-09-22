# Streamlining Neural Certificate Synthesis

Code, thesis and presentation for the MSc thesis **"Streamlining Neural Certificate
Synthesis"** (William Bailkoski, Universitat Pompeu Fabra, 2026).

- Thesis: [`doc/MasterThesis.pdf`](doc/MasterThesis.pdf) (LaTeX source in `doc/`)
- Presentation: [`doc/presentation/streamlining_talk.pdf`](doc/presentation/streamlining_talk.pdf)

The thesis studies counterexample-guided inductive synthesis (CEGIS) of neural
supermartingale certificates for discrete-time stochastic systems. It compares sound
verifiers from three families (symbolic, bound propagation, sampling), benchmarks
black-box optimisers as cheap *refuters*, and shows that running a refuter before the
sound verifier in each CEGIS round removes most sound-verifier calls, speeding up
synthesis by as much as 15.7x.

## Design

One definition, many engines. A single `Env` defines the dynamics, noise, domain
and equilibrium; a single `CertificateSpec` defines the parametric form of `V`; every
engine checks the same drift condition on the same system:

```
verified  <=>  E_w[ V(f(x, w)) ] - V(x) + eps <= 0   for all x in domain \ equilibrium
```

A counterexample is a state where the left-hand side is positive. The jax forward
pass, the Z3 / Gurobi encodings and the torch module of a certificate are all derived
from one spec, so training and verification never diverge (checked by `tests/`).

## Repository layout

```
src/
  benchmarks/     one Env per system (dynamics, noise, domain, equilibrium)
  certificates/   certificate structures + Z3 / Gurobi / torch encoders
  verifiers/      sound engines behind one interface + the shared drift functional
  refuters/       black-box maximisers behind one interface
  training/       CEGIS learner
  results/        run directories and certificate parameter I/O
  plotting/       shared figure helpers
experiments/
  common/         shared axes (config.py), SLURM profiles, array launcher
  runners/        one process = one task: cegis, refute_bbob, refute_cert
  cegis_verify/   verifier-only CEGIS, one launcher per engine
  cegis_improve/  refuter-first CEGIS (random / whale pre-screen), LiRPA engines
  refute_bbob/    refuter benchmark on BBOB functions, one launcher per refuter
  refute_certs/   refuters replayed on verifier-rejected ("faux") certificates
  *.py            standalone studies (Lipschitz budget, MILP noise bins, anytime bounds)
visualisation/    one script per thesis figure (--doc writes into doc/Figures/)
results/analysis/ compile a campaign into compiled.csv + summary
tests/            cross-backend consistency and soundness tests
doc/              thesis source + PDF, presentation source + PDF
```

## Installation

Python 3.11 or newer.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

- **Gurobi.** The MILP engine uses `gurobipy`. Its bundled licence only handles
  small models; the MILP experiments need a full (e.g. free academic) licence.
- **GPU (optional).** The LiRPA and MAB engines use a GPU when one is available.
  For JAX on GPU install `jax[cuda12]`; torch picks up CUDA automatically.

## Benchmarks

| Thesis name | Registry key(s) | Dynamics | Noise |
|---|---|---|---|
| LinSwitch  | `linear2D` (`linear3D`, `linear4D`) | switched linear | multiplicative (random mode switch) |
| LinStoch   | `linstoch2D` (`linstoch3D`, `linstoch4D`) | linear | additive bounded |
| DoubleWell | `doublewell` | polynomial gradient flow | additive bounded |
| Pendulum   | `pendulum_lqr` | inverted pendulum, LQR feedback | additive triangular |

Keys index `src/benchmarks/utils.py::DIRECTORY`; build any system with
`make_env("<key>")`.

## Verification engines

| `--engine` | Method | Family |
|---|---|---|
| `smt`         | Z3, exact encoding of the ReLU certificate | symbolic |
| `milp`        | Gurobi big-M MILP | symbolic |
| `ibp`, `crown`, `alpha-crown` | autoLiRPA bound propagation + branch and bound | bound propagation |
| `mc`          | Monte Carlo with a Lipschitz mean-to-max bound | sampling |
| `mab`         | adaptive multi-armed-bandit refinement | sampling |

Continuous noise is binned into `--noise_disc` cells per dimension, each bounded as a
box: a sound over-approximation of the expectation that tightens as the bin count
grows.

## Refuters

`random`, `grid`, `gradient` (hill climbing), `direct` (DIRECT), `adalip` (AdaLIPO) and
`whale` (whale optimisation), all in `src/refuters/`. Each is a single jit-compiled
`lax.scan` behind one interface:

```python
refuter(objective, domain, key, *, budget, batch_size) -> (best_x, best_y, evals_trace, best_trace)
```

## Running experiments

Each launcher expands its axes into one command per combination, writes them to
`results/<campaign>/<tag>/` together with a SLURM array script, and prints the
`sbatch` line. The same launcher can run locally or collate finished runs:

```bash
python -m experiments.cegis_verify.crown                 # write the array + print sbatch
python -m experiments.cegis_verify.crown --local 2       # run the first 2 combos locally
python -m experiments.cegis_verify.crown --aggregate     # collate -> compiled.csv
```

A single combination can also be run directly:

```bash
python -m experiments.runners.cegis --env linear2D --engine crown --refuter whale --seed 0
python -m experiments.runners.refute_bbob --function rastrigin --refuter whale --seed 0
```

Shared axes (systems, seeds, CEGIS settings) live in `experiments/common/config.py`,
cluster settings in `experiments/common/slurm.py` (edit the partition and venv path
for your cluster). Each run writes its manifest, statistics and any saved
certificates under `results/<campaign>/<tag>/runs/<run_id>/`. The `visualisation/`
scripts read the latest tag of each campaign; pass `--doc` to write the figure into
`doc/Figures/` under the name the thesis uses.

## Reproducing the thesis

All commands are run from the repository root as `python -m <module>`.

| Chapter | Experiment | Figures |
|---|---|---|
| 3 Lipschitz bounds | `experiments.lipschitz_budget --plot` | `visualisation.lipschitz_budget --separate --doc`, `visualisation.lipschitz_gridding --doc` |
| 4 Benchmarks, App. A | none | `visualisation.env_sets --env <key> --doc` for `linear2D`, `linstoch2D`, `doublewell`, `pendulum_lqr` |
| 5 Verification: MILP noise bins | `experiments.milp_noise_disc` | `visualisation.milp_noise_disc --doc` |
| 5 Verification: bound propagation | `experiments.pendulum_certs`, then `experiments.pendulum_anytime --plot` | `visualisation.pendulum_anytime --doc`, `visualisation.verifier_grid --method crown --env pendulum_lqr --doc` |
| 5 Verification: MAB | `experiments.cegis_verify.mab` | `visualisation.mab_dashboard --doc` |
| 5 Capability boundary | `experiments.cegis_verify.{smt,milp,mc,mab,ibp,crown,alpha_crown}` | none |
| 6 Refutation: BBOB | `experiments.bbob_profile`, `experiments.refute_bbob.<refuter>` for all six | `visualisation.bbob_heatmaps --doc`, `visualisation.bbob_quality_per_function --doc`, `visualisation.bbob_quality_vs_time --doc`, `visualisation.refuter_patterns --doc` (App. B) |
| 6 Refutation: faux certificates | `experiments.refute_certs.<refuter> --source results/cegis_verify_<engine>/<tag>` | `visualisation.faux_certs --doc` |
| 7 Streamlining CEGIS | `experiments.cegis_verify.{ibp,crown,alpha_crown}` + `experiments.cegis_improve.{ibp,crown,alpha_crown}` | `results.analysis.compile`, `visualisation.cegis_pipelines` |

The Chapter 7 campaign is three LiRPA engines x three systems (LinSwitch, DoubleWell,
Pendulum) x three strategies (verifier-only, random pre-screen, whale pre-screen) x
five seeds, with an 8192-evaluation refuter budget and 16 Monte-Carlo samples per
point. MILP runs need a Gurobi licence; in the thesis they ran locally (`--local`)
rather than on the cluster.

## Building the documents

```bash
cd doc && pdflatex MasterThesis && bibtex MasterThesis && pdflatex MasterThesis && pdflatex MasterThesis
cd doc/presentation && pdflatex streamlining_talk && pdflatex streamlining_talk
```

The two illustrative talk figures are regenerated with
`python doc/presentation/boat_demo.py` and `python doc/presentation/mab_demo.py`.
