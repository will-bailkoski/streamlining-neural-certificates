# The bigtest campaign — runbook

One coherent tag (`bigtest`) across every campaign, so all the gallery figures
resolve to the same data without `--tag` juggling. Two arms per engine:
`cegis_verify_*` (verifier-only) and `cegis_improve_*` (identical config + the
whale pre-screen, BBOB-winner HP), plus the `cross_check` matrix where every
engine re-verifies the others' certificates.

## What changed vs the last cluster round (why the reruns are worth it)

* **MAB "inconclusive (no-progress)" is fixed at the source**: the engine's only
  inconclusive path is a budget `RuntimeError`; it now records
  `hit = max_boxes | max_samples | timeout` so the runner reports the real
  reason (it previously fell through to the generic "no-progress" label — the
  message was always in `v_error`, just never surfaced).
* **MAB and the lirpa engines have a wall-clock `timeout`** that fires INSIDE
  the SLURM wall. Before, a wall kill left `status=running` with no stats —
  those rows are silently dropped by the figures.
* **Refuter counterexamples are recheck-gated** (`--refuter_recheck_n 4096`,
  default on) and the refuter objective uses `mc_samples=128` (was 16): the
  whale arm no longer trains on noise peaks.
* The previously submitted `cegis_verify_mab/final` array was generated from an
  older commit (grid_per_dim=1, 4,4 nets, 5 envs) — the current launcher config
  (grid 32, 10 envs, per-env widths) was never actually run.
* `cegis_improve_mab` previously inherited the UNSOUND relu_pwl cert and
  SLURM_CPU; it now mirrors `cegis_verify_mab` (bounded_pwl, per-env groups,
  SLURM_MAB) with the whale arm as the only difference.
* `noise_disc` is now a per-env-family axis (`config.noise_groups`): the
  discrete-noise linear* envs no longer waste sweep combos on a flag they
  ignore, and each dimension only sweeps the disc it can afford (disc^dim
  cells). NOTE the measured MILP window on linstoch2D is nd={4,5,6} with nd=7
  timing out at 600s (results/milp_noise_disc) — nd=8 is kept as the sweep's
  upper edge, not the default.

## 0. Pre-flight

```bash
git push                      # cluster pulls this branch (overhaul-experiments)
# on the cluster:
git pull && source venv/bin/activate
```

## 1. Cluster arrays (run on the cluster, each prints its sbatch)

```bash
# verifier-only arm                        # combos  slurm
python -m experiments.cegis_verify.mab          --tag bigtest   # 30   MAB 64G/24h GPU
python -m experiments.cegis_verify.ibp          --tag bigtest   # 48   GPU 12h
python -m experiments.cegis_verify.crown        --tag bigtest   # 48   GPU 12h
python -m experiments.cegis_verify.alpha_crown  --tag bigtest   # 48   GPU 12h
python -m experiments.cegis_verify.smt          --tag bigtest   # 33   CPU 24h
python -m experiments.cegis_verify.mc           --tag bigtest   # 24   CPU 8h

# whale arm (same config + whale pre-screen)
python -m experiments.cegis_improve.mab          --tag bigtest  # 30
python -m experiments.cegis_improve.ibp          --tag bigtest  # 48
python -m experiments.cegis_improve.crown        --tag bigtest  # 48
python -m experiments.cegis_improve.alpha_crown  --tag bigtest  # 48
python -m experiments.cegis_improve.smt          --tag bigtest  # 33
python -m experiments.cegis_improve.mc           --tag bigtest  # 24

# then sbatch each printed submit.sbatch
```

## 2. Local laptop (Gurobi WLS license) — MILP arm

```powershell
.\run_milp_local.ps1          # verify (45) + improve (45) + cross-check milp column
```

Resumable: finished run_ids are skipped, re-run it any time.

## 3. Cross-check matrix (on the cluster, AFTER the campaigns finish)

```bash
python -m experiments.cross_check --tag bigtest   # scans the campaigns' verified
sbatch results/cross_check/bigtest/submit.sbatch  # certs, one cell per (cert, engine)
```

The milp column of the same matrix runs locally (step 2 / run_milp_local.ps1),
against the run dirs synced back in step 4.

## 4. Sync results back (whole run dirs — the cert-based figures need objects/)

```bash
rsync -av cluster:masters_thesis/results/cegis_verify_* results/
rsync -av cluster:masters_thesis/results/cegis_improve_* results/
rsync -av cluster:masters_thesis/results/cross_check results/
```

Then collate: each launcher's `--aggregate --tag bigtest`, and
`python -m experiments.consolidate` for the master all_results.csv.

## 5. Figures (all auto-resolve to the newest tag; pass --tag bigtest to pin)

| figure | reads |
|---|---|
| `visualisation.cegis_pipelines` | all cegis_verify_* + cegis_improve_* compiled.csv |
| `visualisation.verifier_grid` | certs from cegis_verify_{crown,ibp,mab} run dirs |
| `visualisation.mab_dashboard` | a cegis_verify_mab run (record replay, cached) |
| `visualisation.drift_heatmaps` | any 2D cert run dir |
| `visualisation.cross_check` | results/cross_check/bigtest/compiled.csv |
| `visualisation.performance results/` | everything with a compiled.csv |

Diagnosis columns in compiled.csv: `verdict`, `v_hit` (which budget stopped an
inconclusive run — now populated for mab too), `v_error` (the raw message),
`v_resid_bound`/`v_obj_bound` (how close to verified), `n_refuter_spurious`
(how many whale CEs the recheck gate rejected).

## Expectations (so the negatives read as findings, not failures)

* smt: expected to stay undecidable everywhere (Z3 timed out even on
  discrete-noise 2D at 120s; the 600s crank makes that an honest negative).
* milp at 3D/4D: the nd window was measured EMPTY at 3D — the runs document the
  dimension wall.
* lirpa at 4D + high nd: disc^dim cells per bound call; the 4D groups cap at
  nd=2 for exactly that reason.
* mab 3D/4D: box count ~ (domain*L/M)^d — the 4D combos exhausting their budget
  IS the dimension-scaling datapoint.
