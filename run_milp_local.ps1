# The MILP arm of the bigtest campaigns, run locally (the Gurobi WLS license
# lives on this laptop; the cluster token server is login-node-only — see
# experiments/common/slurm.py). Every run is resumable: finished run_ids are
# skipped on relaunch, so this script can be re-run after interruptions and it
# picks up where it left off.
#
#     .\run_milp_local.ps1              # both campaigns + aggregation
#     .\run_milp_local.ps1 -Tag pilot   # a different results tag

param([string]$Tag = "bigtest")

$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
Set-Location $PSScriptRoot

Write-Host "== cegis_verify_milp (verifier-only) --tag $Tag" -ForegroundColor Cyan
& $py -m experiments.cegis_verify.milp --local --tag $Tag

Write-Host "== cegis_improve_milp (whale pre-screen) --tag $Tag" -ForegroundColor Cyan
& $py -m experiments.cegis_improve.milp --local --tag $Tag

# cross-check milp column: only meaningful once the OTHER campaigns' run dirs
# (with their certificates) are synced back from the cluster into results/.
Write-Host "== cross_check (milp column) --tag $Tag" -ForegroundColor Cyan
& $py -m experiments.cross_check --engines milp --local --tag $Tag

Write-Host "== aggregate" -ForegroundColor Cyan
& $py -m experiments.cegis_verify.milp --aggregate --tag $Tag
& $py -m experiments.cegis_improve.milp --aggregate --tag $Tag
& $py -m experiments.cross_check --aggregate --tag $Tag
