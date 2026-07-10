"""
Experiment 3 — replay harvested invalid certs with random search.

Scans a cegis_verify campaign (--source) for every invalid-cert snapshot and fans
one array task per (snapshot x seed), timing this refuter against the verifier
that originally refuted each cert.

    python -m experiments.refute_certs.random --source results/cegis_verify_smt/<tag>
    python -m experiments.refute_certs.random --source results/cegis_verify_smt/<tag> --local 3
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.array import find_invalid_snapshots
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "refute_certs_random"

REFUTER_HP: dict = {}


def _build(ns):
    return dict(snapshot=find_invalid_snapshots(ns.source), seed=config.SEEDS,
                refuter="random", budget=8192, batch_size=256, **REFUTER_HP)


if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="refute_cert",
        slurm=SLURM_GPU,
        build_args=_build,
        extra_cli=[("--source", dict(required=True,
                    help="cegis_verify campaign tag dir (or parent) to scan for invalid certs"))],
    )
