"""
Experiment 3 — replay harvested invalid certs with DIRECT (dividing rectangles).

    python -m experiments.refute_certs.direct --source results/cegis_verify_smt/<tag>
"""

from __future__ import annotations
from experiments.common import config
from experiments.common.array import find_invalid_snapshots
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU

CAMPAIGN_TITLE = "refute_certs_direct"

# BBOB tuning winner (Table tab:bbob-best).
REFUTER_HP = dict(po_eps=1e-3)


def _build(ns):
    return dict(snapshot=find_invalid_snapshots(ns.source), seed=config.SEEDS,
                refuter="direct", budget=8192, batch_size=64, **REFUTER_HP)


if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="refute_cert",
        slurm=SLURM_GPU,
        build_args=_build,
        extra_cli=[("--source", dict(required=True,
                    help="cegis_verify campaign tag dir (or parent) to scan for invalid certs"))],
    )
