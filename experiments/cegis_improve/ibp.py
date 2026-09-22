"""
Experiment 4 - refuter-first CEGIS, autoLiRPA IBP engine (GPU).

The random and whale pre-screen arms of the Chapter 7 campaign: identical to
experiments.cegis_verify.ibp (same engine HP, systems and seeds) except that a
refuter attacks each candidate before the sound verifier is called.

    python -m experiments.cegis_improve.ibp --tag final
    python -m experiments.cegis_improve.ibp --local 2
"""

from __future__ import annotations
from experiments.cegis_improve._common import groups
from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cegis_improve_ibp"

if __name__ == "__main__":
    launch(campaign_title=CAMPAIGN_TITLE, runner="cegis", slurm=SLURM_GPU_LONG,
           args=groups("ibp"))
