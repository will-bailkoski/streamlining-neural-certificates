"""
The one entrypoint every experiment launcher calls.

A launcher imports the shared config, sets a PRESET `campaign_title`, names its
runner, and supplies its axes; `launch` expands the product, writes the SLURM
array, and prints the `sbatch` command to copy-paste. Each combo line is also
exactly the local command, so nothing is hidden in param files.

Axes are given as either:
  * `args`       a dict of scalars/lists (list = sweep axis), or a list of such
                 dicts whose expansions are concatenated (e.g. one group per
                 refuter, each with its own hyperparameters); or
  * `build_args` a callable(parsed_cli) -> (dict | list[dict]), for axes derived
                 at launch time (e.g. scanning a results dir for snapshots).
`extra_cli` adds launcher-specific flags (e.g. --source) to the CLI.

    python -m experiments.cegis_verify.ibp                 # write the array + print sbatch
    python -m experiments.cegis_verify.ibp --local 2       # run the first 2 combos now
    python -m experiments.cegis_verify.ibp --aggregate     # collate latest tag -> compiled.csv
    python -m experiments.cegis_verify.ibp --tag pilot     # use results/<campaign>/pilot/
"""

from __future__ import annotations
import argparse
from pathlib import Path

from experiments.common import array
from experiments.common.grid import expand_args
from experiments.common.slurm import SLURM_CPU


def launch(*, campaign_title: str, runner: str, args=None, build_args=None,
           extra_cli: list | None = None, slurm: dict | None = None,
           iters_aggregate: bool = False) -> None:
    slurm = slurm or SLURM_CPU
    p = argparse.ArgumentParser(description=f"Launch the {campaign_title} SLURM array")
    p.add_argument("--tag", default=None,
                   help="results/<campaign>/<tag>/  (default: launch datetime)")
    p.add_argument("--local", nargs="?", const=-1, type=int, metavar="N",
                   help="run combos locally in series (optionally just the first N)")
    p.add_argument("--aggregate", action="store_true",
                   help="collate finished runs -> compiled.csv and exit")
    for name, kw in (extra_cli or []):
        p.add_argument(name, **kw)
    a = p.parse_args()

    campaign_root = Path("results") / campaign_title

    if a.aggregate:
        campaign_dir = campaign_root / a.tag if a.tag else _latest_tag(campaign_root)
        if campaign_dir is None:
            print(f"no runs under {campaign_root.as_posix()}")
            return
        array.compile_campaign(campaign_dir, iters=iters_aggregate)
        return

    resolved = args if args is not None else build_args(a)
    groups = resolved if isinstance(resolved, list) else [resolved]
    combos = [c for g in groups for c in expand_args(g)]
    if not combos:
        print(f"[{campaign_title}] no combos to run (empty axes)")
        return

    tag = a.tag or array.now_tag()
    campaign_dir = campaign_root / tag
    (campaign_dir / "logs").mkdir(parents=True, exist_ok=True)

    common = ["--results_dir", campaign_root.as_posix(), "--campaign", tag]
    lines = [" ".join(array.to_argv(c) + common) for c in combos]
    array.write_combos(campaign_dir, lines)
    sbatch = array.write_sbatch(campaign_dir, len(lines), slurm, runner)

    if a.local is not None:
        array.run_local(lines, runner, a.local)
        array.compile_campaign(campaign_dir, iters=iters_aggregate)
        return

    print(f"[{campaign_title}]  {len(lines)} combos  (tag={tag})")
    print(f"  runner    : experiments.runners.{runner}")
    print(f"  combos    : {(campaign_dir / 'combos.txt').as_posix()}")
    print(f"  results   : {campaign_dir.as_posix()}/runs/   (+ logs/)")
    print()
    print(f"  submit:            sbatch {sbatch.as_posix()}")
    print(f"  test one locally:  python -m experiments.runners.{runner} {lines[0]}")
    print(f"  collate results:   <launcher> --aggregate --tag {tag}")


def _latest_tag(campaign_root: Path) -> Path | None:
    tags = sorted(d for d in campaign_root.glob("*") if d.is_dir())
    return tags[-1] if tags else None
