"""Experiment-framework plumbing: arg expansion + the Run results directory."""

import json

import numpy as np
import jax.random as jrn

from experiments.common.grid import expand_args
from src.certificates.structures import glorot_init
from src.results.run_dir import Run, aggregate, make_run_id


def test_expand_args_cartesian():
    combos = expand_args({"env": ["a", "b"], "engine": "smt", "seed": [0, 1]})
    assert len(combos) == 4
    assert {"env": "a", "engine": "smt", "seed": 0} in combos
    assert all(c["engine"] == "smt" for c in combos)
    assert expand_args({}) == [{}]  # no axes -> a single empty combo


def test_run_id_drops_meta_and_none():
    a = make_run_id({"env": "x", "seed": 0, "lr": None, "campaign": "c", "cert_root": "certs"})
    b = make_run_id({"env": "x", "seed": 0})  # campaign/cert_root/None must not affect identity
    assert a == b


def test_run_dir_all_result_shapes(tmp_path):
    args = {"env": "linstoch2D", "seed": 0, "campaign": "t", "results_dir": str(tmp_path)}
    run = Run("demo", args, campaign="t", results_dir=str(tmp_path))
    assert not run.done

    run.stats({"verdict": "verified", "time": 1.5})
    run.log_iter({"round": 0, "loss": 0.3})
    run.log_iter({"round": 1, "loss": 0.1})
    params = [(np.asarray(W), np.asarray(b)) for W, b in glorot_init([2, 4, 1], jrn.key(0))]
    run.object("certificate", params, meta={"lipschitz": 2.0})
    run.finish()

    man = json.loads((run.dir / "manifest.json").read_text())
    assert man["status"] == "completed"
    assert "certificate" in man["outputs"]["objects"]
    assert (run.dir / "iterations.csv").exists()
    # objects round-trip
    loaded = run.load_object("certificate")
    assert np.allclose(loaded[0][0], params[0][0])

    # resume: re-opening a completed run is skipped
    assert Run("demo", args, campaign="t", results_dir=str(tmp_path)).done

    # aggregate joins args + stats into one CSV row
    out = aggregate("t", str(tmp_path))
    import csv
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["verdict"] == "verified" and rows[0]["env"] == "linstoch2D"
