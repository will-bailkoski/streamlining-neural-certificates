"""Certificate store round-trip + query (no training -> fast)."""

import numpy as np
import jax.numpy as jnp
import jax.random as jrn

from src.certificates.structures import glorot_init
from src.results import cert_store
from src.results.cert_store import CertEntry, make_cert_id


def _dummy_params(seed):
    return [(np.asarray(W), np.asarray(b)) for W, b in glorot_init([2, 8, 1], jrn.key(seed))]


def test_cert_store_roundtrip_and_query(tmp_path):
    root = str(tmp_path / "certs")
    params = _dummy_params(0)
    cid = make_cert_id("linstoch2D", "relu_pwl", [8], 0, "valid")
    entry = CertEntry(
        cert_id=cid, env="linstoch2D", structure="relu_pwl", hidden=[8], seed=0,
        epsilon=1e-3, validity="valid", max_drift=-0.01,
        path="linstoch2D/relu_pwl_8_s0_valid.npz", known_cex=None,
    )
    cert_store.save_cert(entry, params, root)

    # also an invalid cert with a known_cex
    inv = CertEntry(
        cert_id=make_cert_id("linstoch2D", "relu_pwl", [8], 0, "invalid"),
        env="linstoch2D", structure="relu_pwl", hidden=[8], seed=0, epsilon=1e-3,
        validity="invalid", max_drift=0.5, path="linstoch2D/relu_pwl_8_s0_invalid.npz",
        known_cex=[1.0, -2.0],
    )
    cert_store.save_cert(inv, _dummy_params(1), root)

    assert cert_store.has_cert(cid, root)
    assert len(cert_store.load_manifest(root)) == 2

    spec, loaded, e2 = cert_store.load_cert(cid, root)
    assert e2.env == "linstoch2D" and e2.validity == "valid"
    for (W0, b0), (W1, b1) in zip(params, loaded):
        assert np.allclose(W0, W1) and np.allclose(b0, b1)
    # spec.forward must run on the reloaded params
    assert np.isfinite(float(spec.forward(loaded, jnp.array([0.3, -0.4]))))

    assert [e.cert_id for e in cert_store.query(root, validity="valid")] == [cid]
    assert cert_store.query(root, validity="invalid")[0].known_cex == [1.0, -2.0]
    assert cert_store.query(root, env="nope") == []
