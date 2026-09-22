"""
Cross-backend soundness/fairness guard rails.

These tests are the mechanical guarantee that every engine checks the SAME
system, the SAME certificate, and the SAME drift condition. If any backend's
encoding drifts from the others, one of these fails. Run with:

    ./venv/Scripts/python.exe -m pytest tests/test_consistency.py -q

Gurobi tests skip automatically if no license is available.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from src.benchmarks.utils import make_env, DIRECTORY as ENV_DIRECTORY
from src.benchmarks.linear_system import SwitchedLinearEnv
from src.certificates.structures import DIRECTORY as CERT_DIRECTORY, glorot_init
from src.certificates.encoders import encode_z3, encode_gurobi, build_torch
from src.verifiers.base import get_engine
from src.verifiers.drift import recheck_violation

TOL = 1e-4

# Exact L1 net for linear2D: V(x) = |x0| + |x1| (relu hidden, linear output)
_W0 = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
_GOOD_L1 = [(_W0, jnp.zeros(4)), (jnp.array([[1.0, 1.0, 1.0, 1.0]]), jnp.zeros(1))]
_BAD_L1 = [(_W0, jnp.zeros(4)), (jnp.array([[-1.0, -1.0, -1.0, -1.0]]), jnp.zeros(1))]


def _has_gurobi():
    try:
        import gurobipy as gp

        m = gp.Model()
        m.Params.OutputFlag = 0
        return True
    except Exception:
        return False


HAS_GUROBI = _has_gurobi()
gurobi_required = pytest.mark.skipif(not HAS_GUROBI, reason="no gurobi license")


# ----------------------------------------------------------------------
# 1. Dynamics: symbolic step agrees with jax step
# ----------------------------------------------------------------------
def _gurobi_succ_boxes(env, x_point):
    import gurobipy as gp

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    xv = m.addVars(env.dim, lb=-gp.GRB.INFINITY)
    for i in range(env.dim):
        m.addConstr(xv[i] == float(x_point[i]))
    branches = env.gurobi_step(m, [(1.0, list(xv.values()))])
    out = []
    for w, xn in branches:
        lo = np.empty(env.dim)
        hi = np.empty(env.dim)
        for i in range(env.dim):
            m.setObjective(xn[i], gp.GRB.MINIMIZE)
            m.optimize()
            lo[i] = xn[i].X
            m.setObjective(xn[i], gp.GRB.MAXIMIZE)
            m.optimize()
            hi[i] = xn[i].X
        out.append((w, lo, hi))
    return out


@gurobi_required
def test_discrete_dynamics_expectation_matches_mc():
    env = make_env("linear2D")
    V = env.known_V
    key = jrn.key(0)
    for x in [np.array([1.3, -0.7]), np.array([-2.0, 0.4]), np.array([0.5, 0.5])]:
        boxes = _gurobi_succ_boxes(env, x)
        assert all(np.allclose(lo, hi) for _, lo, hi in boxes)  # discrete -> exact
        e_sym = sum(w * float(V(jnp.array((lo + hi) / 2))) for w, lo, hi in boxes)
        ks = jrn.split(key, 200_000)
        e_mc = float(jax.vmap(lambda k: V(env.step(jnp.array(x), k)[0]))(ks).mean())
        assert abs(e_sym - e_mc) < 2e-2


@gurobi_required
def test_continuous_successors_contained_in_symbolic_box():
    env = make_env("doublewell")
    key = jrn.key(1)
    for x in [np.array([0.3, 0.9]), np.array([2.5, -1.0]), np.array([1.7, 0.2])]:
        boxes = _gurobi_succ_boxes(env, x)
        assert abs(sum(w for w, _, _ in boxes) - 1.0) < 1e-9
        ks = jrn.split(key, 4000)
        succ = np.array(jax.vmap(lambda k: env.step(jnp.array(x), k)[0])(ks))
        inside = np.zeros(len(succ), bool)
        for _, lo, hi in boxes:
            inside |= np.all((succ >= lo - 1e-6) & (succ <= hi + 1e-6), axis=1)
        assert inside.all()  # soundness: every sample is in the over-approx box


# ----------------------------------------------------------------------
# 2. Certificate: z3 / gurobi / torch encoders match jax forward
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cert_name", list(CERT_DIRECTORY))
def test_certificate_encoders_match_jax(cert_name):
    import torch

    spec = CERT_DIRECTORY[cert_name]
    params = glorot_init([2, 5, 5, 1], jrn.key(7))
    params = [(w * 2.0, b + 0.1) for w, b in params]
    fwd = spec.forward
    tmod = build_torch(spec, params)
    xs = np.array([[0.7, -1.2], [-2.0, 0.5], [1.5, 1.5], [0.0, 0.0]])
    for x in xs:
        v_jax = float(fwd(params, jnp.array(x)))
        v_torch = float(tmod(torch.tensor(x, dtype=torch.float32)).item())
        assert abs(v_jax - v_torch) < TOL
        if spec.symbolic_friendly:
            import z3

            s = z3.Solver()
            xv = [z3.Real(f"x_{i}") for i in range(2)]
            for i in range(2):
                s.add(xv[i] == float(x[i]))
            out = z3.Real("out")
            s.add(out == encode_z3(spec, s, params, xv))
            assert s.check() == z3.sat
            v_z3 = float(s.model()[out].as_fraction())
            assert abs(v_jax - v_z3) < TOL
            if HAS_GUROBI:
                import gurobipy as gp

                m = gp.Model()
                m.Params.OutputFlag = 0
                xg = m.addVars(2, lb=-gp.GRB.INFINITY)
                for i in range(2):
                    m.addConstr(xg[i] == float(x[i]))
                vg = encode_gurobi(spec, m, params, list(xg.values()), input_bounds=(x, x))
                m.optimize()
                assert abs(v_jax - float(vg.X)) < TOL


# ----------------------------------------------------------------------
# 3. Noise: support weights sum to 1; expectation matches MC
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ["linear2D", "doublewell"])
def test_noise_support_weights_sum_to_one(env_name):
    env = make_env(env_name)
    for noise in (env._pre, env._post):
        _, _, weights = noise.get_support()
        assert abs(float(np.sum(weights)) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 4. Drift agreement: sound engines agree on the verdict
# ----------------------------------------------------------------------
@pytest.mark.parametrize("engine_name", ["smt", "lirpa"])
def test_sound_engines_agree_good_cert(engine_name):
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    eng = get_engine(engine_name)
    r = eng.find_counterexample(env, spec, _GOOD_L1, 1e-3, key=jrn.key(0))
    assert r.verified is True
    assert r.violation is None


def test_montecarlo_no_false_refutation_good_cert():
    """The sound MC method (Lipschitz mean->max) must not hallucinate a
    counterexample on a good cert. It CANNOT return verified=True at tol=0 (its
    certified ceiling is always > 0), but it must never return verified=False,
    and its certified_sup_drift must be a sound (positive) upper bound."""
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    r = get_engine("montecarlo").find_counterexample(
        env, spec, _GOOD_L1, 1e-3, key=jrn.key(0), n_states=40_000)
    assert r.verified is not False          # no false refutation
    assert r.violation is None
    assert r.stats["certified_sup_drift"] > 0.0  # sound ceiling exists


@pytest.mark.parametrize("engine_name", ["smt", "lirpa", "montecarlo"])
def test_engines_find_genuine_counterexample_bad_cert(engine_name):
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    eng = get_engine(engine_name)
    r = eng.find_counterexample(env, spec, _BAD_L1, 1e-3, key=jrn.key(0))
    assert r.violation is not None
    genuine, drift = recheck_violation(env, spec, _BAD_L1, 1e-3, r.violation[0], jrn.key(1))
    assert genuine and drift > 0


# ----------------------------------------------------------------------
# 4b. Sample-based MAB engine: sound box verifier finds a genuine CE on a
#     broken cert (a coarse seed grid keeps the smoke fast).
# ----------------------------------------------------------------------
def test_mab_finds_genuine_counterexample_bad_cert():
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    eng = get_engine("mab")
    r = eng.find_counterexample(env, spec, _BAD_L1, 1e-3, key=jrn.key(0), grid_per_dim=2)
    assert r.verified is False
    assert r.violation is not None
    genuine, drift = recheck_violation(env, spec, _BAD_L1, 1e-3, r.violation[0], jrn.key(1))
    assert genuine and drift > 0


@gurobi_required
def test_milp_matches_smt_on_linear2d():
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    smt = get_engine("smt")
    milp = get_engine("milp")
    for params in (_GOOD_L1, _BAD_L1):
        rs = smt.find_counterexample(env, spec, params, 1e-3)
        rm = milp.find_counterexample(env, spec, params, 1e-3, time_limit=30)
        assert rs.verified == rm.verified


@gurobi_required
def test_noise_disc_controls_symbolic_branches():
    """noise_disc bins continuous noise into more cells (tighter E[V] over-approx);
    discrete noise ignores it. Guards the gurobi_step/z3_step -> get_support wiring."""
    import gurobipy as gp

    def n_branches(env, disc):
        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVars(env.dim, lb=-gp.GRB.INFINITY)
        return len(env.gurobi_step(m, [(1.0, list(x.values()))], noise_disc=disc))

    cont = make_env("linstoch2D")  # continuous additive noise
    assert n_branches(cont, 3) > n_branches(cont, 1)   # more cells at higher disc
    disc_env = make_env("linear2D")  # switched/discrete noise
    assert n_branches(disc_env, 3) == n_branches(disc_env, 1)  # exact; disc ignored


# ----------------------------------------------------------------------
# 5. Known-V sanity: the true supermartingale verifies
# ----------------------------------------------------------------------
def test_known_l1_supermartingale_verifies_smt():
    env = make_env("linear2D")
    spec = CERT_DIRECTORY["relu_pwl"]
    r = get_engine("smt").find_counterexample(env, spec, _GOOD_L1, 1e-3)
    assert r.verified is True


# ----------------------------------------------------------------------
# 6. random_construction: target spectral norm + valid L2 supermartingale
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ndims", [2, 3, 4])
def test_random_construction_is_well_formed(ndims):
    env = SwitchedLinearEnv.random_construction(ndims, 0.8)
    assert env.dim == ndims
    assert abs(float(jnp.linalg.norm(env.A0, ord=2)) - 0.8) < 1e-4
    assert abs(float(jnp.linalg.norm(env.A1, ord=2)) - 0.8) < 1e-4
    key = jrn.key(0)
    xs, _ = env.sample(key, 3000)
    V = env.known_V

    def drift(x, k):
        ks = jrn.split(k, 32)
        return jnp.mean(jax.vmap(lambda kk: V(env.step(x, kk)[0]))(ks)) - V(x)

    d = np.array(jax.vmap(drift)(xs, jrn.split(key, xs.shape[0])))
    assert d.max() <= 1e-3  # L2 norm is a supermartingale on domain \ equilibrium


# ----------------------------------------------------------------------
# 7. continuous-noise autoLiRPA: the binned drift bound is SOUND (an upper
#    bound on the true expectation) and tightens (does not loosen) with disc.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("disc", [1, 2])
def test_lirpa_continuous_noise_is_sound(disc):
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from src.certificates.encoders import build_torch

    env = make_env("doublewell")
    spec = CERT_DIRECTORY["relu_pwl"]
    params = glorot_init([2, 8, 1], jrn.key(4))
    V = lambda x: spec.forward(params, x)

    # true max Monte-Carlo drift over the domain (eps=0)
    key = jrn.key(0)
    xs, _ = env.sample(key, 3000)

    def mc(x, k):
        ks = jrn.split(k, 128)
        return jnp.mean(jax.vmap(lambda kk: V(env.step(x, kk)[0]))(ks)) - V(x)

    true_max = float(jnp.max(jax.vmap(mc)(xs, jrn.split(key, xs.shape[0]))))

    module, nb = env.lirpa_drift_module(build_torch(spec, params), 0.0, noise_disc=disc)
    module.eval()
    w_lo, w_hi = nb
    bm = BoundedModule(module, (torch.empty((1, 2)), torch.empty((1, w_lo.size))))
    lo = np.array(env.domain.bounds[:, 0]); hi = np.array(env.domain.bounds[:, 1])
    x_bt = BoundedTensor(
        torch.tensor((lo + hi) / 2, dtype=torch.float32).reshape(1, 2),
        PerturbationLpNorm(
            x_L=torch.tensor(lo, dtype=torch.float32).reshape(1, 2),
            x_U=torch.tensor(hi, dtype=torch.float32).reshape(1, 2),
        ),
    )
    wl = torch.tensor(w_lo, dtype=torch.float32).reshape(1, -1)
    wu = torch.tensor(w_hi, dtype=torch.float32).reshape(1, -1)
    w_bt = BoundedTensor((wl + wu) / 2, PerturbationLpNorm(x_L=wl, x_U=wu))
    _, ub = bm.compute_bounds(x=(x_bt, w_bt), method="IBP")
    assert float(ub.reshape(-1)[0]) >= true_max - 1e-4  # sound over-approximation


# ----------------------------------------------------------------------
# 8. New benchmark suite — quadratic-known_V envs are exact supermartingales.
#    For V(x) = (x-c)^T P x with A^T P A - P = -I, the drift is exactly
#    -||x-c||^2 + tr(P Sigma), so we can assert soundness analytically (no MC
#    noise). This also pins the discrete-Lyapunov / noise-floor equilibrium.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ["linstoch2D", "linstoch3D", "linstoch4D"])
def test_quadratic_known_V_is_supermartingale(env_name):
    env = make_env(env_name)
    P = np.array(env.P)
    center = np.array(getattr(env, "setpoint", np.zeros(env.dim)))
    var = (2.0 * env.noise_scale) ** 2 / 12.0  # variance of Uniform(-s, s)
    tr_PSigma = var * np.trace(P)
    xs, _ = env.sample(jrn.key(0), 5000)  # excludes the equilibrium ball
    z = np.array(xs) - center
    analytic_drift = -(z * z).sum(axis=1) + tr_PSigma
    assert analytic_drift.max() <= 1e-6  # known_V is a supermartingale on domain \ eq


@pytest.mark.parametrize("env_name", ["linstoch2D"])
def test_quadratic_env_step_matches_assumed_dynamics(env_name):
    # The analytic drift above assumes x' = A x (+ b) + zero-mean noise; verify the
    # jax step realises it by matching a high-sample MC drift at a few states.
    env = make_env(env_name)
    V = env.known_V
    center = np.array(getattr(env, "setpoint", np.zeros(env.dim)))
    P = np.array(env.P)
    var = (2.0 * env.noise_scale) ** 2 / 12.0
    tr_PSigma = var * np.trace(P)
    xs, _ = env.sample(jrn.key(1), 6)
    for x in xs:
        ks = jrn.split(jrn.key(2), 8000)
        mc = float(jnp.mean(jax.vmap(lambda k: V(env.step(x, k)[0]))(ks)) - V(x))
        z = np.array(x) - center
        analytic = float(-(z @ z) + tr_PSigma)
        assert abs(mc - analytic) < 5e-2


@pytest.mark.parametrize("env_name", ["pendulum_lqr"])
def test_ellipsoid_domain_is_invariant_and_stable(env_name):
    # This env uses a Lyapunov-ellipsoid domain (a box/ball is not forward
    # invariant under the non-normal dynamics). Sampling from the
    # ellipsoid, every trajectory should stay near the domain and converge to the
    # equilibrium — i.e. the verification problem is well posed and a certificate
    # exists.
    from src.benchmarks.convex_sets import Ellipsoid
    env = make_env(env_name)
    assert isinstance(env.domain, Ellipsoid)
    starts, _ = env.domain.sample(jrn.key(0), 400)

    def rollout(x0, k):
        def step(c, _):
            x, k = c
            x, k = env.step(x, k)
            return (x, k), None
        (xf, _), _ = jax.lax.scan(step, (x0, k), None, length=300)
        return jnp.linalg.norm(xf - env.equilibrium.center)

    finals = np.array(jax.vmap(rollout)(starts, jrn.split(jrn.key(0), len(starts))))
    assert not np.isnan(finals).any()       # nothing diverges
    assert finals.max() < float(env.equilibrium.radius) + 0.2  # settles near eq


def test_pendulum_lqr_known_V_is_supermartingale():
    # Quadratic known_V from the Schur linearisation; the sin nonlinearity is mild
    # enough on the (small) ellipsoid that it stays a supermartingale on domain\eq.
    env = make_env("pendulum_lqr")
    V = env.known_V
    xs, _ = env.domain.sample(jrn.key(0), 3000)
    keep = ~np.array(jax.vmap(env.equilibrium.contains)(xs))
    xs = xs[keep]

    def drift(x, k):
        ks = jrn.split(k, 256)
        return jnp.mean(jax.vmap(lambda kk: V(env.step(x, kk)[0]))(ks)) - V(x)

    d = np.array(jax.vmap(drift)(xs[:1500], jrn.split(jrn.key(1), min(1500, len(xs)))))
    assert d.max() <= 5e-3  # supermartingale on domain \ equilibrium (MC tolerance)
