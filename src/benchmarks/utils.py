"""
Environment registry — the single name -> Env mapping used by experiments.

`DIRECTORY` maps a short config name (e.g. "linear2D") to a zero-argument
factory that builds a fresh Env. Factories (not instances) keep envs cheap to
register and avoid sharing mutable state across runs.

Thesis names (Chapter 4): LinSwitch = linear{2,3,4}D, LinStoch = linstoch{2,3,4}D,
DoubleWell = doublewell, Pendulum = pendulum_lqr.
"""

from src.benchmarks.linear_system import SwitchedLinearEnv
from src.benchmarks.linear_stochastic import LinearStochasticEnv
from src.benchmarks.double_well import DoubleWellEnv
from src.benchmarks.inverted_pendulum_lqr import InvertedPendulumLQREnv


DIRECTORY = {
    # Switched-linear, default 2D matrices (L1-contractive; exact in every engine)
    "linear2D": lambda: SwitchedLinearEnv(p=0.5),
    "linear2D_p0": lambda: SwitchedLinearEnv(p=0.0),
    "linear2D_p1": lambda: SwitchedLinearEnv(p=1.0),
    # Switched-linear, randomly constructed at a target spectral norm (any dim)
    "linear3D": lambda: SwitchedLinearEnv.random_construction(3, 0.8),
    "linear4D": lambda: SwitchedLinearEnv.random_construction(4, 0.8),
    # Linear with additive bounded noise (quadratic known_V; exact in every engine)
    "linstoch2D": lambda: LinearStochasticEnv(),
    "linstoch3D": lambda: LinearStochasticEnv.random_construction(3, 0.9),
    "linstoch4D": lambda: LinearStochasticEnv.random_construction(4, 0.9),
    # Double-well gradient flow (continuous noise; sound over-approx in SMT/MILP).
    # The equilibrium is centred on the true attractor x* ~ 1.125 and enlarged to
    # cover its noise floor; the domain is a +-0.7 neighbourhood of it.
    "doublewell": lambda: DoubleWellEnv(
        eq_center=(1.12, 0.0), eq_radius=0.45,
        domain_bounds=((0.42, 1.82), (-0.7, 0.7))),
    # Inverted pendulum, LQR feedback (spiral; ellipsoidal; quadratic known_V).
    "pendulum_lqr": lambda: InvertedPendulumLQREnv(level=0.5, eq_radius=0.15),
}


def make_env(name: str):
    try:
        return DIRECTORY[name]()
    except KeyError:
        raise ValueError(
            f"No such env '{name}'. Available: {sorted(DIRECTORY)}"
        )
