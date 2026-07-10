"""
Environment registry — the single name -> Env mapping used by experiments.

`DIRECTORY` maps a short config name (e.g. "linear2D") to a zero-argument
factory that builds a fresh Env. Factories (not instances) keep envs cheap to
register and avoid sharing mutable state across runs.
"""

from src.benchmarks.linear_system import SwitchedLinearEnv
from src.benchmarks.linear_stochastic import LinearStochasticEnv
from src.benchmarks.double_well import DoubleWellEnv
from src.benchmarks.van_der_pol import VanDerPolEnv
from src.benchmarks.building_automation import BuildingThermalEnv
from src.benchmarks.inverted_pendulum import InvertedPendulumEnv
from src.benchmarks.inverted_pendulum_lqr import InvertedPendulumLQREnv


DIRECTORY = {
    # Switched-linear, default 2D matrices (L1-contractive; exact in every engine)
    "linear2D": lambda: SwitchedLinearEnv(p=0.5),
    "linear2D_p0": lambda: SwitchedLinearEnv(p=0.0),
    "linear2D_p1": lambda: SwitchedLinearEnv(p=1.0),
    # Switched-linear, randomly constructed at a target spectral norm (any dim)
    "linear2D_rand": lambda: SwitchedLinearEnv.random_construction(2, 0.8),
    "linear3D": lambda: SwitchedLinearEnv.random_construction(3, 0.8),
    "linear4D": lambda: SwitchedLinearEnv.random_construction(4, 0.8),
    "linear3D_steep": lambda: SwitchedLinearEnv.random_construction(3, 0.95),
    # Linear with additive bounded noise (quadratic known_V; exact in every engine)
    "linstoch2D": lambda: LinearStochasticEnv(),
    "linstoch2D_tri": lambda: LinearStochasticEnv(noise_kind="triangular"),
    # Fatter-equilibrium VERIFY variants (results/eq_noise_probe, 2026-07-09):
    # training pins the thinnest drift margin exactly on the eq rim (the trainer
    # rejection-samples domain\eq and the hinge loss stops at -train_eps), so
    # verifying with the SAME eq makes every sound engine tile/relax against a
    # ~3x-thinner margin than the bulk. Verifying the same cert against a 2x
    # eq radius (eq_margin 3.0 vs the 1.5 default) excludes that shell:
    # milp/crown flip from budget-exhausted to ~6s verified, mab from an
    # unbounded grind to minutes. Pass via the cegis runner's --verify_env.
    "linstoch2D_eqm2": lambda: LinearStochasticEnv(eq_margin=2.0),
    "linstoch2D_eqm3": lambda: LinearStochasticEnv(eq_margin=3.0),
    "linstoch3D": lambda: LinearStochasticEnv.random_construction(3, 0.9),
    "linstoch4D": lambda: LinearStochasticEnv.random_construction(4, 0.9),
    # Double-well gradient flow (continuous noise; sound over-approx in SMT/MILP).
    # DEFAULT = the tuned geometry (was "doublewell_mab"): the legacy default
    # (eq (1.0, r=0.1)) MISSES the true attractor at x*=1.125, whose noise floor
    # reaches r=0.121 — NO valid certificate existed, so every engine refuted
    # forever. Recentred on the attractor, eq enlarged, domain = a +-0.7
    # neighbourhood (2026-07-03: MAB + MILP + CROWN all verify a cert here).
    "doublewell": lambda: DoubleWellEnv(
        eq_center=(1.12, 0.0), eq_radius=0.45,
        domain_bounds=((0.42, 1.82), (-0.7, 0.7))),
    "doublewell_legacy": lambda: DoubleWellEnv(),  # cert-free geometry (see above)
    "doublewell_fine": lambda: DoubleWellEnv(noise_disc=4),
    # Reversed Van der Pol (polynomial nonlinear; stable origin; FOSSIL bridge)
    "vanderpol": lambda: VanDerPolEnv(),
    "vanderpol_tri": lambda: VanDerPolEnv(noise_kind="triangular"),
    # Building-automation thermal regulation (affine linear; BAS family; setpoint)
    "thermal": lambda: BuildingThermalEnv(),
    # Inverted pendulum with a trained NN controller (nonlinear; sampling engine)
    "pendulum": lambda: InvertedPendulumEnv(),
    # Inverted pendulum, FOSSIL LQR feedback (spiral; ellipsoidal; quadratic known_V).
    # DEFAULT = the tuned geometry (was "pendulum_lqr_mab"): the legacy eq
    # r=0.035 is razor-thin against the noise floor -> untrainable margins.
    "pendulum_lqr": lambda: InvertedPendulumLQREnv(level=0.5, eq_radius=0.15),
    "pendulum_lqr_legacy": lambda: InvertedPendulumLQREnv(),

    # -- MAB-feasible tuned variants ------------------------------------------
    # The default nonlinear envs put the equilibrium / domain right where the
    # achievable drift margin -> 0, which the sound sample-based MAB verifier
    # cannot tile to (box count ~ (domain * L / margin)^d). These variants size
    # the equilibrium to comfortably cover the noise floor and shrink the domain
    # to a neighbourhood where a bounded certificate holds a workable margin, so
    # CEGIS terminates with a MAB-verified certificate (sound, if slow).
    #
    #   * doublewell: default eq (1.0, r=0.1) MISSES the true attractor at
    #     x*=1.125 (its noise floor reaches r=0.121) -> no valid cert exists.
    #     Recentre on the attractor, enlarge, and verify a +-0.7 neighbourhood
    #     (MILP-confirmed margin ~7.9e-3 here).
    #   * vanderpol: default level=6 sits on the ROA boundary (unstable limit
    #     cycle) where the drift -> 0. level=1.5 pulls the domain inside the ROA.
    #   * pendulum_lqr: default eq r=0.035 is razor-thin; enlarge to 0.15.
    # doublewell_mab / pendulum_lqr_mab are now ALIASES of the promoted defaults
    # (kept so old campaign lines and results remain resolvable).
    "doublewell_mab": lambda: DoubleWellEnv(
        eq_center=(1.12, 0.0), eq_radius=0.45,
        domain_bounds=((0.42, 1.82), (-0.7, 0.7))),
    "vanderpol_mab": lambda: VanDerPolEnv(level=1.5, eq_radius=0.35),
    "pendulum_lqr_mab": lambda: InvertedPendulumLQREnv(level=0.5, eq_radius=0.15),
}


def make_env(name: str):
    try:
        return DIRECTORY[name]()
    except KeyError:
        raise ValueError(
            f"No such env '{name}'. Available: {sorted(DIRECTORY)}"
        )
