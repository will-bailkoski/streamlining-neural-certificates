"""
CEGIS experiment entrypoint — one config, any engine.

    parse_args() -> ExperimentConfig -> main(config)

The loop is engine-agnostic: train a certificate V, ask the chosen engine for a
counterexample to the drift condition, and either refine (counterexample),
finish (verified), or stop (inconclusive). All result I/O goes through
`src.results.recorder.Recorder`. Every engine checks the SAME drift
`E_w[V(x')] - V(x) + epsilon` over `domain \\ equilibrium`; counterexamples from
the sound-but-over-approximating engines are re-checked with the jax drift
before being fed back to the trainer.

Examples:
    python -m src.experiment --env linear2D --engine smt --hidden_layers 8,8
    python -m src.experiment --env linear2D --engine milp --epsilon 1e-3
    python -m src.experiment --env doublewell --engine sampling
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    env: str = "linear2D"
    engine: str = "smt"
    cert_structure: str = "bounded_pwl"
    hidden_layers: str = "8,8"
    epsilon: float = 1e-3
    key: int = 0
    run_id: Optional[str] = None
    results_dir: str = "results"
    # verifier
    noise_disc: int = 1  # noise cells per dim for lirpa/symbolic over-approximation
    # training
    sample_count: int = 2000
    batch_size: int = 256
    lr: float = 1e-2
    mc_samples: int = 16
    max_epochs: int = 200
    # CEGIS
    max_rounds: int = 30
    # engine knobs (forwarded as **hp to find_counterexample)
    engine_kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        d = {k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        return cls(**d)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CEGIS neural-certificate experiment")
    p.add_argument("--env", default="linear2D")
    p.add_argument("--engine", default="smt", choices=["smt", "milp", "sampling", "lirpa"])
    p.add_argument("--cert_structure", default="bounded_pwl")
    p.add_argument("--hidden_layers", default="8,8")
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--key", type=int, default=0)
    p.add_argument("--run_id", default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--noise_disc", type=int, default=1)
    p.add_argument("--sample_count", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc_samples", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--max_rounds", type=int, default=30)
    return p.parse_args()


def main(config: ExperimentConfig) -> bool:
    import jax.numpy as jnp
    import jax.random as jrn

    from src.benchmarks.utils import make_env
    from src.certificates.structures import get_spec, glorot_init, get_spectral_norm_product
    from src.verifiers.base import get_engine
    import src.verifiers  # noqa: F401  (registers engines)
    from src.verifiers.drift import recheck_violation
    from src.training.training_cycles import train_fixed_dataset, cegis_trainer
    from src.results.recorder import Recorder
    from src.results.types import RoundSummary

    env = make_env(config.env)
    spec = get_spec(config.cert_structure)
    engine = get_engine(config.engine)
    cert = spec.forward
    epsilon = config.epsilon
    training_eps = epsilon * 10.0  # train to a stricter margin than we verify

    hidden = [int(x) for x in config.hidden_layers.split(",") if x != ""]
    layer_sizes = [env.dim] + hidden + [1]

    run_id = config.run_id or f"{config.env}_{config.engine}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    recorder = Recorder(config.results_dir, run_id, config.to_dict())
    print(f"[run {run_id}] env={config.env} engine={config.engine} cert={config.cert_structure}")

    train_hp = dict(
        batch_size=config.batch_size,
        lr=config.lr,
        n=config.mc_samples,
        max_epochs=config.max_epochs,
        threshold=0.0,
        decay=0.9,
        certificate=cert,
        printing=False,
    )

    try:
        dataset_key, key = jrn.split(jrn.key(config.key))
        init = glorot_init(layer_sizes, jrn.fold_in(jrn.key(config.key), 99))
        params, _, loss, full_x = train_fixed_dataset(
            env, training_eps, init, N=config.sample_count, dataset_key=dataset_key, **train_hp
        )
        gen = cegis_trainer(env, training_eps, params, seed_dataset=full_x, **train_hp)
        params, _ = next(gen)

        for rnd in range(config.max_rounds):
            vkey, key = jrn.split(key)
            result = engine.find_counterexample(
                env, spec, params, epsilon, key=vkey,
                noise_disc=config.noise_disc, **config.engine_kwargs
            )
            lip = float(get_spectral_norm_product(params))
            vtime = float(result.stats.get("time", 0.0))

            if result.violation is not None:
                rk, key = jrn.split(key)
                genuine, drift = recheck_violation(env, spec, params, epsilon, result.violation[0], rk)
                summary = RoundSummary(
                    round=rnd, verdict="counterexample", n_counterexamples=len(result.violation),
                    train_samples=int(full_x.shape[0]) + rnd, loss=float(loss), lipschitz=lip,
                    verifier_time=vtime, counterexample_genuine=genuine,
                    counterexample_drift=drift, engine=result.engine,
                )
                recorder.log_round(summary, params=params, counterexamples=result.violation)
                print(f"  round {rnd}: counterexample (genuine={genuine}, drift={drift:.4f}) "
                      f"via {result.engine} in {vtime:.2f}s")
                if not genuine:
                    print("  spurious counterexample (over-approximation too loose); stopping.")
                    recorder.finish(status="inconclusive", verified=False)
                    return False
                params, loss = gen.send(jnp.array(result.violation))
                continue

            if result.verified:
                summary = RoundSummary(
                    round=rnd, verdict="verified", n_counterexamples=0,
                    train_samples=int(full_x.shape[0]) + rnd, loss=float(loss), lipschitz=lip,
                    verifier_time=vtime, engine=result.engine,
                )
                recorder.log_round(summary, params=params)
                recorder.finish(status="completed", verified=True)
                print(f"  round {rnd}: VERIFIED via {result.engine} in {vtime:.2f}s")
                return True

            # inconclusive (budget/timeout/refuter exhausted)
            summary = RoundSummary(
                round=rnd, verdict="inconclusive", n_counterexamples=0,
                train_samples=int(full_x.shape[0]) + rnd, loss=float(loss), lipschitz=lip,
                verifier_time=vtime, engine=result.engine,
            )
            recorder.log_round(summary, params=params)
            recorder.finish(status="inconclusive", verified=None)
            print(f"  round {rnd}: inconclusive via {result.engine}; stopping.")
            return False

        recorder.finish(status="inconclusive", verified=None)
        print(f"  reached max_rounds={config.max_rounds} without a verdict.")
        return False

    except BaseException:
        recorder.finish(status="failed", verified=None)
        raise


if __name__ == "__main__":
    main(ExperimentConfig.from_args(parse_args()))
