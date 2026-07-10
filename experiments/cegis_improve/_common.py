"""
Shared arg-group builder for the experiment-4 (cegis_improve) launchers.

For a fixed engine, sweeps each active refuter over the shared envs x seeds x
history_len, using the best hyperparameters from best_hp. Engine hyperparameters
pass as bare flags (-> engine); refuter hyperparameters beyond budget/batch_size
are r_-prefixed (-> refuter), so the two never collide on the command line.
Returns a list of arg groups, which launch() expands and concatenates.

`noise_groups` (from the engine's cegis_verify twin) gives each env family its
own noise_disc axis; without it every env runs with the engine's default disc.
`cegis` overrides config.CEGIS wholesale (e.g. mc's bounded_pwl requirement).
"""

from __future__ import annotations

from experiments.common import config
from experiments.cegis_improve import best_hp


def groups(engine: str, *, noise_groups: list[dict] | None = None,
           cegis: dict | None = None) -> list[dict]:
    eng_hp = best_hp.ENGINE[engine]
    base = cegis if cegis is not None else config.CEGIS
    ngs = noise_groups if noise_groups is not None else [dict(env=config.ENVS)]
    out: list[dict] = []
    for rname in best_hp.ACTIVE_REFUTERS:
        for ng in ngs:
            g = dict(**ng, seed=config.SEEDS, **base,
                     engine=engine, refuter=rname, **eng_hp)
            g["history_len"] = best_hp.HISTORY_LEN  # override the scalar from base
            if rname != "none":
                rhp = dict(best_hp.REFUTER[rname])
                for core in ("budget", "batch_size"):
                    if core in rhp:
                        g[core] = rhp.pop(core)        # runner core flags
                g.update({f"r_{k}": v for k, v in rhp.items()})  # -> the refuter
            out.append(g)
    return out
