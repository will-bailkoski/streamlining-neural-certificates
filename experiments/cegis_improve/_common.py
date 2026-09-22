"""
Shared arg-group builder for the experiment-4 (cegis_improve) launchers.

For a fixed engine, sweeps each active refuter over the Chapter 7 systems x seeds,
using the tuned refuter hyperparameters from best_hp. Engine hyperparameters pass
as bare flags (-> engine); refuter hyperparameters beyond budget/batch_size are
r_-prefixed (-> refuter), so the two never collide on the command line. Returns a
list of arg groups, which launch() expands and concatenates.
"""

from __future__ import annotations

from experiments.common import config
from experiments.cegis_improve import best_hp


def groups(engine: str) -> list[dict]:
    eng_hp = best_hp.ENGINE[engine]
    out: list[dict] = []
    for rname in best_hp.ACTIVE_REFUTERS:
        g = dict(env=config.STREAMLINE_ENVS, seed=config.STREAMLINE_SEEDS,
                 **config.CEGIS, engine=engine, refuter=rname, **eng_hp)
        g["history_len"] = best_hp.HISTORY_LEN  # override the scalar from base
        rhp = dict(best_hp.REFUTER[rname])
        for core in ("budget", "batch_size"):
            if core in rhp:
                g[core] = rhp.pop(core)        # runner core flags
        g.update({f"r_{k}": v for k, v in rhp.items()})  # -> the refuter
        out.append(g)
    return out
