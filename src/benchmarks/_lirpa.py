"""
Shared torch helper for the continuous-noise autoLiRPA path.

The drift over continuous noise is over-approximated exactly like the symbolic
engines: partition the noise support into K cells (boxes), and bound

    d(x) = sum_k weight_k * V(step(x, w_k)) - V(x) + epsilon

jointly over the state box and the per-cell noise boxes. autoLiRPA then picks the
worst noise in each cell (a sound upper bound on E[V]); refining the cell count
(`noise_disc`) tightens it.

The noise cells are packed into a single (B, K * noise_dim) input and sliced
along the FEATURE axis, so the box-batch dimension is preserved throughout —
which keeps the bound propagation well-behaved.
"""


def cell_drift_module(V_module, step_module, weights, noise_dim, epsilon):
    """
    V_module:    nn.Module x -> V(x)            (B, d) -> (B, 1)
    step_module: nn.Module (x, w) -> next state (B, d), (B, nd) -> (B, d)
    weights:     per-cell probabilities (length K, summing to 1)
    noise_dim:   nd, the size of each cell's noise vector
    """
    import torch.nn as nn

    w = [float(x) for x in weights]

    class _CellDrift(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = V_module
            self.step = step_module
            self.weights = w
            self.K = len(w)
            self.nd = int(noise_dim)
            self.eps = float(epsilon)

        def forward(self, x, w_packed):  # x:(B,d)  w_packed:(B, K*nd)
            out = -self.V(x) + self.eps
            for k in range(self.K):
                wk = w_packed[:, k * self.nd : (k + 1) * self.nd]
                out = out + self.weights[k] * self.V(self.step(x, wk))
            return out

    return _CellDrift()
