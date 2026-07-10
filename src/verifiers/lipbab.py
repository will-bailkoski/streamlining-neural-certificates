"""
Method of gaining tighter Lipschitz bounds on a ReLU neural network, as estabilished in
Bhowmick, A., D'Souza, M., & Raghavan, G. S. (2021, September). LipBaB: computing exact Lipschitz constant of ReLU networks.
https://arxiv.org/abs/2105.05495

Adapted to return Jacobian matrix for cancelation.
"""

import numpy as np
import time
from collections import deque
from queue import PriorityQueue
from cvxopt import matrix, solvers
from copy import deepcopy

solvers.options["show_progress"] = False


class LipBaBSubproblem:
    def __init__(self):
        self.Lub = 0.0
        self.actvp = []
        self.ast_ns = deque()
        self.H = []
        self.t = 0
        self.tev = []
        self.depth = 0  # <--- Added to track depth in the tree


class LipBaBSolver:
    def __init__(self, weights, biases, bnds, pnorm=2, af=1.0, max_depth=None):
        self.weights = weights
        self.biases = biases
        self.bnds = bnds
        self.pnorm = pnorm
        self.af = af
        self.max_depth = max_depth  # <--- Added max depth constraint

        self.layer_sizes = [len(weights[1][0])] + [
            len(weights[i]) for i in range(1, len(weights))
        ]
        self.L = len(self.layer_sizes) - 1
        self.input_dim = self.layer_sizes[0]

        self.output_bounds = []
        self.ast_neurons = deque()
        self.var = [[b[0], b[1]] for b in bnds]
        self.pcnt = 0
        self.pq = PriorityQueue()
        self.glb = 0.0

        # Track the worst-case concrete Jacobian matrix found so far
        self.max_jacobian = None  # <--- Added to store the actual matrix

        self.act_pat = [[[1, 1] for _ in range(size)] for size in self.layer_sizes]

    @staticmethod
    def rdn(a):
        return round(a - 5e-16, 15)

    @staticmethod
    def rup(a):
        return round(a + 5e-16, 15)

    def ii_add(self, a, b):
        return [self.rdn(a[0] + b[0]), self.rup(a[1] + b[1])]

    def ii_mul(self, a, b):
        mn = min(
            self.rdn(a[0] * b[0]),
            self.rdn(a[0] * b[1]),
            self.rdn(a[1] * b[0]),
            self.rdn(a[1] * b[1]),
        )
        mx = max(
            self.rup(a[0] * b[0]),
            self.rup(a[0] * b[1]),
            self.rup(a[1] * b[0]),
            self.rup(a[1] * b[1]),
        )
        return [mn, mx]

    def ic_mul(self, a, b):
        return [
            min(self.rdn(b * a[0]), self.rdn(b * a[1])),
            max(self.rup(b * a[0]), self.rup(b * a[1])),
        ]

    @staticmethod
    def mm_mul(m1, m2):
        return np.dot(np.array(m1), np.array(m2)).tolist()

    @staticmethod
    def mv_spmul(m1, m2):
        m1_arr = np.array(m1)
        m2_arr = np.array(m2).flatten()
        return (m1_arr * m2_arr).tolist()

    def mim_mul(self, m1, m2):
        m = [[[0.0, 0.0] for _ in range(len(m2[0]))] for _ in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m2)):
                    m[i][j] = self.ii_add(m[i][j], self.ic_mul(m2[k][j], m1[i][k]))
        return m

    def ivim_spmul(self, m1, m2):
        m = [[[0.0, 0.0] for _ in range(len(m2[0]))] for _ in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                m[i][j] = self.ii_mul(m1[i], m2[i][j])
        return m

    def e_bounds(self, a, b):
        lb, ub = 0.0, 0.0
        for i in range(len(a)):
            if a[i] > 0:
                lb = self.rdn(lb + self.rdn(a[i] * self.var[i][0]))
                ub = self.rup(ub + self.rup(a[i] * self.var[i][1]))
            elif a[i] < 0:
                lb = self.rdn(lb + self.rdn(a[i] * self.var[i][1]))
                ub = self.rup(ub + self.rup(a[i] * self.var[i][0]))
        lb = self.rdn(lb + b)
        ub = self.rup(ub + b)
        return lb, ub

    def symprop(self):
        ev = [np.identity(self.input_dim), np.zeros(self.input_dim)]
        for l in range(1, self.L + 1):
            ev = [
                np.dot(self.weights[l], ev[0]),
                np.dot(self.weights[l], ev[1]) + self.biases[l],
            ]
            if l == self.L:
                break
            for i in range(len(ev[0])):
                lb, ub = self.e_bounds(ev[0][i], ev[1][i])
                if lb >= 0:
                    self.act_pat[l][i] = [1, 1]
                elif ub <= 0:
                    self.act_pat[l][i] = [0, 0]
                    ev[0][i] = np.zeros(len(ev[0][i]))
                    ev[1][i] = 0.0
                else:
                    self.act_pat[l][i] = [0, 1]
                    self.ast_neurons.append([l, i])
                    ev[0] = np.hstack((ev[0], np.zeros((len(ev[0]), 1))))
                    ev[0][i] = np.zeros(len(ev[0][i]))
                    ev[1][i] = 0.0
                    ev[0][i][len(ev[0][i]) - 1] = 1.0
                    self.var.append([0.0, ub])

        for i in range(len(ev[0])):
            lb, ub = self.e_bounds(ev[0][i], ev[1][i])
            self.output_bounds.append([lb, ub])

    def lip_bound(self, p):
        """Calculates Upper Bound and extracts the corresponding Jacobian matrix."""
        if len(p.ast_ns) == 0:
            # Concrete leaf node: compute the exact concrete Jacobian matrix.
            # Every neuron is fixed here, so actvp[l] holds [s, s] pairs with the
            # ReLU slope s in {0, 1}; take the scalar slope per neuron (not the pair).
            J = deepcopy(self.weights[self.L])
            for l in range(self.L - 1, 0, -1):
                slopes = [a[0] for a in p.actvp[l]]
                J = self.mv_spmul(J, slopes)
                J = self.mm_mul(J, self.weights[l])

            norm_val = np.linalg.norm(J, self.pnorm)

            # If this is the highest verified lower-bound norm so far, cache the actual Jacobian matrix
            if norm_val > self.glb:
                self.max_jacobian = np.array(J)

            return norm_val
        else:
            # Intermediate node: compute interval Jacobian matrix
            J = [
                [
                    [self.weights[1][i][j], self.weights[1][i][j]]
                    for j in range(len(self.weights[1][0]))
                ]
                for i in range(len(self.weights[1]))
            ]
            for l in range(2, self.L + 1):
                J = self.ivim_spmul(p.actvp[l - 1], J)
                J = self.mim_mul(self.weights[l], J)
            U = [
                [max(abs(J[i][j][0]), abs(J[i][j][1])) for j in range(len(J[0]))]
                for i in range(len(J))
            ]

            norm_val = np.linalg.norm(U, self.pnorm)

            # Fallback optimization context: if we have to force-stop early (like max depth),
            # we can save the worst-case absolute scalar bound upper-limit matrix profile.
            if self.max_jacobian is None or norm_val > self.glb:
                self.max_jacobian = np.array(U)

            return norm_val

    def linprop(self, p):
        l = p.t
        ev = p.tev
        while l < self.L:
            if not p.ast_ns or p.ast_ns[0][0] == l:
                break
            for i in range(len(ev[0])):
                if p.actvp[l][i] == [0, 0]:
                    ev[0][i] = np.zeros(len(ev[0][i]))
                    ev[1][i] = 0.0
            l += 1
            ev = [
                np.dot(self.weights[l], ev[0]),
                np.dot(self.weights[l], ev[1]) + self.biases[l],
            ]
        p.t = l
        p.tev = ev

    def ffilter(self, p):
        A, b = deepcopy(p.H[0]), deepcopy(p.H[1])
        while len(p.ast_ns) > 0:
            l, i = p.ast_ns[0][0], p.ast_ns[0][1]
            A.append(p.tev[0][i].tolist())
            b.append(-p.tev[1][i])
            sol = solvers.lp(
                matrix([0.0] * self.input_dim),
                matrix(np.array(A)),
                matrix(b),
                solver="glpk",
                options={"glpk": {"msg_lev": "GLP_MSG_OFF"}},
            )
            if sol["status"] != "optimal":
                p.actvp[l][i] = [1, 1]
                p.ast_ns.popleft()
                self.linprop(p)
                A.pop()
                b.pop()
                continue
            A.pop()
            b.pop()
            A.append((-p.tev[0][i]).tolist())
            b.append(p.tev[1][i])
            sol = solvers.lp(
                matrix([0.0] * self.input_dim),
                matrix(np.array(A)),
                matrix(b),
                solver="glpk",
                options={"glpk": {"msg_lev": "GLP_MSG_OFF"}},
            )
            if sol["status"] != "optimal":
                p.actvp[l][i] = [0, 0]
                p.ast_ns.popleft()
                self.linprop(p)
                A.pop()
                b.pop()
                continue
            A.pop()
            b.pop()
            break

    def branch(self, p, sgn):
        l, i = p.ast_ns[0][0], p.ast_ns[0][1]
        A, b = deepcopy(p.H[0]), deepcopy(p.H[1])
        A.append((-sgn * p.tev[0][i]).tolist())
        b.append(sgn * p.tev[1][i])
        res = solvers.lp(
            matrix([0.0] * self.input_dim),
            matrix(np.array(A)),
            matrix(b),
            solver="glpk",
            options={"glpk": {"msg_lev": "GLP_MSG_OFF"}},
        )
        if res["status"] == "optimal":
            pb = LipBaBSubproblem()
            self.pcnt += 1
            pb.H = [A, b]
            pb.actvp = deepcopy(p.actvp)
            pb.ast_ns = deepcopy(p.ast_ns)
            pb.t = p.t
            pb.tev = deepcopy(p.tev)
            pb.depth = p.depth + 1  # <--- Increment subproblem depth tracking
            pb.actvp[l][i] = [max(0, sgn), max(0, sgn)]
            pb.ast_ns.popleft()
            self.linprop(pb)
            self.ffilter(pb)
            pb.Lub = self.lip_bound(pb)
            self.pq.put((-pb.Lub, self.pcnt, pb))
            if len(pb.ast_ns) == 0:
                self.glb = max(self.glb, pb.Lub)

    def solve(self, time_out=300, verbose=True):
        start_time = time.time()
        HC = [[], []]
        for i in range(self.input_dim):
            b = self.bnds[i]
            a = [0.0] * self.input_dim
            a[i] = -1.0
            HC[0].append(a)
            HC[1].append(-b[0])
            a = [0.0] * self.input_dim
            a[i] = 1.0
            HC[0].append(a)
            HC[1].append(b[1])

        self.symprop()
        self.glb = 0.0

        p = LipBaBSubproblem()
        self.pcnt += 1
        p.H = deepcopy(HC)
        p.actvp = deepcopy(self.act_pat)
        p.ast_ns = deepcopy(self.ast_neurons)
        p.tev = [np.identity(self.input_dim), np.zeros(self.input_dim)]
        p.depth = 0

        self.linprop(p)
        p.Lub = self.lip_bound(p)
        self.pq.put((-p.Lub, self.pcnt, p))

        if len(p.ast_ns) == 0:
            self.glb = p.Lub

        tp = p
        while not self.pq.empty():
            if (time.time() - start_time) >= time_out:
                if verbose:
                    print(f"Timeout reached ({time_out}s).")
                break

            tp = self.pq.get()[2]
            if verbose:
                print(f"Current Upper Bound: {tp.Lub:.6f} | Depth: {tp.depth}")

            if tp.Lub <= self.af * self.glb:
                break

            # <--- Max Depth Condition Check before executing branch splits
            if self.max_depth is not None and tp.depth >= self.max_depth:
                if verbose:
                    print(
                        f"Skipping branch split: Max depth threshold ({self.max_depth}) reached."
                    )
                continue

            self.branch(tp, -1)
            self.branch(tp, 1)

        execution_time = time.time() - start_time
        return tp.Lub, self.max_jacobian, execution_time


# ------------------ UPDATED INTERFACE METHOD ------------------


def lipschitz_v_bound(spec, params, boundaries, method="lipbab", timeout=20.0):
    """Sound upper bound on the Lipschitz constant of V = spec.forward(params, .).

    Composes the Lipschitz bound of the raw ReLU net with the output activation's
    Lipschitz constant (`spec.output_lipschitz`) — so a hard-sigmoid output's 1/6
    contraction is captured, not dropped.

      method='lipbab'   : LipBaB branch-and-bound on the raw net (tight; needs ReLU
                          hidden; sound even on timeout). Falls back to spectral for
                          non-ReLU hidden activations.
      method='spectral' : product of per-layer spectral norms (loose, any hidden).

    `boundaries` is the input domain as (d, 2) [lo, hi] rows.
    """
    from src.certificates.structures import get_spectral_norm_product

    out_lip = float(spec.output_lipschitz)
    if method == "lipbab" and spec.hidden_activation == "relu":
        weights = [None] + [np.asarray(W, dtype=float) for W, _ in params]
        biases = [None] + [np.asarray(b, dtype=float) for _, b in params]
        bnds = np.asarray(boundaries, dtype=float).tolist()
        l_raw, _jac, _secs = compute_lipschitz(
            weights, biases, bnds, pnorm=2, timeout=timeout, verbose=False
        )
        return out_lip * float(l_raw)
    return out_lip * float(get_spectral_norm_product(params))


def compute_lipschitz(
    weights,
    biases,
    boundaries,
    pnorm=2,
    approximation_factor=1.0,
    timeout=300,
    max_depth=None,
    verbose=True,
):
    """
    Returns:
    --------
    estimated_lipschitz : float
        The ultimate scalar upper bound.
    max_jacobian : np.ndarray
        The underlying Jacobian matrix corresponding to the maximum norm region.
    elapsed_time : float
        Runtime execution info.
    """
    solver = LipBaBSolver(
        weights=weights,
        biases=biases,
        bnds=boundaries,
        pnorm=pnorm,
        af=approximation_factor,
        max_depth=max_depth,
    )
    return solver.solve(time_out=timeout, verbose=verbose)
