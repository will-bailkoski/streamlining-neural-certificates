from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Patch

OUT = Path(__file__).resolve().parent / "mab_refine_demo.png"
rng = np.random.default_rng(1)

# --- drift  D(x,y) <= 0 everywhere (a valid certificate), thin margin near the centre ---
def D(x, y):
    return -0.82 + 0.62 * np.exp(-7.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

# sound-ish drift-Lipschitz constant (max |grad D| over the domain, inflated)
xx, yy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
gx, gy = np.gradient(D(xx, yy), 1 / 399)
L_D = 1.1 * np.max(np.sqrt(gx ** 2 + gy ** 2))

SNOISE = 0.02          # observation noise on each drift sample
C_STAT = 0.14          # statistical-radius constant (shrinks as 1/sqrt(n))
M0     = 7             # samples drawn per cell when it is opened

class Cell:
    def __init__(self, x0, y0, w):
        self.x0, self.y0, self.w = x0, y0, w
        self.sx, self.sy, self.sv = [], [], []
        self.status = "undecided"
    def diam(self):
        return self.w * np.sqrt(2)
    def sample(self, m):
        px = self.x0 + rng.random(m) * self.w
        py = self.y0 + rng.random(m) * self.w
        self.sx += list(px); self.sy += list(py)
        self.sv += list(D(px, py) + rng.normal(0, SNOISE, m))
    def evaluate(self):
        n = len(self.sv)
        self.mean = float(np.mean(self.sv))
        self.stat = C_STAT / np.sqrt(n)
        self.disc = L_D * self.diam()
        self.ucb = self.mean + self.stat + self.disc
        self.lcb = self.mean - self.stat - self.disc
        self.status = "safe" if self.ucb <= 0 else ("unsafe" if self.lcb > 0 else "undecided")
    def split(self):
        h = self.w / 2
        return [Cell(self.x0, self.y0, h), Cell(self.x0 + h, self.y0, h),
                Cell(self.x0, self.y0 + h, h), Cell(self.x0 + h, self.y0 + h, h)]

def snap(cells):
    return [dict(x0=c.x0, y0=c.y0, w=c.w, status=c.status, ucb=c.ucb,
                 sx=list(c.sx), sy=list(c.sy)) for c in cells]

# --- run the MAB refinement, capturing one snapshot per round ---
cells = [Cell(i / 4, j / 4, 1 / 4) for i in range(4) for j in range(4)]
for c in cells:
    c.sample(M0); c.evaluate()
snaps = [snap(cells)]

for _ in range(2):                                  # two refinement rounds
    nxt = []
    for c in cells:
        if c.status == "undecided":
            if c.disc >= c.stat:                    # discretisation dominates -> split
                kids = c.split()
                for k in kids:
                    k.sample(M0); k.evaluate()
                nxt += kids
            else:                                   # statistics dominate -> sample more
                c.sample(2 * M0); c.evaluate()
                nxt.append(c)
        else:
            nxt.append(c)
    cells = nxt
    snaps.append(snap(cells))

# --- plot: three panels, cells shaded green-by-safety, sample points marked ---
def colour(s):
    if s["status"] == "safe":
        m = min(max(-s["ucb"], 0.0) / 0.45, 1.0)    # margin -> deeper green
        return cm.Greens(0.32 + 0.55 * m)
    if s["status"] == "unsafe":
        return "#dc2626"
    return "#fcd34d"                                 # undecided -> amber

titles = ["initial grid", "refine $\\times$1", "refine $\\times$2"]
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
for ax, cells_s, t in zip(axes, snaps, titles):
    for s in cells_s:
        ax.add_patch(Rectangle((s["x0"], s["y0"]), s["w"], s["w"],
                     facecolor=colour(s), edgecolor="#64748b", linewidth=0.7, zorder=1))
        ax.scatter(s["sx"], s["sy"], s=4.5, c="#111827", alpha=0.75,
                   linewidths=0, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    n_safe = sum(s["status"] == "safe" for s in cells_s)
    ax.set_title(f"{t}\n{len(cells_s)} cells, {n_safe} verified", fontsize=11)

legend = [Patch(facecolor=cm.Greens(0.75), edgecolor="#64748b", label="verified safe"),
          Patch(facecolor="#fcd34d", edgecolor="#64748b", label="undecided"),
          plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#111827",
                     markersize=5, label="samples")]
fig.legend(handles=legend, ncol=3, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("saved", OUT)
