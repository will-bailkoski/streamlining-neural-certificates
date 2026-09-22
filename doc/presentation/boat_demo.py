from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = Path(__file__).resolve().parent / "boat_tides_demo.png"

# --- Great Britain outline: approx (lon, lat) coastline landmarks, clockwise from north ---
GB = [
 (-3.4, 58.7), (-3.0, 58.6), (-2.0, 57.7), (-1.8, 57.5), (-2.1, 57.1),
 (-2.5, 56.4), (-2.9, 56.2), (-2.0, 56.0), (-1.8, 55.6), (-1.4, 55.0),
 (-1.1, 54.6), (-0.1, 54.1), (-0.2, 53.6), ( 0.3, 52.9), ( 1.7, 52.5),
 ( 1.2, 51.9), ( 0.9, 51.4), ( 1.4, 51.1), ( 0.3, 50.7), (-1.1, 50.6),
 (-2.5, 50.5), (-3.5, 50.3), (-5.1, 49.95), (-5.7, 50.1), (-4.6, 50.6),
 (-4.2, 51.2), (-3.0, 51.35), (-4.1, 51.6), (-5.3, 51.7), (-4.5, 52.3),
 (-4.8, 52.8), (-4.3, 53.3), (-3.1, 53.4), (-3.0, 54.0), (-3.6, 54.5),
 (-3.5, 54.9), (-4.9, 54.65), (-4.9, 55.3), (-4.8, 55.9), (-5.7, 55.35),
 (-5.4, 56.4), (-6.2, 56.7), (-5.9, 57.3), (-5.0, 58.0), (-5.0, 58.6),
 (-4.5, 58.6),
]

def draw_uk(ax, cx=0.0, cy=0.0, height=1.75, zorder=5):
    p = np.array(GB, float)
    p[:, 0] *= np.cos(np.radians(55.0))     # true lon spacing
    p -= p.mean(0)
    p /= np.abs(p).max()                     # normalise (y-extent ~ [-1, 1])
    p *= height / 2.0
    p[:, 0] += cx; p[:, 1] += cy
    ax.add_patch(Polygon(p, closed=True, fc="#67d9a6", ec="#065f46",
                         lw=2.2, alpha=0.96, zorder=zorder, joinstyle="round"))

# --- tide field: a stable spiral sink at the goal (origin) ---
xmin, xmax, ymin, ymax = -3.2, 3.2, -3.0, 3.0
gy, gx = np.mgrid[ymin:ymax:240j, xmin:xmax:240j]
a, w = 0.32, 0.95
U = -a * gx - w * gy
Vf =  w * gx - a * gy

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.set_facecolor("#E9F3FD")                      # calm light-blue sea

# tides (bluish streamlines)
ax.streamplot(gx, gy, U, Vf, density=1.35, color="#3b7dd8",
              linewidth=1.05, arrowsize=0.85, zorder=1)

# --- a noisy sample path: the boat carried home ---
rng = np.random.default_rng(3)
dt, pos = 0.035, np.array([-2.35, 2.15])
traj = [pos.copy()]
for _ in range(520):
    u = -a * pos[0] - w * pos[1]
    v =  w * pos[0] - a * pos[1]
    pos = pos + dt * np.array([u, v]) + np.sqrt(dt) * 0.16 * rng.standard_normal(2)
    traj.append(pos.copy())
traj = np.array(traj)
ax.plot(traj[:, 0], traj[:, 1], color="#12306e", lw=1.7, alpha=0.55, zorder=3)

# --- the goal (equilibrium): a green outline of the United Kingdom ---
draw_uk(ax, 0, 0, height=1.75)

# --- little boat character at the start of the path ---
def boat(ax, x, y, s=0.44):
    ax.add_patch(Polygon([[x-1.05*s, y], [x+1.05*s, y],
                          [x+0.62*s, y-0.55*s], [x-0.62*s, y-0.55*s]],
                 closed=True, fc="#7c3a12", ec="black", lw=1.1, zorder=8))   # hull
    ax.plot([x, x], [y, y+1.55*s], color="black", lw=1.3, zorder=8)          # mast
    ax.add_patch(Polygon([[x+0.02*s, y+1.5*s], [x+0.02*s, y+0.18*s],
                          [x+1.0*s, y+0.3*s]],
                 closed=True, fc="#dc2626", ec="black", lw=1.0, zorder=9))   # sail
boat(ax, traj[0, 0], traj[0, 1] + 0.05)

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#9db8d6")
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
