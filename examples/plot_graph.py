"""Plot a foam2ml graph: mesh edges near the walls, and a solution field on the nodes.

    python examples/plot_graph.py path/to/case --field p --out docs/airfoil_graph.png

Needs matplotlib. Uses only foam2ml's public API.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from foam2ml import to_graph

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--field", default="p")
ap.add_argument("--wall", default="walls", help="patch to zoom in on")
ap.add_argument("--out", default="graph.png")
args = ap.parse_args()

g = to_graph(args.case, fields=[args.field])
pos, (src, dst) = g.pos, g.edge_index
wall = g.column(f"on_{args.wall}") > 0
lo, hi = pos[wall].min(axis=0), pos[wall].max(axis=0)
chord = (hi - lo).max()
centre = (lo + hi) / 2

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

ax = axes[0]
wall_pos = pos[wall]
nose = wall_pos[np.argmin(wall_pos[:, 0])]                # upstream-most wall cell = leading edge
zoom_c = nose + np.array([0.06 * chord, 0.0])
half = 0.1 * chord * np.array([1.0, 0.6])
box = (np.abs(pos - zoom_c) < half).all(axis=1)
keep = box[src] & box[dst] & (src < dst)                  # each face once, inside the zoom box
ax.add_collection(LineCollection(np.stack([pos[src[keep]], pos[dst[keep]]], axis=1), lw=0.35, color="0.35"))
ax.scatter(*pos[box & ~wall].T, s=1.2, color="tab:blue", zorder=2, label="cells (nodes)")
ax.scatter(*pos[box & wall].T, s=5, color="tab:red", zorder=3, label=f"cells tagged on_{args.wall}")
ax.set_xlim(zoom_c[0] - half[0], zoom_c[0] + half[0])
ax.set_ylim(zoom_c[1] - half[1], zoom_c[1] + half[1])
ax.set_aspect("equal")
ax.set_title(f"Graph near the leading edge: {g.num_nodes:,} nodes, {g.num_edges:,} directed edges")
ax.legend(loc="lower right", markerscale=4, fontsize=8)

ax = axes[1]
view = (np.abs(pos - centre) < 1.2 * chord).all(axis=1)
sc = ax.scatter(*pos[view].T, c=g.column(args.field)[view], s=2, cmap="RdBu_r")
fig.colorbar(sc, ax=ax, label=args.field)
ax.set_aspect("equal")
ax.set_title(f"'{args.field}' on the graph nodes (converged simpleFoam)")

for a in axes:
    a.set_xticks([]), a.set_yticks([])
fig.savefig(args.out, dpi=160)
print(f"wrote {args.out}: {g}")
