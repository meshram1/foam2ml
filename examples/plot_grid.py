"""Plot a foam2ml grid sample: the FNO input channels (mask, signed distance) and the resampled fields.

    python examples/plot_grid.py path/to/case --wall walls --out docs/airfoil_grid.png

Needs matplotlib. Uses only foam2ml's public API.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from foam2ml import Case, patch_bounds, to_grid

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--wall", default="walls", help="patch to crop around")
ap.add_argument("--pad", type=float, default=0.6)
ap.add_argument("--shape", type=int, nargs=2, default=[256, 128])
ap.add_argument("--out", default="grid.png")
args = ap.parse_args()

case = Case(args.case)
gs = to_grid(case, fields=["U", "p"], shape=tuple(args.shape), bounds=patch_bounds(case.mesh, args.wall, args.pad))
extent = [gs.xs[0], gs.xs[-1], gs.ys[0], gs.ys[-1]]
solid = gs.mask == 0

panels = [
    ("mask", "input: mask (1 = fluid)", "gray", None),
    ("sdf", "input: signed distance to wall", "viridis", None),
    ("p", "target: p", "RdBu_r", None),
    ("U_x", "target: U_x", "magma", None),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), constrained_layout=True)
for ax, (name, title, cmap, _) in zip(axes.ravel(), panels):
    img = np.ma.masked_where(solid & (name not in ("mask", "sdf")), gs.channel(name))
    im = ax.imshow(img.T, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    if name == "sdf":
        ax.contour(gs.xs, gs.ys, gs.channel("sdf").T, levels=[0], colors="white", linewidths=1.2)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle(f"to_grid(): {gs.x.shape[1]}x{gs.x.shape[2]} pixels, {gs.mask.mean():.1%} fluid — "
             f"channel-first arrays ready for an FNO")
fig.savefig(args.out, dpi=150)
print(f"wrote {args.out}: {gs}")
