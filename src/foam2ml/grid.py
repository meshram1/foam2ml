"""Grid export for neural operators (FNO and friends): resample a case onto a regular grid.

Each grid point is located inside the actual mesh cell that contains it, so values are never
interpolated across a solid body. Points in no cell get mask = 0. Within a cell, values use
finite-volume linear reconstruction: phi(p) = phi_c + grad_c . (p - C_c), with grad_c a least-squares
gradient over the cell's face neighbours only. That reconstruction is exact for linear fields.

Arrays are channel-first with "ij" indexing: x[c, i, j] is channel c at (xs[i], ys[j]).
To plot with matplotlib's imshow, transpose: imshow(x[c].T, origin="lower").
Grid points are pixel centres: xs[i] = x_min + (i + 0.5) * dx.

2D (one-cell-thick, 'empty'-patch) meshes only for now.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .case import Case
from .mesh import Mesh

_AXES = "xyz"
_CHUNK = 4_000_000  # max candidate (triangle, point) pairs held in memory at once


# ---------------------------------------------------------------------------------------------
# 2D geometry helpers
# ---------------------------------------------------------------------------------------------

def _plane(mesh: Mesh) -> tuple[list[int], int]:
    if not mesh.is_2d:
        raise NotImplementedError("grid export currently supports 2D meshes (with 'empty' patches) only")
    normal = mesh.normal_axis
    return [a for a in range(3) if a != normal], normal


def _cell_triangles(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate each cell's in-plane polygon (its face on one 'empty' plane).

    Returns (triangles (T, 3, 2), cell id per triangle (T,)).
    """
    axes, normal = _plane(mesh)
    empty = np.concatenate([np.arange(p.start_face, p.start_face + p.n_faces) for p in mesh.empty_patches])
    mid = mesh.points[:, normal].min() + 0.5 * np.ptp(mesh.points[:, normal])
    front = empty[mesh.face_centres[empty, normal] < mid]
    if len(front) != mesh.n_cells or len(np.unique(mesh.owner[front])) != mesh.n_cells:
        raise ValueError("could not find exactly one front face per cell; is this a one-cell-thick 2D mesh?")
    tris, cells = [], []
    sizes = np.diff(mesh.face_offsets)[front]
    for k in np.unique(sizes):
        faces = front[sizes == k]
        ids = mesh.face_points[mesh.face_offsets[faces][:, None] + np.arange(k)]
        pts = mesh.points[ids][:, :, axes]                     # (m, k, 2)
        for t in range(1, k - 1):                                # fan from vertex 0
            tris.append(np.stack([pts[:, 0], pts[:, t], pts[:, t + 1]], axis=1))
            cells.append(mesh.owner[faces])
    return np.concatenate(tris), np.concatenate(cells)


def _expand(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For groups (start, count) return (group id, start + offset) for every member, flattened."""
    group = np.repeat(np.arange(len(counts)), counts)
    offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return group, starts[group] + offset


def locate_points(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    """Index of the cell containing each in-plane point, or -1 if it lies in no cell (solid or outside).

    Works on the cells' true polygons (via a triangle-bucket search), so it is exact for the mesh as
    stored, including stretched cells and holes such as an airfoil.
    """
    tris, tri_cell = _cell_triangles(mesh)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    found = np.full(len(points), -1, dtype=np.int64)
    if not len(points):
        return found

    # bucket the query points on a uniform grid of about 4 points per bucket
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-300)
    h = max(np.sqrt(span[0] * span[1] * 4.0 / len(points)), span.max() / 4096, 1e-300)
    nb = np.maximum(np.ceil(span / h).astype(np.int64), 1)
    b = np.minimum(((points - lo) / h).astype(np.int64), nb - 1)
    bucket = b[:, 0] * nb[1] + b[:, 1]
    order = np.argsort(bucket, kind="stable")
    bucket_start = np.searchsorted(bucket[order], np.arange(nb[0] * nb[1] + 1))

    # bucket range covered by each triangle's bounding box
    t_lo = ((tris.min(axis=1) - lo) / h).astype(np.float64)
    t_hi = ((tris.max(axis=1) - lo) / h).astype(np.float64)
    keep = (t_hi >= 0).all(axis=1) & (t_lo[:, 0] < nb[0]) & (t_lo[:, 1] < nb[1])
    tris, tri_cell, t_lo, t_hi = tris[keep], tri_cell[keep], t_lo[keep], t_hi[keep]
    b0 = np.clip(np.floor(t_lo), 0, nb - 1).astype(np.int64)
    b1 = np.clip(np.floor(t_hi), 0, nb - 1).astype(np.int64)
    wx, wy = b1[:, 0] - b0[:, 0] + 1, b1[:, 1] - b0[:, 1] + 1

    # process triangles in chunks bounded by their candidate-pair count
    per_tri_pairs = wx * wy
    running = np.concatenate([[0], np.cumsum(per_tri_pairs)])
    s = 0
    while s < len(tris):
        e = int(np.searchsorted(running, running[s] + _CHUNK, side="right")) - 1
        e = min(max(e, s + 1), len(tris))          # at least one triangle per chunk
        t_idx, k = _expand(np.zeros(e - s, dtype=np.int64), per_tri_pairs[s:e])
        s_chunk, s = s, e
        t_idx += s_chunk
        bx = b0[t_idx, 0] + k % wx[t_idx]
        by = b0[t_idx, 1] + k // wx[t_idx]
        bid = bx * nb[1] + by
        cnt = bucket_start[bid + 1] - bucket_start[bid]
        pair, slot = _expand(bucket_start[bid], cnt)
        tri_of = t_idx[pair]
        pt_of = order[slot]
        undecided = found[pt_of] < 0
        tri_of, pt_of = tri_of[undecided], pt_of[undecided]
        a, bb, c = tris[tri_of, 0], tris[tri_of, 1], tris[tri_of, 2]
        p = points[pt_of]
        cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]  # noqa: E731
        d1, d2, d3 = cross(bb - a, p - a), cross(c - bb, p - bb), cross(a - c, p - c)
        scale = np.abs(cross(bb - a, c - a))
        eps = 1e-12 * scale
        inside = ((d1 >= -eps) & (d2 >= -eps) & (d3 >= -eps)) | ((d1 <= eps) & (d2 <= eps) & (d3 <= eps))
        hit_pts, first = np.unique(pt_of[inside], return_index=True)
        found[hit_pts] = tri_cell[tri_of[inside][first]]
    return found


def least_squares_gradient(mesh: Mesh, values: np.ndarray) -> np.ndarray:
    """In-plane least-squares gradient of a cell field over face neighbours, weights 1/|d|^2.

    values: (n_cells,) or (n_cells, C). Returns (n_cells, 2) or (n_cells, C, 2).
    Cells whose neighbours don't span the plane get a zero gradient.
    """
    axes, _ = _plane(mesh)
    c = mesh.cell_centres[:, axes]
    v = np.asarray(values, dtype=np.float64)
    scalar = v.ndim == 1
    v = v[:, None] if scalar else v
    own, nei = mesh.owner[: mesh.n_internal_faces], mesh.neighbour
    d = c[nei] - c[own]
    w = 1.0 / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
    dd = w[:, None, None] * d[:, :, None] * d[:, None, :]           # (F, 2, 2)
    dv = v[nei] - v[own]                                            # (F, C)
    rhs_f = w[:, None, None] * d[:, :, None] * dv[:, None, :]       # (F, 2, C)
    A = np.zeros((mesh.n_cells, 2, 2))
    rhs = np.zeros((mesh.n_cells, 2, v.shape[1]))
    np.add.at(A, own, dd)
    np.add.at(A, nei, dd)
    np.add.at(rhs, own, rhs_f)
    np.add.at(rhs, nei, rhs_f)                                      # (-d)(-dv) = d dv
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    ok = np.abs(det) > 1e-12 * np.maximum(A[:, 0, 0] * A[:, 1, 1], 1e-300)
    grad = np.zeros((mesh.n_cells, v.shape[1], 2))
    grad[ok] = np.linalg.solve(A[ok][:, None], rhs[ok].transpose(0, 2, 1)[..., None])[..., 0]
    return grad[:, 0] if scalar else grad


def _wall_segments(mesh: Mesh, walls: list[str]) -> np.ndarray:
    """In-plane segments (S, 2, 2) of the given patches' faces (each 2D face projects to a segment)."""
    axes, _ = _plane(mesh)
    segs = []
    for name in walls:
        p = mesh.patch(name)
        for f in range(p.start_face, p.start_face + p.n_faces):
            pts = mesh.points[mesh.face_points[mesh.face_offsets[f]:mesh.face_offsets[f + 1]]][:, axes]
            far = np.argmax(np.linalg.norm(pts - pts[0], axis=1))
            segs.append((pts[0], pts[far]))
    return np.array(segs, dtype=np.float64).reshape(-1, 2, 2)


def distance_to_segments(points: np.ndarray, segs: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Exact unsigned distance from each point to the nearest segment."""
    out = np.full(len(points), np.inf)
    if not len(segs):
        return out
    a, ab = segs[:, 0], segs[:, 1] - segs[:, 0]
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    step = max(1, chunk // len(segs))
    for s in range(0, len(points), step):
        p = points[s:s + step, None, :]                                    # (n, 1, 2)
        t = np.clip(np.einsum("nsk,sk->ns", p - a, ab) / ab2, 0.0, 1.0)
        d = p - (a + t[..., None] * ab)
        out[s:s + step] = np.sqrt(np.einsum("nsk,nsk->ns", d, d).min(axis=1))
    return out


def patch_bounds(mesh: Mesh, patches: str | list[str], pad: float = 0.5):
    """A bounding box around some patches, padded by `pad` times their largest in-plane extent.

    Handy for cropping an external-flow domain around a body: bounds=patch_bounds(mesh, "walls", pad=0.5).
    """
    axes, _ = _plane(mesh)
    names = [patches] if isinstance(patches, str) else patches
    segs = _wall_segments(mesh, names).reshape(-1, 2)
    lo, hi = segs.min(axis=0), segs.max(axis=0)
    margin = pad * (hi - lo).max()
    return tuple((float(lo[i] - margin), float(hi[i] + margin)) for i in range(2))


# ---------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------

@dataclass
class GridSample:
    x: np.ndarray                  # (C_in, nx, ny) inputs: mask, sdf (if walls), pos_x, pos_y
    y: np.ndarray                  # (C_out, nx, ny) fields, zero where mask == 0
    xs: np.ndarray                 # (nx,) pixel-centre coordinates along the first in-plane axis
    ys: np.ndarray                 # (ny,)
    cell_index: np.ndarray         # (nx, ny) containing cell, -1 outside the fluid
    x_names: list[str]
    y_names: list[str]
    params: dict[str, float] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return self.x[self.x_names.index("mask")]

    def channel(self, name: str) -> np.ndarray:
        for names, arr in ((self.x_names, self.x), (self.y_names, self.y)):
            if name in names:
                return arr[names.index(name)]
        raise KeyError(f"no channel '{name}'")

    def sample(self, values: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Bilinearly sample grid values (nx, ny) or (C, nx, ny) at in-plane points (N, 2).

        Use it to evaluate a model's grid prediction back on the mesh cell centres. Points outside the
        grid's pixel-centre range are clamped to the edge.
        """
        v = np.asarray(values)
        v = v[None] if v.ndim == 2 else v
        pts = np.asarray(points, dtype=np.float64)
        dx, dy = self.xs[1] - self.xs[0], self.ys[1] - self.ys[0]
        fi = np.clip((pts[:, 0] - self.xs[0]) / dx, 0, len(self.xs) - 1)
        fj = np.clip((pts[:, 1] - self.ys[0]) / dy, 0, len(self.ys) - 1)
        i0 = np.minimum(fi.astype(np.int64), len(self.xs) - 2)
        j0 = np.minimum(fj.astype(np.int64), len(self.ys) - 2)
        ti, tj = fi - i0, fj - j0
        out = (v[:, i0, j0] * (1 - ti) * (1 - tj) + v[:, i0 + 1, j0] * ti * (1 - tj)
               + v[:, i0, j0 + 1] * (1 - ti) * tj + v[:, i0 + 1, j0 + 1] * ti * tj)
        return out[0] if np.asarray(values).ndim == 2 else out.T

    def to_torch(self, dtype=None):
        """(x, y) as torch tensors, channel-first, ready for an FNO. Add a batch dim with x[None]."""
        import torch
        dtype = dtype or torch.float32
        return torch.as_tensor(self.x, dtype=dtype), torch.as_tensor(self.y, dtype=dtype)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, x=self.x, y=self.y, xs=self.xs, ys=self.ys, cell_index=self.cell_index,
                            x_names=np.array(self.x_names), y_names=np.array(self.y_names),
                            param_names=np.array(list(self.params)),
                            param_values=np.array(list(self.params.values()), dtype=float))

    @classmethod
    def load(cls, path: str | Path) -> "GridSample":
        d = np.load(path, allow_pickle=False)
        return cls(x=d["x"], y=d["y"], xs=d["xs"], ys=d["ys"], cell_index=d["cell_index"],
                   x_names=d["x_names"].tolist(), y_names=d["y_names"].tolist(),
                   params=dict(zip(d["param_names"].tolist(), d["param_values"].tolist())))

    def __repr__(self) -> str:
        return (f"GridSample(shape={self.x.shape[1:]}, fluid={self.mask.mean():.1%}, x={self.x_names}, "
                f"y={self.y_names}, params={self.params})")


def to_grid(
    source: Case | Mesh | str | Path,
    fields: list[str] | tuple[str, ...] | dict[str, np.ndarray] = (),
    time: str | None = None,
    *,
    shape: tuple[int, int] = (128, 128),
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    method: str = "linear",
    walls: list[str] | None = None,
    params: dict[str, float] | None = None,
) -> GridSample:
    """Resample a 2D case onto a regular grid for neural operators.

    Parameters
    ----------
    fields : field names to read from the case, or a dict {name: cell array} for your own cell data.
    shape  : (nx, ny) number of pixels along the two in-plane axes.
    bounds : ((min, max), (min, max)) region to sample; the whole mesh by default.
             For external flows use patch_bounds(mesh, "walls", pad=...).
    method : "linear" (finite-volume linear reconstruction, default) or "cell" (piecewise constant).
    walls  : patches the signed-distance channel measures to; all 'wall'-type patches by default.
             Pass [] to omit the sdf channel.
    """
    if isinstance(source, (str, Path)):
        source = Case(source)
    case = source if isinstance(source, Case) else None
    mesh = source.mesh if case is not None else source
    if method not in ("linear", "cell"):
        raise ValueError("method must be 'linear' or 'cell'")
    axes, _ = _plane(mesh)

    if bounds is None:
        pts2 = mesh.points[:, axes]
        bounds = tuple((float(pts2[:, i].min()), float(pts2[:, i].max())) for i in range(2))
    (x0, x1), (y0, y1) = bounds
    nx, ny = shape
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)

    cell = locate_points(mesh, P)
    inside = cell >= 0
    mask = inside.astype(np.float64)

    x_cols, x_names = [mask], ["mask"]
    wall_names = [p.name for p in mesh.patches if p.type == "wall"] if walls is None else list(walls)
    if wall_names:
        dist = distance_to_segments(P, _wall_segments(mesh, wall_names))
        x_cols.append(np.where(inside, dist, -dist))
        x_names.append("sdf")
    x_cols += [P[:, 0], P[:, 1]]
    x_names += [f"pos_{_AXES[a]}" for a in axes]

    if isinstance(fields, dict):
        named = {k: np.asarray(v, dtype=np.float64) for k, v in fields.items()}
    else:
        if fields and case is None:
            raise ValueError("field names need a Case or case path; pass a dict of arrays for a bare Mesh")
        named = {n: case.field(n, time) for n in fields}

    centres = mesh.cell_centres[:, axes]
    y_cols, y_names = [], []
    for name, vals in named.items():
        if len(vals) != mesh.n_cells:
            raise ValueError(f"field '{name}' has {len(vals)} values for {mesh.n_cells} cells")
        if vals.ndim == 2 and vals.shape[1] == 3:
            vals, names = vals[:, axes], [f"{name}_{_AXES[a]}" for a in axes]
        elif vals.ndim == 2:
            names = [f"{name}_{i}" for i in range(vals.shape[1])]
        else:
            names = [name]
        v2 = vals.reshape(mesh.n_cells, -1)
        out = np.zeros((len(P), v2.shape[1]))
        c = cell[inside]
        out[inside] = v2[c]
        if method == "linear":
            grad = least_squares_gradient(mesh, v2)                       # (n_cells, C, 2)
            out[inside] += np.einsum("pck,pk->pc", grad[c], P[inside] - centres[c])
        y_cols += list(out.T)
        y_names += names

    to_img = lambda a: a.reshape(nx, ny)  # noqa: E731
    return GridSample(
        x=np.stack([to_img(a) for a in x_cols]),
        y=np.stack([to_img(a) for a in y_cols]) if y_cols else np.empty((0, nx, ny)),
        xs=xs, ys=ys, cell_index=cell.reshape(nx, ny),
        x_names=x_names, y_names=y_names, params=dict(params or {}),
    )
