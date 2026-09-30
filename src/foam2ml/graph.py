"""Graph export: cells become nodes, internal faces become edges.

The core is NumPy-only. `Graph.to_pyg()` converts to a PyTorch Geometric `Data` object and imports
torch / torch_geometric only when called.

Conventions
-----------
- `x`  : model inputs per node — geometry and boundary tags (never solution fields unless asked).
- `y`  : model targets per node — the requested solution fields.
- `pos`: cell centres.
- Edges are directed and, by default, include both directions of every internal face.
- For OpenFOAM's one-cell-thick 2D meshes the out-of-plane axis is dropped everywhere, and face areas
  are divided by the mesh depth so they become true 2D face lengths.
Every column has a name in `x_names`, `y_names` and `edge_names`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .case import Case
from .mesh import Mesh

_AXES = "xyz"


@dataclass
class Graph:
    pos: np.ndarray            # (N, D)
    x: np.ndarray              # (N, Fx)
    y: np.ndarray              # (N, Fy)
    edge_index: np.ndarray     # (2, E) int64, row 0 = source, row 1 = target
    edge_attr: np.ndarray      # (E, Fe)
    x_names: list[str]
    y_names: list[str]
    edge_names: list[str]
    params: dict[str, float] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.pos)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def column(self, name: str) -> np.ndarray:
        """Look up any named column of x, y or edge_attr."""
        for names, arr in ((self.x_names, self.x), (self.y_names, self.y), (self.edge_names, self.edge_attr)):
            if name in names:
                return arr[:, names.index(name)]
        raise KeyError(f"no column '{name}'")

    def to_pyg(self, dtype=None):
        """A torch_geometric.data.Data with x, y, pos, edge_index, edge_attr, params and the column names."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:  # pragma: no cover - exercised only without the extra installed
            raise ImportError("to_pyg() needs PyTorch Geometric: pip install 'foam2ml[pyg]'") from e
        dtype = dtype or torch.float32
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)  # noqa: E731
        return Data(
            x=t(self.x), y=t(self.y), pos=t(self.pos),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_attr=t(self.edge_attr),
            params=t(np.array([list(self.params.values())], dtype=np.float64)).reshape(1, len(self.params)),
            x_names=list(self.x_names), y_names=list(self.y_names), edge_names=list(self.edge_names),
            param_names=list(self.params),
        )

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz that loads without foam2ml's dependencies (just NumPy)."""
        np.savez_compressed(
            path, pos=self.pos, x=self.x, y=self.y, edge_index=self.edge_index, edge_attr=self.edge_attr,
            x_names=np.array(self.x_names), y_names=np.array(self.y_names),
            edge_names=np.array(self.edge_names),
            param_names=np.array(list(self.params)), param_values=np.array(list(self.params.values()), dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        d = np.load(path, allow_pickle=False)
        return cls(pos=d["pos"], x=d["x"], y=d["y"], edge_index=d["edge_index"], edge_attr=d["edge_attr"],
                   x_names=d["x_names"].tolist(), y_names=d["y_names"].tolist(),
                   edge_names=d["edge_names"].tolist(),
                   params=dict(zip(d["param_names"].tolist(), d["param_values"].tolist())))

    def __repr__(self) -> str:
        return (f"Graph(nodes={self.num_nodes}, edges={self.num_edges}, x={self.x_names}, y={self.y_names}, "
                f"edge_attr={self.edge_names}, params={self.params})")


def _field_columns(name: str, values: np.ndarray, axes: list[int]) -> tuple[np.ndarray, list[str]]:
    if values.ndim == 1:
        return values[:, None], [name]
    if values.shape[1] == 3:
        return values[:, axes], [f"{name}_{_AXES[a]}" for a in axes]
    return values, [f"{name}_{i}" for i in range(values.shape[1])]


def to_graph(
    source: Case | Mesh | str | Path,
    fields: list[str] | tuple[str, ...] = (),
    time: str | None = None,
    *,
    params: dict[str, float] | None = None,
    patch_tags: str = "name",
    include_volume: bool = True,
    bidirectional: bool = True,
) -> Graph:
    """Build a graph from an OpenFOAM case (or just its mesh).

    Parameters
    ----------
    source        : a Case, a Mesh, or a path to a case directory.
    fields        : solution fields to put in `y`, e.g. ("U", "p"). Needs a Case/path, not a bare Mesh.
    time          : time directory to read fields from; the latest by default.
    params        : case-level parameters (e.g. {"aoa": 4.0, "U_inf": 25.0}), stored on the graph.
    patch_tags    : "name" -> one 0/1 input column per boundary patch name (cells touching that patch);
                    "type" -> one column per patch type (wall, patch, symmetryPlane, ...); "none" -> no tags.
                    'empty' patches (the 2D front/back planes) are never tagged.
    include_volume: add the cell volume (cell area in 2D) as an input column.
    bidirectional : include both directions of each internal face (standard for message passing).
    """
    if isinstance(source, (str, Path)):
        source = Case(source)
    case = source if isinstance(source, Case) else None
    mesh = source.mesh if case is not None else source
    if fields and case is None:
        raise ValueError("fields need a Case or case path; a bare Mesh has no solution data")

    normal = mesh.normal_axis
    axes = [a for a in range(3) if a != normal]
    depth = np.ptp(mesh.points[:, normal]) if normal is not None else 1.0

    # --- nodes ------------------------------------------------------------------------------
    pos = mesh.cell_centres[:, axes]
    x_cols, x_names = [pos], [f"pos_{_AXES[a]}" for a in axes]
    if include_volume:
        x_cols.append((mesh.cell_volumes / depth)[:, None])
        x_names.append("area" if normal is not None else "volume")

    if patch_tags not in ("name", "type", "none"):
        raise ValueError("patch_tags must be 'name', 'type' or 'none'")
    if patch_tags != "none":
        groups: dict[str, list] = {}
        for p in mesh.patches:
            if p.type != "empty":
                groups.setdefault(p.name if patch_tags == "name" else p.type, []).append(p)
        for key, patches in groups.items():
            tag = np.zeros(mesh.n_cells)
            for p in patches:
                tag[mesh.owner[p.face_slice]] = 1.0
            x_cols.append(tag[:, None])
            x_names.append(f"on_{key}")

    y_cols, y_names = [], []
    for name in fields:
        cols, names = _field_columns(name, case.field(name, time), axes)
        y_cols.append(cols)
        y_names += names

    # --- edges ------------------------------------------------------------------------------
    n_int = mesh.n_internal_faces
    src, dst = mesh.owner[:n_int], mesh.neighbour
    d = pos[dst] - pos[src]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    area_vec = mesh.face_areas[:n_int][:, axes]            # in 2D the in-plane part carries the whole area
    area = np.linalg.norm(mesh.face_areas[:n_int], axis=1, keepdims=True) / depth
    unit_n = area_vec / np.maximum(np.linalg.norm(area_vec, axis=1, keepdims=True), 1e-300)

    edge_attr = np.hstack([d, dist, area, unit_n])
    edge_names = ([f"d{_AXES[a]}" for a in axes] + ["dist", "length" if normal is not None else "area"]
                  + [f"n_{_AXES[a]}" for a in axes])
    edge_index = np.stack([src, dst])
    if bidirectional:
        k = len(axes)
        flipped = edge_attr.copy()
        flipped[:, :k] *= -1          # displacement reverses
        flipped[:, k + 2:] *= -1      # so does the face normal; distance and area don't
        edge_attr = np.vstack([edge_attr, flipped])
        edge_index = np.hstack([edge_index, edge_index[::-1]])

    return Graph(
        pos=pos,
        x=np.hstack(x_cols),
        y=np.hstack(y_cols) if y_cols else np.empty((mesh.n_cells, 0)),
        edge_index=edge_index.astype(np.int64),
        edge_attr=edge_attr,
        x_names=x_names, y_names=y_names, edge_names=edge_names,
        params=dict(params or {}),
    )
