"""OpenFOAM polyMesh: topology plus the geometry OpenFOAM itself computes.

Face centres/areas and cell centres/volumes follow OpenFOAM's primitiveMesh algorithms
(triangle decomposition of faces about their point average; pyramid decomposition of cells
about the average of their face centres), so results agree with OpenFOAM to rounding error.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .parsing import FoamFormatError, Patch, read_boundary, read_faces, read_labels, read_points

_ROOT_VSMALL = 1e-150
_VSMALL = 1e-300


@dataclass(eq=False)
class Mesh:
    """An OpenFOAM polyMesh.

    Faces are stored in CSR form: face i uses points face_points[face_offsets[i]:face_offsets[i+1]].
    Faces 0 .. n_internal_faces-1 are internal (owner < neighbour); the rest belong to patches.
    """

    points: np.ndarray        # (n_points, 3)
    face_offsets: np.ndarray  # (n_faces + 1,)
    face_points: np.ndarray   # (sum of face sizes,)
    owner: np.ndarray         # (n_faces,)
    neighbour: np.ndarray     # (n_internal_faces,)
    patches: list[Patch]

    @classmethod
    def read(cls, case_dir: str | Path) -> "Mesh":
        """Read constant/polyMesh from an OpenFOAM case directory."""
        poly = Path(case_dir) / "constant" / "polyMesh"
        if not poly.is_dir():
            raise FileNotFoundError(f"no constant/polyMesh in {case_dir}")
        offsets, ids = read_faces(poly / "faces")
        mesh = cls(
            points=read_points(poly / "points"),
            face_offsets=offsets,
            face_points=ids,
            owner=read_labels(poly / "owner"),
            neighbour=read_labels(poly / "neighbour"),
            patches=read_boundary(poly / "boundary"),
        )
        mesh.validate()
        return mesh

    # ---- sizes -------------------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_faces(self) -> int:
        return len(self.face_offsets) - 1

    @property
    def n_internal_faces(self) -> int:
        return len(self.neighbour)

    @cached_property
    def n_cells(self) -> int:
        top = self.owner.max(initial=-1)
        if self.neighbour.size:
            top = max(top, self.neighbour.max())
        return int(top) + 1

    def validate(self) -> None:
        """Cheap consistency checks; raises FoamFormatError on a malformed mesh."""
        if len(self.owner) != self.n_faces:
            raise FoamFormatError(f"owner has {len(self.owner)} entries for {self.n_faces} faces")
        if self.n_internal_faces > self.n_faces:
            raise FoamFormatError("more neighbours than faces")
        if self.face_points.size and self.face_points.max() >= self.n_points:
            raise FoamFormatError("a face references a point index beyond the points list")
        covered = sum(p.n_faces for p in self.patches)
        if covered != self.n_faces - self.n_internal_faces:
            raise FoamFormatError(
                f"patches cover {covered} faces but there are {self.n_faces - self.n_internal_faces} boundary faces")

    # ---- boundary ----------------------------------------------------------------------------
    def patch(self, name: str) -> Patch:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(f"no patch '{name}'; patches are {[p.name for p in self.patches]}")

    def patch_cells(self, name: str) -> np.ndarray:
        """Cells adjacent to a patch, one entry per patch face (a cell can repeat)."""
        return self.owner[self.patch(name).face_slice]

    @property
    def empty_patches(self) -> list[Patch]:
        return [p for p in self.patches if p.type == "empty"]

    @property
    def is_2d(self) -> bool:
        """True for OpenFOAM's one-cell-thick 2D meshes, which mark the out-of-plane faces 'empty'."""
        return bool(self.empty_patches)

    @cached_property
    def normal_axis(self) -> int | None:
        """For a 2D mesh, the axis (0=x, 1=y, 2=z) normal to the empty patches; None for 3D."""
        if not self.is_2d:
            return None
        p = self.empty_patches[0]
        # front and back faces point in opposite directions, so sum magnitudes, not vectors
        return int(np.argmax(np.abs(self.face_areas[p.face_slice]).sum(axis=0)))

    # ---- connectivity ------------------------------------------------------------------------
    def cell_adjacency(self) -> np.ndarray:
        """(2, n_internal_faces) array: each internal face joins cells owner[i] and neighbour[i]."""
        return np.stack([self.owner[: self.n_internal_faces], self.neighbour])

    # ---- geometry (OpenFOAM primitiveMesh algorithms) -----------------------------------------
    def _face_groups(self):
        """Yield (face_indices, point_ids of shape (m, k)) for each distinct face size k."""
        sizes = np.diff(self.face_offsets)
        for k in np.unique(sizes):
            faces = np.nonzero(sizes == k)[0]
            idx = self.face_offsets[faces][:, None] + np.arange(k)
            yield faces, self.face_points[idx]

    @cached_property
    def _face_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        centres = np.empty((self.n_faces, 3))
        areas = np.empty((self.n_faces, 3))
        for faces, ids in self._face_groups():
            p = self.points[ids]                      # (m, k, 3)
            k = ids.shape[1]
            if k == 3:
                centres[faces] = p.mean(axis=1)
                areas[faces] = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                continue
            f_est = p.mean(axis=1, keepdims=True)     # point-average estimate, (m, 1, 3)
            nxt = np.roll(p, -1, axis=1)
            n = np.cross(nxt - p, f_est - p)          # 2x triangle area vectors, (m, k, 3)
            a = np.linalg.norm(n, axis=2)             # (m, k)
            c = p + nxt + f_est                       # 3x triangle centroids
            sum_a = a.sum(axis=1)
            ok = sum_a >= _ROOT_VSMALL
            centres[faces] = np.where(ok[:, None], (a[..., None] * c).sum(axis=1) / (3 * np.maximum(sum_a, _ROOT_VSMALL))[:, None],
                                      f_est[:, 0])
            areas[faces] = np.where(ok[:, None], 0.5 * n.sum(axis=1), 0.0)
        return centres, areas

    @property
    def face_centres(self) -> np.ndarray:
        return self._face_geometry[0]

    @property
    def face_areas(self) -> np.ndarray:
        """Face area vectors (normal times area), pointing out of the owner cell."""
        return self._face_geometry[1]

    @cached_property
    def _cell_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        fc, fa = self._face_geometry
        own, nei = self.owner, self.neighbour
        n_int = self.n_internal_faces

        c_est = np.zeros((self.n_cells, 3))
        n_cell_faces = np.zeros(self.n_cells)
        np.add.at(c_est, own, fc)
        np.add.at(n_cell_faces, own, 1)
        np.add.at(c_est, nei, fc[:n_int])
        np.add.at(n_cell_faces, nei, 1)
        c_est /= n_cell_faces[:, None]

        ctrs = np.zeros((self.n_cells, 3))
        vols = np.zeros(self.n_cells)
        pyr = np.einsum("ij,ij->i", fa, fc - c_est[own])            # 3x pyramid volumes, owner side
        np.add.at(ctrs, own, pyr[:, None] * (0.75 * fc + 0.25 * c_est[own]))
        np.add.at(vols, own, pyr)
        pyr = np.einsum("ij,ij->i", fa[:n_int], c_est[nei] - fc[:n_int])  # neighbour side
        np.add.at(ctrs, nei, pyr[:, None] * (0.75 * fc[:n_int] + 0.25 * c_est[nei]))
        np.add.at(vols, nei, pyr)

        ok = np.abs(vols) > _VSMALL
        ctrs = np.where(ok[:, None], ctrs / np.where(ok, vols, 1.0)[:, None], c_est)
        return ctrs, vols / 3.0

    @property
    def cell_centres(self) -> np.ndarray:
        return self._cell_geometry[0]

    @property
    def cell_volumes(self) -> np.ndarray:
        return self._cell_geometry[1]

    def __repr__(self) -> str:
        kind = f"2D (normal axis {'xyz'[self.normal_axis]})" if self.is_2d else "3D"
        return (f"Mesh({kind}: {self.n_cells} cells, {self.n_faces} faces "
                f"({self.n_internal_faces} internal), {self.n_points} points, "
                f"patches={[p.name for p in self.patches]})")
