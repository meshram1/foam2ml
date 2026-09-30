"""An OpenFOAM case directory: its mesh, time directories and fields."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from .mesh import Mesh
from .parsing import FieldData, read_field


def _as_time(name: str) -> float | None:
    try:
        return float(name)
    except ValueError:
        return None


class Case:
    """A single OpenFOAM case.

    >>> case = Case("cavity")
    >>> case.mesh.n_cells
    400
    >>> U = case.field("U")          # latest time by default, shape (n_cells, 3)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "constant" / "polyMesh").is_dir():
            raise FileNotFoundError(f"{self.path} is not an OpenFOAM case (no constant/polyMesh)")

    @cached_property
    def mesh(self) -> Mesh:
        return Mesh.read(self.path)

    def times(self) -> list[str]:
        """Time directory names, sorted numerically ('0', '0.1', ...)."""
        dirs = [d.name for d in self.path.iterdir() if d.is_dir() and _as_time(d.name) is not None]
        return sorted(dirs, key=_as_time)

    @property
    def latest_time(self) -> str:
        times = self.times()
        if not times:
            raise FileNotFoundError(f"{self.path} has no time directories")
        return times[-1]

    def field_names(self, time: str | None = None) -> list[str]:
        d = self.path / (time or self.latest_time)
        return sorted(f.name for f in d.iterdir() if f.is_file() and not f.name.startswith("."))

    def read(self, name: str, time: str | None = None) -> FieldData:
        """Full field definition, including boundary patches."""
        return read_field(self.path / (time or self.latest_time) / name)

    def field(self, name: str, time: str | None = None) -> np.ndarray:
        """Cell values of a field as an array: (n_cells,) for scalars, (n_cells, 3) for vectors.

        A uniform internalField is expanded to one value per cell.
        """
        data = self.read(name, time).internal
        n = self.mesh.n_cells
        if np.isscalar(data):
            return np.full(n, data, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1 and data.size != n:     # a uniform vector such as (1 0 0)
            return np.tile(data, (n, 1))
        if len(data) != n:
            raise ValueError(f"field '{name}' has {len(data)} values for {n} cells")
        return data

    def __repr__(self) -> str:
        return f"Case({str(self.path)!r}, times={self.times()})"
