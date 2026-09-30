"""foam2ml: turn OpenFOAM cases into machine-learning-ready data."""
from .case import Case
from .graph import Graph, to_graph
from .grid import GridSample, locate_points, patch_bounds, to_grid
from .mesh import Mesh
from .parsing import FieldData, FoamFormatError, Patch

__version__ = "0.1.0.dev0"
__all__ = ["Case", "Mesh", "Graph", "to_graph", "GridSample", "to_grid", "locate_points", "patch_bounds", "Patch", "FieldData", "FoamFormatError", "__version__"]
