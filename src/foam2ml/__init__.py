"""foam2ml: turn OpenFOAM cases into machine-learning-ready data."""
from .case import Case
from .graph import Graph, to_graph
from .mesh import Mesh
from .parsing import FieldData, FoamFormatError, Patch

__version__ = "0.1.0.dev0"
__all__ = ["Case", "Mesh", "Graph", "to_graph", "Patch", "FieldData", "FoamFormatError", "__version__"]
