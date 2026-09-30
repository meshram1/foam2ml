# foam2ml

Turn OpenFOAM cases into machine-learning data: graphs for GNNs, grids for neural operators (FNOs).

## Install

```bash
pip install git+https://github.com/meshram1/foam2ml
```

Needs Python 3.9+ and NumPy. Add `[pyg]` for PyTorch Geometric export.

## Read a case

```python
from foam2ml import Case

case = Case("cavity")
U = case.field("U")            # (n_cells, 3), latest time
p = case.field("p", "0.1")
case.mesh.cell_centres         # matches OpenFOAM's own values
```

## Graph

```python
from foam2ml import to_graph

g = to_graph("airFoil2D", fields=["U", "p"])
data = g.to_pyg()              # torch_geometric Data
g.save("case.npz")
```

Cells are nodes, faces are edges. `x` holds position, cell size and boundary-patch tags; `y` holds the fields.
Every column is named.

## FNO grid

```python
from foam2ml import Case, patch_bounds, to_grid

case = Case("airFoil2D")
gs = to_grid(case, fields=["U", "p"], shape=(256, 128),
             bounds=patch_bounds(case.mesh, "walls", pad=0.6))
x, y = gs.to_torch()           # (4, 256, 128) inputs, (3, 256, 128) targets
```

Inputs are a fluid mask, signed distance to the walls, and coordinates. Values never cross solid bodies.
`gs.sample(prediction, points)` maps a grid prediction back onto the mesh. 2D meshes only.

## License

Apache-2.0. Not approved or endorsed by OpenCFD Ltd, owner of the OPENFOAM® trade mark.
