import os

import numpy as np
import pytest

from foam2ml import Case, Graph, Mesh, to_graph

AIRFOIL = "/opt/openfoam10/tutorials/incompressible/simpleFoam/airFoil2D"


def test_cavity_graph_shapes_and_names(cavity_dir):
    g = to_graph(cavity_dir, fields=["U", "p"], params={"U_lid": 1.0})
    assert g.num_nodes == 400 and g.num_edges == 2 * 760
    assert g.x_names == ["pos_x", "pos_y", "area", "on_movingWall", "on_fixedWalls"]
    assert g.y_names == ["U_x", "U_y", "p"]            # 2D: out-of-plane U_z dropped
    assert g.edge_names == ["dx", "dy", "dist", "length", "n_x", "n_y"]
    assert g.x.shape == (400, 5) and g.y.shape == (400, 3) and g.edge_attr.shape == (1520, 6)
    assert g.params == {"U_lid": 1.0}
    assert np.isfinite(g.x).all() and np.isfinite(g.y).all() and np.isfinite(g.edge_attr).all()


def test_cavity_geometry_is_exact(cavity_dir):
    """The cavity is a 20x20 grid of 0.005 m cells, 0.01 m deep — every value is known in closed form."""
    g = to_graph(cavity_dir)
    np.testing.assert_allclose(g.column("area"), 0.005 ** 2, rtol=1e-12)       # depth removed
    np.testing.assert_allclose(g.column("length"), 0.005, rtol=1e-12)          # face length, not area
    np.testing.assert_allclose(g.column("dist"), 0.005, rtol=1e-12)


def test_patch_tags(cavity_dir):
    g = to_graph(cavity_dir)
    assert g.column("on_movingWall").sum() == 20     # top row
    assert g.column("on_fixedWalls").sum() == 58     # 60 wall faces, 2 bottom corner cells counted once
    both = g.column("on_movingWall") * g.column("on_fixedWalls")
    assert both.sum() == 2                           # the two top corner cells
    by_type = to_graph(cavity_dir, patch_tags="type")
    assert by_type.x_names[-1] == "on_wall" and by_type.column("on_wall").sum() == 76
    assert "on_frontAndBack" not in g.x_names        # empty patches are never tagged


def test_edges_are_consistent(cavity_dir):
    g = to_graph(cavity_dir)
    src, dst = g.edge_index
    d = g.edge_attr[:, :2]
    np.testing.assert_allclose(d, g.pos[dst] - g.pos[src], atol=1e-15)        # displacement matches
    n = g.edge_attr[:, 4:6]
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-12)    # unit normals
    assert (np.einsum("ij,ij->i", n, d) > 0).all()                            # normal points src -> dst
    one_way = to_graph(cavity_dir, bidirectional=False)
    assert one_way.num_edges == 760


def test_mesh_only_graph_and_errors(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    g = to_graph(mesh)
    assert g.y.shape == (400, 0)
    with pytest.raises(ValueError, match="bare Mesh"):
        to_graph(mesh, fields=["U"])
    with pytest.raises(ValueError, match="patch_tags"):
        to_graph(mesh, patch_tags="bogus")


def test_save_load_roundtrip(cavity_dir, tmp_path):
    g = to_graph(Case(cavity_dir), fields=["U", "p"], time="0.5", params={"Re": 10.0})
    g.save(tmp_path / "g.npz")
    h = Graph.load(tmp_path / "g.npz")
    for a in ("pos", "x", "y", "edge_index", "edge_attr"):
        np.testing.assert_array_equal(getattr(g, a), getattr(h, a))
    assert (h.x_names, h.y_names, h.edge_names, h.params) == (g.x_names, g.y_names, g.edge_names, g.params)


@pytest.mark.skipif(not os.path.isdir(AIRFOIL), reason="OpenFOAM airFoil2D tutorial not found")
def test_airfoil_graph():
    """Curved, stretched mesh: normals must still point across every face, and nothing degenerate."""
    g = to_graph(AIRFOIL, fields=["U", "p"], time="0", params={"aoa": 0.0})
    assert g.num_nodes == 10720 and g.num_edges == 2 * 21254
    src, dst = g.edge_index
    d, n = g.edge_attr[:, :2], g.edge_attr[:, 4:6]
    assert (np.einsum("ij,ij->i", n, d) > 0).all()
    assert (g.column("dist") > 0).all() and (g.column("length") > 0).all()
    assert {"on_walls", "on_inlet", "on_outlet"} <= set(g.x_names)
    assert g.column("on_walls").sum() > 0


def test_to_pyg_and_forward_pass(cavity_dir):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv

    data = to_graph(cavity_dir, fields=["U", "p"], params={"U_lid": 1.0}).to_pyg()
    assert data.num_nodes == 400 and data.num_edges == 1520
    assert data.x.dtype == torch.float32 and data.edge_index.dtype == torch.long
    assert data.params.shape == (1, 1) and data.y_names == ["U_x", "U_y", "p"]
    data.validate(raise_on_error=True)
    out = GCNConv(data.num_node_features, data.y.shape[1])(data.x, data.edge_index)
    assert out.shape == data.y.shape
