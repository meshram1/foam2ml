import dataclasses
import os

import numpy as np
import pytest

from foam2ml import Case, GridSample, Mesh, locate_points, patch_bounds, to_grid
from foam2ml.parsing import Patch

AIRFOIL = "/opt/openfoam10/tutorials/incompressible/simpleFoam/airFoil2D"
needs_airfoil = pytest.mark.skipif(not os.path.isdir(AIRFOIL), reason="OpenFOAM airFoil2D tutorial not found")


def test_cavity_pixels_on_cell_centres(cavity_dir):
    """A 20x20 grid over the 20x20-cell cavity puts every pixel centre on a cell centre."""
    case = Case(cavity_dir)
    for method in ("cell", "linear"):
        gs = to_grid(case, fields=["U", "p"], shape=(20, 20), method=method)
        assert gs.x_names == ["mask", "sdf", "pos_x", "pos_y"] and gs.y_names == ["U_x", "U_y", "p"]
        assert gs.mask.all()
        i, j = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        np.testing.assert_array_equal(gs.cell_index, j * 20 + i)     # blockMesh numbers x fastest
        np.testing.assert_allclose(gs.channel("p"), case.field("p")[gs.cell_index], atol=1e-12)
        np.testing.assert_allclose(gs.channel("U_x"), case.field("U")[gs.cell_index, 0], atol=1e-12)


def test_cavity_sdf_is_exact(cavity_dir):
    gs = to_grid(cavity_dir, shape=(37, 23))                          # odd sizes, pixels off the cell centres
    X, Y = np.meshgrid(gs.xs, gs.ys, indexing="ij")
    np.testing.assert_allclose(gs.channel("sdf"), np.minimum.reduce([X, 0.1 - X, Y, 0.1 - Y]), atol=1e-15)
    np.testing.assert_allclose(gs.xs, (np.arange(37) + 0.5) * 0.1 / 37, rtol=1e-14)   # pixel centres
    assert to_grid(cavity_dir, shape=(8, 8), walls=[]).x_names == ["mask", "pos_x", "pos_y"]


def test_linear_reconstruction_and_sampling_are_exact_for_linear_fields(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    c = mesh.cell_centres
    gs = to_grid(mesh, {"phi": 2.0 * c[:, 0] - 5.0 * c[:, 1] + 1.0}, shape=(31, 17))
    X, Y = np.meshgrid(gs.xs, gs.ys, indexing="ij")
    np.testing.assert_allclose(gs.channel("phi"), 2 * X - 5 * Y + 1, atol=1e-13)
    pts = np.random.default_rng(0).uniform([gs.xs[0], gs.ys[0]], [gs.xs[-1], gs.ys[-1]], size=(200, 2))
    np.testing.assert_allclose(gs.sample(gs.channel("phi"), pts), 2 * pts[:, 0] - 5 * pts[:, 1] + 1, atol=1e-13)


def test_outside_bounds_are_masked(cavity_dir):
    gs = to_grid(cavity_dir, shape=(40, 40), bounds=((-0.05, 0.15), (-0.05, 0.15)))
    X, Y = np.meshgrid(gs.xs, gs.ys, indexing="ij")
    inside = (X > 0) & (X < 0.1) & (Y > 0) & (Y < 0.1)
    np.testing.assert_array_equal(gs.mask > 0, inside)
    assert (gs.channel("sdf")[~inside] < 0).all() and (gs.channel("sdf")[inside] > 0).all()


@needs_airfoil
def test_airfoil_every_cell_centre_locates_to_itself():
    mesh = Mesh.read(AIRFOIL)
    found = locate_points(mesh, mesh.cell_centres[:, :2])
    np.testing.assert_array_equal(found, np.arange(mesh.n_cells))


@needs_airfoil
def test_airfoil_grid_has_a_solid_hole_and_exact_linear_fields():
    mesh = Mesh.read(AIRFOIL)
    c = mesh.cell_centres
    gs = to_grid(mesh, {"phi": 3 * c[:, 0] - 2 * c[:, 1] + 0.5}, shape=(256, 128),
                 bounds=patch_bounds(mesh, "walls", pad=0.5))
    fluid = gs.mask > 0
    assert 0 < (~fluid).sum() < 0.2 * fluid.size                       # the airfoil is a hole, not the whole box
    assert (gs.channel("sdf")[~fluid] < 0).all() and (gs.channel("sdf")[fluid] > 0).all()
    X, Y = np.meshgrid(gs.xs, gs.ys, indexing="ij")
    np.testing.assert_allclose(gs.channel("phi")[fluid], (3 * X - 2 * Y + 0.5)[fluid], atol=1e-10)
    assert (gs.channel("phi")[~fluid] == 0).all()


def test_save_load_and_torch(cavity_dir, tmp_path):
    gs = to_grid(cavity_dir, fields=["U", "p"], shape=(16, 12), params={"Re": 10.0})
    gs.save(tmp_path / "g.npz")
    h = GridSample.load(tmp_path / "g.npz")
    for a in ("x", "y", "xs", "ys", "cell_index"):
        np.testing.assert_array_equal(getattr(gs, a), getattr(h, a))
    assert (h.x_names, h.y_names, h.params) == (gs.x_names, gs.y_names, gs.params)
    torch = pytest.importorskip("torch")
    x, y = gs.to_torch()
    assert x.shape == (4, 16, 12) and y.shape == (3, 16, 12) and x.dtype == torch.float32


def test_errors(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    with pytest.raises(ValueError, match="method"):
        to_grid(mesh, method="cubic")
    with pytest.raises(ValueError, match="dict of arrays"):
        to_grid(mesh, fields=["U"])
    as_3d = dataclasses.replace(mesh, patches=[dataclasses.replace(p, type="patch") if p.type == "empty" else p
                                               for p in mesh.patches])
    with pytest.raises(NotImplementedError, match="2D"):
        to_grid(as_3d)
