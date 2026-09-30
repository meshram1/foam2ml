import shutil
import subprocess

import numpy as np
import pytest

from foam2ml import Case, Mesh
from foam2ml.parsing import read_field

from .conftest import openfoam_available


def test_cavity_sizes(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    # from the polyMesh header note: nPoints: 882 nCells: 400 nFaces: 1640 nInternalFaces: 760
    assert (mesh.n_points, mesh.n_cells, mesh.n_faces, mesh.n_internal_faces) == (882, 400, 1640, 760)
    assert mesh.is_2d and mesh.normal_axis == 2


def test_geometry_matches_openfoam(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    C = read_field(cavity_dir / "0.5" / "C").internal
    V = read_field(cavity_dir / "0.5" / "V").internal
    np.testing.assert_allclose(mesh.cell_centres, C, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mesh.cell_volumes, V, rtol=1e-12)
    assert mesh.cell_volumes.sum() == pytest.approx(0.1 * 0.1 * 0.01, rel=1e-12)


def test_cells_are_closed(cavity_dir):
    """Divergence theorem: each cell's outward face-area vectors sum to zero."""
    mesh = Mesh.read(cavity_dir)
    closure = np.zeros((mesh.n_cells, 3))
    np.add.at(closure, mesh.owner, mesh.face_areas)
    np.add.at(closure, mesh.neighbour, -mesh.face_areas[: mesh.n_internal_faces])
    assert np.abs(closure).max() < 1e-18


def test_adjacency(cavity_dir):
    mesh = Mesh.read(cavity_dir)
    edges = mesh.cell_adjacency()
    assert edges.shape == (2, 760)
    assert (edges[0] < edges[1]).all()                      # OpenFOAM's owner < neighbour ordering
    degree = np.bincount(edges.ravel(), minlength=mesh.n_cells)
    assert degree.min() == 2 and degree.max() == 4          # corner cells have 2 in-plane neighbours


def test_case_fields(cavity_dir):
    case = Case(cavity_dir)
    assert case.times() == ["0", "0.5"] and case.latest_time == "0.5"
    assert case.field("U").shape == (400, 3)
    assert case.field("p").shape == (400,)
    np.testing.assert_array_equal(case.field("U", "0"), np.zeros((400, 3)))   # uniform expanded
    assert set(case.mesh.patch_cells("movingWall")) == set(range(380, 400))


# ---- live comparison against OpenFOAM on harder meshes (skipped without OpenFOAM) -----------

TUTORIALS = "/opt/openfoam10/tutorials/incompressible/simpleFoam"


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM not installed")
@pytest.mark.parametrize("tutorial,needs_blockmesh", [("pitzDaily", True), ("airFoil2D", False)])
def test_geometry_matches_openfoam_live(tmp_path, tutorial, needs_blockmesh):
    """Graded (pitzDaily) and curved, strongly stretched (airFoil2D) meshes, where a naive
    vertex-average cell centre would be visibly wrong."""
    src = f"{TUTORIALS}/{tutorial}"
    if not shutil.os.path.isdir(src):
        pytest.skip(f"{src} not found")
    case_dir = tmp_path / tutorial
    shutil.copytree(src, case_dir)
    cd = case_dir / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("writePrecision  6;", "writePrecision  17;")
                  .replace("writePrecision 6;", "writePrecision 17;"))
    if needs_blockmesh:
        _run(["blockMesh"], case_dir)
    _run(["postProcess", "-func", "writeCellCentres", "-time", "0"], case_dir)
    _run(["postProcess", "-func", "writeCellVolumes", "-time", "0"], case_dir)

    mesh = Mesh.read(case_dir)
    C = read_field(case_dir / "0" / "C").internal
    V = read_field(case_dir / "0" / "V").internal
    scale = np.ptp(mesh.points, axis=0).max()
    np.testing.assert_allclose(mesh.cell_centres, C, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(mesh.cell_volumes, V, rtol=1e-9)
