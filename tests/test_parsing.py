import numpy as np
import pytest

from foam2ml.parsing import (FoamFormatError, read_boundary, read_faces, read_field, read_labels,
                             read_points)

HEADER = """FoamFile
{{
    format      {fmt};
    class       {cls};
    object      {obj};
}}
// * * comment * * //
"""


def write(tmp_path, name, cls, body, fmt="ascii"):
    p = tmp_path / name
    p.write_text(HEADER.format(fmt=fmt, cls=cls, obj=name) + body)
    return p


def test_polymesh_files(cavity_dir):
    poly = cavity_dir / "constant" / "polyMesh"
    assert read_points(poly / "points").shape == (882, 3)
    assert read_labels(poly / "owner").shape == (1640,)
    assert read_labels(poly / "neighbour").shape == (760,)
    offsets, ids = read_faces(poly / "faces")
    assert offsets.shape == (1641,) and offsets[-1] == ids.size == 1640 * 4


def test_boundary(cavity_dir):
    patches = read_boundary(cavity_dir / "constant" / "polyMesh" / "boundary")
    assert [(p.name, p.type) for p in patches] == [
        ("movingWall", "wall"), ("fixedWalls", "wall"), ("frontAndBack", "empty")]
    assert patches[0].start_face == 760
    assert patches[-1].start_face + patches[-1].n_faces == 1640


def test_classic_and_compact_faces_agree(tmp_path):
    classic = write(tmp_path, "faces", "faceList", "3\n(\n4(0 1 2 3)\n3(3 2 4)\n4(1 5 6 2)\n)\n")
    compact = write(tmp_path, "facesCompact", "faceCompactList",
                    "4\n(\n0 4 7 11\n)\n11\n(\n0 1 2 3 3 2 4 1 5 6 2\n)\n")
    for off_ids_a, off_ids_b in zip(read_faces(classic), read_faces(compact)):
        np.testing.assert_array_equal(off_ids_a, off_ids_b)


def test_binary_is_rejected_clearly(tmp_path):
    p = write(tmp_path, "owner", "labelList", "2\n(\x00\x01)\n", fmt="binary")
    with pytest.raises(FoamFormatError, match="foamFormatConvert"):
        read_labels(p)


def test_count_mismatch_is_caught(tmp_path):
    p = write(tmp_path, "owner", "labelList", "3\n(\n0 1\n)\n")
    with pytest.raises(FoamFormatError, match="header says 3"):
        read_labels(p)


def test_fields_uniform_and_nonuniform(cavity_dir):
    U0 = read_field(cavity_dir / "0" / "U")
    assert np.allclose(U0.internal, [0, 0, 0])
    assert U0.boundary["movingWall"]["type"] == "fixedValue"
    assert np.allclose(U0.boundary["movingWall"]["value"], [1, 0, 0])
    assert U0.boundary["frontAndBack"] == {"type": "empty"}

    U = read_field(cavity_dir / "0.5" / "U")
    assert U.field_class == "volVectorField" and U.internal.shape == (400, 3)
    p = read_field(cavity_dir / "0.5" / "p")
    assert p.internal.shape == (400,)


def test_nonuniform_boundary_value(tmp_path):
    body = """dimensions [0 1 -1 0 0 0 0];
internalField   nonuniform List<scalar> 3(1.5 2.5 -3e-2);
boundaryField
{
    inlet  { type fixedValue; value nonuniform List<vector> 2((1 0 0) (2 0 0)); }
    outlet { type zeroGradient; }
}
"""
    f = read_field(write(tmp_path, "T", "volScalarField", body))
    np.testing.assert_allclose(f.internal, [1.5, 2.5, -0.03])
    np.testing.assert_allclose(f.boundary["inlet"]["value"], [[1, 0, 0], [2, 0, 0]])
    assert f.boundary["outlet"] == {"type": "zeroGradient"}
