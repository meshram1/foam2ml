"""Readers for OpenFOAM's ASCII file format.

Covers what foam2ml needs: polyMesh files (points, faces, owner, neighbour, boundary)
and volume field files (internalField plus per-patch boundary values).
Binary-format files are rejected with a clear error rather than misread.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_HEADER = re.compile(r"FoamFile\s*\{(.*?)\}", re.S)
_ENTRY = re.compile(r"(\w+)\s+([^;]*);")
_LIST_START = re.compile(r"(\d+)\s*\(")
_LIST_END = re.compile(r"\)\s*;")


class FoamFormatError(ValueError):
    """Raised when a file is not in a format foam2ml can read."""


def _strip_comments(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def read_foam_file(path: str | Path) -> tuple[dict, str]:
    """Return (header, body): the FoamFile header entries and the text after the header."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = _strip_comments(path.read_text(errors="replace"))
    m = _HEADER.search(text)
    header = dict(_ENTRY.findall(m.group(1))) if m else {}
    header = {k: v.strip().strip('"') for k, v in header.items()}
    if header.get("format", "ascii") != "ascii":
        raise FoamFormatError(
            f"{path}: format '{header['format']}' is not supported yet; convert with "
            "'foamFormatConvert' after setting writeFormat to ascii in system/controlDict"
        )
    return header, text[m.end():] if m else text


def _top_list(body: str, path: Path) -> tuple[int, str]:
    """Count and inner text of the single top-level list that forms a polyMesh file's body."""
    m = _LIST_START.search(body)
    end = body.rfind(")")
    if not m or end < m.end():
        raise FoamFormatError(f"{path}: no top-level list found")
    return int(m.group(1)), body[m.end():end]


def _numbers(text: str, dtype) -> np.ndarray:
    cleaned = text.replace("(", " ").replace(")", " ")
    return np.array(cleaned.split(), dtype=dtype) if cleaned.strip() else np.empty(0, dtype=dtype)


def _check(n_expected: int, n_got: int, path: Path, what: str) -> None:
    if n_expected != n_got:
        raise FoamFormatError(f"{path}: header says {n_expected} {what}, read {n_got}")


def read_labels(path: str | Path) -> np.ndarray:
    """A labelList such as owner or neighbour, as an int64 array."""
    path = Path(path)
    _, body = read_foam_file(path)
    n, inner = _top_list(body, path)
    labels = _numbers(inner, np.int64)
    _check(n, labels.size, path, "labels")
    return labels


def read_points(path: str | Path) -> np.ndarray:
    """A vectorField such as points, as an (N, 3) float64 array."""
    path = Path(path)
    _, body = read_foam_file(path)
    n, inner = _top_list(body, path)
    pts = _numbers(inner, np.float64)
    if pts.size % 3:
        raise FoamFormatError(f"{path}: {pts.size} values is not a multiple of 3")
    pts = pts.reshape(-1, 3)
    _check(n, len(pts), path, "points")
    return pts


def read_faces(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Faces in CSR form: (offsets, point_ids), so face i is point_ids[offsets[i]:offsets[i+1]].

    Handles both the classic faceList ("4(0 1 22 21)") and faceCompactList (offsets list + labels list).
    """
    path = Path(path)
    header, body = read_foam_file(path)
    if header.get("class") == "faceCompactList":
        first = _LIST_START.search(body)
        close = body.index(")", first.end())
        offsets = _numbers(body[first.end():close], np.int64)
        _check(int(first.group(1)), offsets.size, path, "offsets")
        second = _LIST_START.search(body, close + 1)
        ids = _numbers(body[second.end():body.rfind(")")], np.int64)
        _check(int(second.group(1)), ids.size, path, "point labels")
        return offsets, ids
    n, inner = _top_list(body, path)
    matches = re.findall(r"(\d+)\s*\(([^)]*)\)", inner)
    _check(n, len(matches), path, "faces")
    sizes = np.array([int(k) for k, _ in matches], dtype=np.int64)
    ids = np.array(" ".join(v for _, v in matches).split(), dtype=np.int64)
    _check(int(sizes.sum()), ids.size, path, "face point labels")
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return offsets, ids


@dataclass(frozen=True)
class Patch:
    """One boundary patch: faces start_face .. start_face + n_faces - 1."""

    name: str
    type: str
    n_faces: int
    start_face: int
    extra: dict = field(default_factory=dict)

    @property
    def face_slice(self) -> slice:
        return slice(self.start_face, self.start_face + self.n_faces)


def read_boundary(path: str | Path) -> list[Patch]:
    """The polyMesh boundary file, as a list of Patch in file order."""
    path = Path(path)
    _, body = read_foam_file(path)
    n, inner = _top_list(body, path)
    patches = []
    for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", inner):
        entries = {k: v.strip() for k, v in _ENTRY.findall(block)}
        patches.append(Patch(
            name=name,
            type=entries.pop("type"),
            n_faces=int(entries.pop("nFaces")),
            start_face=int(entries.pop("startFace")),
            extra=entries,
        ))
    _check(n, len(patches), path, "patches")
    return patches


def _parse_value(text: str, path: Path, what: str):
    """Parse 'uniform X' or 'nonuniform List<T> N(...)'. Returns a float, a (3,) array, or an array."""
    text = text.strip()
    if text.startswith("uniform"):
        v = _numbers(text[len("uniform"):], np.float64)
        return float(v[0]) if v.size == 1 else v
    m = re.match(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text)
    if not m:
        raise FoamFormatError(f"{path}: cannot parse {what}: {text[:60]!r}")
    kind, n = m.group(1), int(m.group(2))
    vals = _numbers(text[m.end():text.rfind(")")], np.float64)
    width = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9}.get(kind)
    if width is None:
        raise FoamFormatError(f"{path}: unsupported list type List<{kind}> in {what}")
    _check(n * width, vals.size, path, f"values in {what}")
    return vals if width == 1 else vals.reshape(n, width)


@dataclass
class FieldData:
    """A volume field: internal cell values plus boundary patch definitions."""

    name: str
    field_class: str
    internal: float | np.ndarray
    boundary: dict[str, dict]


def read_field(path: str | Path) -> FieldData:
    """A volScalarField / volVectorField file (internalField and boundaryField)."""
    path = Path(path)
    header, body = read_foam_file(path)
    m = re.search(r"internalField\s+", body)
    if not m:
        raise FoamFormatError(f"{path}: no internalField")
    if body[m.end():].startswith("nonuniform"):
        end = _LIST_END.search(body, m.end()).start() + 1  # keep the list's closing ')'
    else:
        end = body.index(";", m.end())
    internal = _parse_value(body[m.end():end], path, "internalField")

    boundary = {}
    bm = re.search(r"boundaryField\s*\{", body)
    if bm:
        for name, block in re.findall(r"([\w\"\.\*\|\(\)-]+)\s*\{([^{}]*)\}", body[bm.end():]):
            entry = {"type": (re.search(r"type\s+(\w+)\s*;", block) or [None, None])[1]}
            vm = re.search(r"\bvalue\s+(uniform[^;]*|nonuniform.*?\)\s*);", block, re.S)
            if vm:
                entry["value"] = _parse_value(vm.group(1), path, f"value on patch {name}")
            boundary[name.strip('"')] = entry
    return FieldData(name=header.get("object", path.name), field_class=header.get("class", ""),
                     internal=internal, boundary=boundary)
