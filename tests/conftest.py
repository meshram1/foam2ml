import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def cavity_dir() -> Path:
    """OpenFOAM's lid-driven cavity (20x20x1 cells), solved to t=0.5.

    0.5/C and 0.5/V are OpenFOAM's own cell centres and volumes (writeCellCentres /
    writeCellVolumes at writePrecision 17), used as ground truth.
    """
    return DATA / "cavity"


def openfoam_available() -> bool:
    return shutil.which("postProcess") is not None and shutil.which("blockMesh") is not None
