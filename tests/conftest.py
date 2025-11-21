from pathlib import Path
import pytest

@pytest.fixture
def project_root() -> Path:
    # returns the project root directory (CalibraFlow)
    return Path(__file__).resolve().parents[1]
