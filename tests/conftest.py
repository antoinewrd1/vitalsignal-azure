import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def spark():
    """One session for the whole test run -- JVM startup dominates otherwise."""
    from vitalsignal.spark import get_spark

    s = get_spark("pytest")
    yield s
    s.stop()
