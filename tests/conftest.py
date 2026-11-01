import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("APOLLO_JPL_EPHEMERIS", "1")


@pytest.fixture(scope="session")
def package_path():
    # the suite's fixtures and expectations are the Jul-16 build's; other slots (apollo11_nominal_package_<slot>.json)
    # sit beside it and are gated with the CLI tools
    jul16 = os.path.join(HERE, "nominal_packages", "apollo11_nominal_package.json")
    if os.path.exists(jul16):
        return jul16
    hits = sorted(glob.glob(os.path.join(HERE, "nominal_packages", "*.json")))
    if not hits:
        pytest.skip("no package in nominal_packages/")
    return hits[-1]


@pytest.fixture(scope="session")
def consumer():
    import apollo11 as A
    return A
