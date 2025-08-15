import os
import shutil
from typing import Iterable

import pytest


@pytest.fixture(scope="session")
def output_root() -> str:
    return "/workspaces/hypertoolz/tests/output/"


@pytest.fixture(scope="session")
def test_files_root() -> str:
    return "/workspaces/hypertoolz/tests/test_files/"


@pytest.fixture(scope="session")
def config_yaml_path(test_files_root: str) -> str:
    return os.path.join(test_files_root, "example_tuner_config.yaml")


@pytest.fixture(autouse=True, scope="function")
def prep_and_cleanup(output_root: str) -> Iterable[None]:
    # Run before the test
    yield
    # Run after the test
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
