from pathlib import Path

import pytest

from asd_eeg.config import load_config
from asd_eeg.self_check import run_self_check

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "configs/default.yaml")


@pytest.fixture(scope="session")
def completed_check(config, tmp_path_factory):
    return run_self_check(config, tmp_path_factory.mktemp("integration"))
