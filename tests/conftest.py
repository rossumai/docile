from pathlib import Path

import pytest

from docile.dataset import Dataset


@pytest.fixture
def sample_dataset_docid() -> str:
    return "516f2d61ea404b30a9192a72"


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path("tests/data/sample-dataset")


@pytest.fixture
def sample_dataset(sample_dataset_path: Path) -> Dataset:
    return Dataset("dev", sample_dataset_path)
