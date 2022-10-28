from pathlib import Path

from docile.dataset import Dataset
from docile.document import Document


def test_dataset_from_file(sample_dataset_docid: str, sample_dataset_path: Path) -> None:
    dataset = Dataset.from_file("dev", sample_dataset_path)
    assert len(dataset) == len(list(dataset)) == 1  # type: ignore
    assert isinstance(dataset[0], Document)
    assert isinstance(dataset[sample_dataset_docid], Document)
