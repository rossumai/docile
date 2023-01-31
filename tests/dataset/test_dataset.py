from pathlib import Path

import pytest

from docile.dataset import Dataset, Document


def test_dataset_init(sample_dataset_docid: str, sample_dataset_path: Path) -> None:
    dataset = Dataset("dev", sample_dataset_path)
    assert len(dataset) == len(list(dataset)) == 1  # type: ignore
    assert isinstance(dataset[0], Document)
    assert isinstance(dataset[sample_dataset_docid], Document)

    custom_dataset = Dataset(
        "non-existent-split", sample_dataset_path, docids=[sample_dataset_docid]
    )
    assert custom_dataset.docids == [sample_dataset_docid]
    # It is possible to give both the index and list of docids if they agree with each other
    assert (
        dataset.docids == Dataset("dev", sample_dataset_path, docids=[sample_dataset_docid]).docids
    )

    with pytest.raises(ValueError):
        Dataset("dev", sample_dataset_path, docids=["different-docid"])
    with pytest.raises(ValueError):
        Dataset("non-existent-split", sample_dataset_path)
