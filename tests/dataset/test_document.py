from pathlib import Path

from docile.dataset import Document


def test_document_load(sample_dataset_docid: str, sample_dataset_path: Path) -> None:
    document = Document(sample_dataset_docid, sample_dataset_path)

    assert document.docid == sample_dataset_docid
    assert document.page_count == 1
    assert document.annotation is not None
