from docile.dataset.bbox import BBox
from docile.dataset.dataset import Dataset


def test_document_annotation_get_table_bbox(
    sample_dataset: Dataset, sample_dataset_docid: str
) -> None:
    doc = sample_dataset[sample_dataset_docid]
    assert doc.annotation.get_table_bbox(page=0) == BBox(
        left=133 / 1240,
        top=579 / 1645,
        right=(133 + 999) / 1240,
        bottom=(579 + 844) / 1645,
    )
