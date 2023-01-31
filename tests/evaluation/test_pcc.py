import pytest

from docile.dataset import BBox, Dataset
from docile.evaluation.pcc import PCC, PCCSet, _calculate_pccs, _get_snapped_ocr_words


def test_get_covered_pccs() -> None:
    pccs = [PCC(0, 0.5, 0), PCC(0.5, 0.5, 0), PCC(1, 1, 0), PCC(0.5, 0.5, 1), PCC(0.4, 0.6, 1)]
    pcc_set = PCCSet(pccs)
    bbox = BBox(0.25, 0.25, 0.75, 0.75)
    assert pcc_set.get_covered_pccs(bbox, 0) == {PCC(0.5, 0.5, 0)}
    assert pcc_set.get_covered_pccs(bbox, 1) == {PCC(0.5, 0.5, 1), PCC(0.4, 0.6, 1)}
    assert pcc_set.get_covered_pccs(BBox(0.39, 0.59, 0.41, 0.61), 1) == {PCC(0.4, 0.6, 1)}


def test_get_snapped_ocr_words(sample_dataset: Dataset, sample_dataset_docid: str) -> None:
    document = sample_dataset[sample_dataset_docid]
    words = _get_snapped_ocr_words(document)
    original_words = []
    for page in range(document.page_count):
        original_words.extend(document.ocr.get_all_words(page))

    for word in words:
        assert _bbox_area(word.bbox) > 0

    words_total_area = sum(_bbox_area(word.bbox) for word in words)
    original_words_total_area = sum(_bbox_area(word.bbox) for word in original_words)

    # On average, area of snapped bounding boxes decreases to 50-60% of the original area (for the
    # default OCR method).
    assert words_total_area < 0.9 * original_words_total_area


def test_calculate_pccs() -> None:
    bbox = BBox(0.2, 0.1, 0.8, 0.3)
    with pytest.raises(ValueError):
        assert _calculate_pccs(bbox, "", 0)

    y = pytest.approx(0.2)
    assert _calculate_pccs(bbox, "x", 0) == [PCC(pytest.approx(0.5), y, 0)]  # type: ignore
    assert _calculate_pccs(bbox, "xx", 1) == [
        PCC(pytest.approx(0.35), y, 1),  # type: ignore
        PCC(pytest.approx(0.65), y, 1),  # type: ignore
    ]
    assert _calculate_pccs(bbox, "xxx", 2) == [
        PCC(pytest.approx(0.3), y, 2),  # type: ignore
        PCC(pytest.approx(0.5), y, 2),  # type: ignore
        PCC(pytest.approx(0.7), y, 2),  # type: ignore
    ]


def _bbox_area(bbox: BBox) -> float:
    return (bbox.right - bbox.left) * (bbox.bottom - bbox.top)
