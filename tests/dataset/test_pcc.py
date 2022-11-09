import pytest

from docile.dataset.bbox import BBox
from docile.dataset.pcc import PCC, PCCSet, calculate_pccs


def test_calculate_pccs() -> None:
    bbox = BBox(0.2, 0.1, 0.8, 0.3)
    with pytest.raises(ValueError):
        assert calculate_pccs(bbox, "", 0)

    y = pytest.approx(0.2)
    assert calculate_pccs(bbox, "x", 0) == [PCC(pytest.approx(0.5), y, 0)]  # type: ignore
    assert calculate_pccs(bbox, "xx", 1) == [
        PCC(pytest.approx(0.35), y, 1),  # type: ignore
        PCC(pytest.approx(0.65), y, 1),  # type: ignore
    ]
    assert calculate_pccs(bbox, "xxx", 2) == [
        PCC(pytest.approx(0.3), y, 2),  # type: ignore
        PCC(pytest.approx(0.5), y, 2),  # type: ignore
        PCC(pytest.approx(0.7), y, 2),  # type: ignore
    ]


def test_get_covered_pccs() -> None:
    pccs = [PCC(0, 0.5, 0), PCC(0.5, 0.5, 0), PCC(1, 1, 0), PCC(0.5, 0.5, 1), PCC(0.4, 0.6, 1)]
    pcc_set = PCCSet(pccs)
    bbox = BBox(0.25, 0.25, 0.75, 0.75)
    assert pcc_set.get_covered_pccs(bbox, 0) == {pccs[1]}
    assert pcc_set.get_covered_pccs(bbox, 1) == {pccs[3], pccs[4]}
