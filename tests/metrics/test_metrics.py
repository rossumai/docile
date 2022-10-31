import math

from docile.dataset.field import PCC, Field
from docile.evaluation.metrics import get_comparisons, pccs_covered, pccs_iou


def test_pccs_covered() -> None:
    sorted_pccs = [PCC(0, 0.5), PCC(0.5, 0.5), PCC(1, 1)]
    bbox = (0.25, 0.25, 0.75, 0.75)
    assert pccs_covered(sorted_pccs, sorted_pccs, bbox) == {sorted_pccs[1]}


def test_pccs_iou() -> None:
    sorted_pccs = [PCC(0, 0.5), PCC(0.5, 0.5), PCC(1, 1)]
    gold_bbox = (0.0, 0.0, 1.0, 1.0)
    pred_bbox = (0.25, 0.25, 0.75, 0.75)
    assert math.isclose(pccs_iou(sorted_pccs, sorted_pccs, gold_bbox, pred_bbox), 1 / 3)


def test_get_comparisons() -> None:
    pccs = [PCC(0, 0), PCC(0.1, 0.1), PCC(0.2, 0.1), PCC(0.5, 0.4), PCC(0.5, 0.6), PCC(1, 1)]

    annotation = [
        Field(fieldtype="miss", text="ab", bbox=(0, 0, 0.3, 0.2)),
        Field(fieldtype="partial_match", text="ab", bbox=(0.05, 0.05, 0.3, 0.2)),
        Field(fieldtype="full_match", text="ab", bbox=(0.4, 0.4, 0.7, 0.7)),
    ]
    prediction = [
        Field(fieldtype="miss", bbox=(0.9, 0.9, 1, 1)),
        Field(fieldtype="partial_match", bbox=(0.15, 0.05, 0.3, 0.2)),
        Field(fieldtype="full_match", bbox=(0.4, 0.4, 0.7, 0.7)),
        Field(fieldtype="extra", bbox=(0.7, 0.7, 0.8, 0.8)),
    ]

    pairs = get_comparisons(annotation, pccs, prediction)
    # Two fields generated by misses, one extra, one partial and one full
    assert len(pairs) == 5
    partial = [p for p in pairs if p.gold and p.gold.fieldtype == "partial_match"][0]
    assert math.isclose(partial.score, 0.5)
    full = [p for p in pairs if p.gold and p.gold.fieldtype == "full_match"][0]
    assert math.isclose(full.score, 1)
    extra = [p for p in pairs if p.pred and p.pred.fieldtype == "extra"][0]
    assert extra.gold is None
    # Field predicted on wrong place does not get paired, so one is true miss and other is extra
    misses = [
        p
        for p in pairs
        if p.gold and p.gold.fieldtype == "miss" or p.pred and p.pred.fieldtype == "miss"
    ]
    assert len(misses) == 2
