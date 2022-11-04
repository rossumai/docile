import math

from docile.dataset import PCC, BBox, Field
from docile.evaluation.pcc_field_matching import get_matches, pccs_covered, pccs_iou


def test_pccs_covered() -> None:
    sorted_pccs = [PCC(0, 0.5, 0), PCC(0.5, 0.5, 0), PCC(1, 1, 0)]
    bbox = BBox(0.25, 0.25, 0.75, 0.75)
    assert pccs_covered(sorted_pccs, sorted_pccs, bbox) == {sorted_pccs[1]}


def test_pccs_iou() -> None:
    sorted_pccs = [PCC(0, 0.5, 1), PCC(0.5, 0.5, 1), PCC(1, 1, 1)]
    gold_bbox = BBox(0.0, 0.0, 1.0, 1.0)
    pred_bbox = BBox(0.25, 0.25, 0.75, 0.75)
    assert math.isclose(pccs_iou(sorted_pccs, sorted_pccs, gold_bbox, pred_bbox), 1 / 3)


def test_get_matches() -> None:
    pccs = [
        PCC(0, 0, 0),
        PCC(0.1, 0.1, 0),
        PCC(0.2, 0.1, 0),
        PCC(0.5, 0.4, 0),
        PCC(0.5, 0.6, 0),
        PCC(1, 1, 0),
        PCC(0.5, 0.5, 1),
    ]

    annotations = [
        Field(fieldtype="full_match", text="ab", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),
        Field(fieldtype="full_match", text="ab", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=1),
        Field(fieldtype="partial_match", text="ab", bbox=BBox(0.05, 0.05, 0.3, 0.2), page=0),
        Field(fieldtype="no_match", text="ab", bbox=BBox(0, 0, 0.3, 0.2), page=0),
        Field(fieldtype="no_match", text="ab", bbox=BBox(0, 0, 1.0, 1.0), page=0),
    ]
    predictions = [
        Field(fieldtype="full_match", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),
        Field(fieldtype="full_match", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=1),
        Field(fieldtype="partial_match", bbox=BBox(0.15, 0.05, 0.3, 0.2), page=0),
        Field(fieldtype="no_match", bbox=BBox(0.9, 0.9, 1, 1), page=0),
        Field(fieldtype="no_match", bbox=BBox(0.7, 0.7, 0.8, 0.8), page=0),
        Field(fieldtype="no_match", bbox=BBox(0, 0, 1.0, 1.0), page=1),  # mismatching page
    ]

    matching = get_matches(predictions=predictions, annotations=annotations, pccs=pccs)
    assert len(matching.matches) == 2
    assert all(
        match.pred.fieldtype == match.gold.fieldtype == "full_match" for match in matching.matches
    )
    assert len(matching.extra) == 4
    assert len(matching.misses) == 3
    matching_iou05 = get_matches(
        predictions=predictions, annotations=annotations, pccs=pccs, iou_threshold=0.5
    )
    assert len(matching_iou05.matches) == 3
    assert all(
        match.pred.fieldtype == match.gold.fieldtype
        and match.pred.fieldtype in ["full_match", "partial_match"]
        for match in matching_iou05.matches
    )
    assert len(matching_iou05.extra) == 3
    assert len(matching_iou05.misses) == 2
