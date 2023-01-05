import pytest

from docile.dataset import BBox, Field
from docile.evaluation.pcc import PCC, PCCSet
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches, pccs_iou


def test_filter_field_matching() -> None:
    bbox = BBox(0, 0, 1, 1)
    field_f1_no_text = Field(bbox, page=0, fieldtype="f1")
    field_f1_text_a = Field(bbox, page=0, fieldtype="f1", text="a")
    field_f1_text_b = Field(bbox, page=0, fieldtype="f1", text="b", use_only_for_ap=True)
    field_f2 = Field(bbox, page=0, fieldtype="f2", text="x")
    field_matching = FieldMatching(
        ordered_predictions_with_match=[
            (field_f1_no_text, None),
            (field_f1_no_text, field_f1_text_a),
            (field_f1_text_a, field_f1_text_a),
            (field_f1_text_b, field_f1_text_a),
            (field_f1_text_b, None),
            (field_f2, field_f2),
        ],
        false_negatives=[field_f2, field_f1_text_a],
    )

    assert field_matching.filter() == field_matching
    assert field_matching.filter(same_text=True) == FieldMatching(
        ordered_predictions_with_match=[
            (field_f1_no_text, None),
            (field_f1_no_text, None),
            (field_f1_text_a, field_f1_text_a),
            (field_f1_text_b, None),
            (field_f1_text_b, None),
            (field_f2, field_f2),
        ],
        false_negatives=[field_f2, field_f1_text_a, field_f1_text_a, field_f1_text_a],
    )
    assert field_matching.filter(fieldtype="f1") == FieldMatching(
        ordered_predictions_with_match=[
            (field_f1_no_text, None),
            (field_f1_no_text, field_f1_text_a),
            (field_f1_text_a, field_f1_text_a),
            (field_f1_text_b, field_f1_text_a),
            (field_f1_text_b, None),
        ],
        false_negatives=[field_f1_text_a],
    )
    assert field_matching.filter(same_text=True, fieldtype="f1") == FieldMatching(
        ordered_predictions_with_match=[
            (field_f1_no_text, None),
            (field_f1_no_text, None),
            (field_f1_text_a, field_f1_text_a),
            (field_f1_text_b, None),
            (field_f1_text_b, None),
        ],
        false_negatives=[field_f1_text_a, field_f1_text_a, field_f1_text_a],
    )
    assert field_matching.filter(exclude_only_for_ap=True) == FieldMatching(
        ordered_predictions_with_match=[
            (field_f1_no_text, None),
            (field_f1_no_text, field_f1_text_a),
            (field_f1_text_a, field_f1_text_a),
            (field_f2, field_f2),
        ],
        false_negatives=[field_f2, field_f1_text_a, field_f1_text_a],
    )


def test_pccs_iou() -> None:
    pcc_set = PCCSet([PCC(0, 0.5, 1), PCC(0.5, 0.5, 1), PCC(1, 1, 1)])
    gold_bbox = BBox(0.0, 0.0, 1.0, 1.0)
    pred_bbox = BBox(0.25, 0.25, 0.75, 0.75)
    assert pccs_iou(pcc_set, gold_bbox, pred_bbox, page=1) == pytest.approx(1 / 3)


def test_pccs_iou_empty() -> None:
    pcc_set = PCCSet([PCC(1, 1, 1)])
    gold_bbox_empty = BBox(0.25, 0.25, 0.75, 0.75)
    pred_bbox_empty = BBox(0.0, 0.0, 0.75, 0.75)
    pred_bbox_nonempty = BBox(0.0, 0.0, 1.25, 1.25)

    assert pccs_iou(pcc_set, gold_bbox_empty, pred_bbox_empty, page=1) == 1.0
    assert pccs_iou(pcc_set, gold_bbox_empty, pred_bbox_nonempty, page=1) == 0.0


def test_get_matches() -> None:
    pcc_set = PCCSet(
        [
            PCC(0, 0, 0),
            PCC(0.1, 0.1, 0),
            PCC(0.2, 0.1, 0),
            PCC(0.5, 0.4, 0),
            PCC(0.5, 0.6, 0),
            PCC(1, 1, 0),
            PCC(0.5, 0.5, 1),
        ]
    )

    annotations = [
        Field(fieldtype="full_match", text="ab", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),
        Field(fieldtype="no_match", text="ab", bbox=BBox(0, 0, 0.3, 0.2), page=0),
        Field(fieldtype="full_match", text="ab", bbox=BBox(0.4, 0.4, 0.7, 0.7), page=1),
        Field(fieldtype="partial_match", text="ab", bbox=BBox(0.05, 0.05, 0.3, 0.2), page=0),
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

    matching = get_matches(predictions=predictions, annotations=annotations, pcc_set=pcc_set)
    assert len(matching.matches) == 2
    assert all(
        match.pred.fieldtype == match.gold.fieldtype == "full_match" for match in matching.matches
    )
    assert len(matching.false_positives) == 4
    assert len(matching.false_negatives) == 3
    assert matching.ordered_predictions_with_match == [
        (predictions[0], annotations[0]),
        (predictions[1], annotations[2]),
        (predictions[2], None),
        (predictions[3], None),
        (predictions[4], None),
        (predictions[5], None),
    ]

    matching_iou05 = get_matches(
        predictions=predictions, annotations=annotations, pcc_set=pcc_set, iou_threshold=0.5
    )
    assert len(matching_iou05.matches) == 3
    assert all(
        match.pred.fieldtype == match.gold.fieldtype
        and match.pred.fieldtype in ["full_match", "partial_match"]
        for match in matching_iou05.matches
    )
    assert len(matching_iou05.false_positives) == 3
    assert len(matching_iou05.false_negatives) == 2
    assert matching_iou05.ordered_predictions_with_match == [
        (predictions[0], annotations[0]),
        (predictions[1], annotations[2]),
        (predictions[2], annotations[3]),
        (predictions[3], None),
        (predictions[4], None),
        (predictions[5], None),
    ]
