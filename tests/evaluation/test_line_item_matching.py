import pytest

from docile.dataset import BBox, Field
from docile.evaluation import PCC, PCCSet
from docile.evaluation.line_item_matching import (
    _get_covering_bbox,
    _get_line_item_id,
    _place_bbox_in_document,
    get_lir_matches,
)
from docile.evaluation.pcc_field_matching import MatchedPair


def test_get_line_item_id() -> None:
    assert _get_line_item_id(Field(bbox=BBox(0, 0, 1, 1), page=0, line_item_id=3)) == 3
    with pytest.raises(ValueError):
        _get_line_item_id(Field(bbox=BBox(0, 0, 1, 1), page=0))


def test_place_bbox_in_document() -> None:
    bbox = BBox(0.2, 0.3, 0.4, 0.5)
    assert _place_bbox_in_document(bbox, 0) == bbox
    assert _place_bbox_in_document(bbox, 1) == BBox(0.2, 1.3, 0.4, 1.5)
    assert _place_bbox_in_document(bbox, 2) == BBox(0.2, 2.3, 0.4, 2.5)


def test_get_covering_bbox() -> None:
    with pytest.raises(ValueError):
        _get_covering_bbox([])

    bboxes = [
        BBox(0.2, 0.3, 0.4, 0.5),
        BBox(0.2, 0.3, 0.4, 0.5),
        BBox(0.3, 0.2, 0.35, 0.35),
        BBox(0.3, 1.1, 0.35, 1.2),
    ]
    assert _get_covering_bbox(bboxes[:1]) == bboxes[0]
    assert _get_covering_bbox(bboxes[:2]) == bboxes[0]
    assert _get_covering_bbox(bboxes[:3]) == BBox(0.2, 0.2, 0.4, 0.5)
    assert _get_covering_bbox(iter(bboxes)) == BBox(0.2, 0.2, 0.4, 1.2)


def test_get_lir_matches() -> None:
    pcc_set = PCCSet(
        [
            PCC(0, 0, 0),
            PCC(0.1, 0.1, 0),
            PCC(0.2, 0.1, 0),
            PCC(0.5, 0.4, 0),
            PCC(0.5, 0.6, 0),
            PCC(1, 1, 0),
            PCC(0.1, 0.1, 1),
        ]
    )

    annotations = [
        Field(fieldtype="a", text="ab", line_item_id=0, bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),
        Field(fieldtype="b", text="ab", line_item_id=0, bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),
        Field(fieldtype="a", text="ab", line_item_id=1, bbox=BBox(0.05, 0.05, 0.3, 0.2), page=0),
        Field(fieldtype="c", text="ab", line_item_id=1, bbox=BBox(0, 0, 0.3, 0.2), page=0),
        Field(fieldtype="a", text="ab", line_item_id=2, bbox=BBox(0.4, 0.5, 1.0, 1.0), page=0),
        Field(fieldtype="b", text="ab", line_item_id=2, bbox=BBox(0, 0, 0.2, 0.2), page=1),
    ]
    predictions = [
        # pred LI 0: 1 match with gold LI 0
        Field(fieldtype="a", line_item_id=4, bbox=BBox(0.4, 0.4, 0.7, 0.7), page=0),  # match in 0
        Field(fieldtype="b", line_item_id=4, bbox=BBox(0.4, 0.4, 0.7, 0.55), page=0),
        # pred LI 1: 1 matches with gold LI 1, 2 matches with gold LI 2
        Field(fieldtype="a", line_item_id=1, bbox=BBox(0.4, 0.5, 1.0, 1.0), page=0),  # match in 2
        Field(fieldtype="c", line_item_id=1, bbox=BBox(0, 0, 0.3, 0.2), page=0),  # match in 1
        Field(fieldtype="b", line_item_id=1, bbox=BBox(0, 0, 0.2, 0.2), page=1),  # match in 2
        # pred LI 2: 2 matches with gold LI 2 + 2 extra matches with gold LI 1 but with predictions
        # marked as `use_only_for_ap=True` that do not affect line item matching.
        Field(
            fieldtype="a", line_item_id=2, bbox=BBox(0.25, 0.59, 1.0, 1.0), page=0
        ),  # match in 2
        Field(
            fieldtype="b", line_item_id=2, bbox=BBox(0.05, 0.05, 0.15, 0.15), page=1
        ),  # match in 2
        Field(
            fieldtype="a",
            line_item_id=2,
            bbox=BBox(0.05, 0.05, 0.3, 0.2),
            page=0,
            use_only_for_ap=True,
        ),  # match in 1
        Field(
            fieldtype="c", line_item_id=2, bbox=BBox(0, 0, 0.3, 0.2), page=0, use_only_for_ap=True
        ),  # match in 1
    ]

    # While greedy matching might assign pred line item (LI) 1 to gold LI 2, maximum matching
    # will assign it to gold LI 1 (so that pred LI 2 can be assigned to gold LI 2). Notice that the
    # predictions marked with `use_only_for_ap=True` are ignored for matching of LIs.

    field_matching, li_matching = get_lir_matches(
        predictions=predictions, annotations=annotations, pcc_set=pcc_set, iou_threshold=1
    )
    assert li_matching == {4: 0, 1: 1, 2: 2}
    assert set(field_matching.matches) == {
        MatchedPair(pred=predictions[0], gold=annotations[0]),
        MatchedPair(pred=predictions[3], gold=annotations[3]),
        MatchedPair(pred=predictions[5], gold=annotations[4]),
        MatchedPair(pred=predictions[6], gold=annotations[5]),
    }
    assert set(field_matching.false_positives) == {
        predictions[1],
        predictions[2],
        predictions[4],
        predictions[7],
        predictions[8],
    }
    assert set(field_matching.false_negatives) == {annotations[1], annotations[2]}

    assert field_matching.ordered_predictions_with_match == [
        (predictions[0], annotations[0]),
        (predictions[1], None),
        (predictions[2], None),
        (predictions[3], annotations[3]),
        (predictions[4], None),
        (predictions[5], annotations[4]),
        (predictions[6], annotations[5]),
        (predictions[7], None),
        (predictions[8], None),
    ]
