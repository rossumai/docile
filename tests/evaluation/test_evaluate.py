import random
from copy import deepcopy
from dataclasses import replace
from typing import Tuple

import pytest

from docile.dataset import BBox, Dataset, Field
from docile.evaluation.evaluate import (
    EvaluationResult,
    PredictionsValidationError,
    _get_prediction_sort_key,
    _sort_predictions,
    _validate_predictions,
    compute_metrics,
    evaluate_dataset,
)
from docile.evaluation.pcc_field_matching import FieldMatching


@pytest.fixture
def mock_evaluation_result() -> EvaluationResult:
    field1 = Field(BBox(0, 0, 1, 1), page=0, score=1, fieldtype="f1")
    field05 = Field(BBox(0, 0, 1, 1), page=0, score=0.5, fieldtype="f05")
    field_ap_only = Field(BBox(0, 0, 1, 1), page=0, score=1, fieldtype="f1", use_only_for_ap=True)
    evaluation_result = EvaluationResult(
        task_to_docid_to_matching={
            "kile": {
                "a": FieldMatching(
                    ordered_predictions_with_match=[(field1, field1), (field1, None)],
                    false_negatives=[field1],
                ),
                "b": FieldMatching.empty([field05, field05], [field1]),
            },
            "lir": {
                "a": FieldMatching.empty([], [field1]),
                "b": FieldMatching(
                    ordered_predictions_with_match=[
                        (field1, field1),
                        (field05, None),
                        (field05, None),
                        (field05, field05),
                        (field_ap_only, None),
                        (field_ap_only, None),
                        (field_ap_only, None),
                        (field_ap_only, field1),
                    ],
                    false_negatives=[],
                ),
            },
        },
        dataset_name="mock-dataset",
        iou_threshold=1.0,
    )
    return evaluation_result


def test_evaluation_result_get_primary_metric(mock_evaluation_result: EvaluationResult) -> None:
    # AP = 0.5 for KILE because recall is 1/3 and the first prediction is the correct one.
    assert mock_evaluation_result.get_primary_metric("kile") == 1 / 3
    # f1 = 1/2 for LIR because both precision and recall are 0.5 (2/4)
    assert mock_evaluation_result.get_primary_metric("lir") == 1 / 2


def test_evaluation_result_get_metrics(mock_evaluation_result: EvaluationResult) -> None:
    assert mock_evaluation_result.get_metrics("kile") == {
        "TP": 1,
        "FP": 3,
        "FN": 2,
        "precision": 1 / 4,
        "recall": 1 / 3,
        "f1": pytest.approx(2 / 7),
        "AP": 1 / 3,
    }
    assert mock_evaluation_result.get_metrics("lir") == {
        "TP": 2,
        "FP": 2,
        "FN": 2,
        "precision": 1 / 2,
        "recall": 1 / 2,
        "f1": 1 / 2,
        "AP": (1 / 4) * (1 / 1) + (1 / 4) * (2 / 4) + (1 / 4) * (3 / 8),
    }
    assert mock_evaluation_result.get_metrics("kile", fieldtype="f1") == {
        "TP": 1,
        "FP": 1,
        "FN": 2,
        "precision": 1 / 2,
        "recall": 1 / 3,
        "f1": pytest.approx(2 / 5),
        "AP": 1 / 3,
    }
    assert mock_evaluation_result.get_metrics("lir", fieldtype="f05", docids=["b"]) == {
        "TP": 1,
        "FP": 2,
        "FN": 0,
        "precision": pytest.approx(1 / 3),
        "recall": 1,
        "f1": 1 / 2,
        "AP": pytest.approx(1 / 3),
    }


def test_evaluation_result_print_report(mock_evaluation_result: EvaluationResult) -> None:
    assert (
        mock_evaluation_result.print_report(include_fieldtypes=False, floatfmt=".2f")
        == """\
Evaluation report for mock-dataset
==================================
KILE
----
Primary metric (AP): 0.3333333333333333

| fieldtype            |   AP |   f1 |   precision |   recall |   TP |   FP |   FN |
|----------------------|------|------|-------------|----------|------|------|------|
| **-> micro average** | 0.33 | 0.29 |        0.25 |     0.33 |    1 |    3 |    2 |

LIR
---
Primary metric (f1): 0.5

| fieldtype            |   AP |   f1 |   precision |   recall |   TP |   FP |   FN |
|----------------------|------|------|-------------|----------|------|------|------|
| **-> micro average** | 0.47 | 0.50 |        0.50 |     0.50 |    2 |    2 |    2 |

Notes:
* For AP all predictions are used. For f1, precision, recall, TP, FP and FN predictions explicitly marked with flag `use_only_for_ap=True` are excluded.
"""
    )


def _assert_metrics_at_least(
    evaluation_result: EvaluationResult,
    minimum_value: float,
    tasks: Tuple[str, ...] = ("kile", "lir"),
    eval_same_text: Tuple[bool, ...] = (False, True),
    check_metric_names: Tuple[str, ...] = ("AP", "f1", "precision", "recall"),
) -> None:
    for task in tasks:
        for same_text in eval_same_text:
            metrics = evaluation_result.get_metrics(task=task, same_text=same_text)
            for metric_name in check_metric_names:
                assert metrics[metric_name] == minimum_value


def test_evaluate_dataset_perfect_predictions(
    sample_dataset: Dataset, sample_dataset_docid: str
) -> None:
    kile_predictions = {
        sample_dataset_docid: sample_dataset[sample_dataset_docid].annotation.fields
    }
    lir_predictions = {
        sample_dataset_docid: sample_dataset[sample_dataset_docid].annotation.li_fields
    }
    evaluation_result = evaluate_dataset(sample_dataset, kile_predictions, lir_predictions)
    _assert_metrics_at_least(evaluation_result, 1.0)


def test_evaluate_dataset_perfect_predictions_with_perturbations(
    sample_dataset: Dataset, sample_dataset_docid: str, random_seed: int = 402269889108107772
) -> None:
    """Test that changing the bboxes by 0.005% of the width/height does not influence the metric."""

    def _field_perturbation(field: Field, rng: random.Random, max_change: float) -> Field:
        """Change each bbox coordinate by -1 or 1 times 'max_change'."""
        new_bbox = BBox(*(x + rng.randrange(-1, 2, 2) * max_change for x in field.bbox.to_tuple()))
        return replace(field, bbox=new_bbox)

    rng = random.Random(random_seed)
    max_change = 0.005 / 100
    kile_predictions = {
        sample_dataset_docid: [
            _field_perturbation(field, rng, max_change)
            for field in sample_dataset[sample_dataset_docid].annotation.fields
        ]
    }
    lir_predictions = {
        sample_dataset_docid: [
            _field_perturbation(field, rng, max_change)
            for field in sample_dataset[sample_dataset_docid].annotation.li_fields
        ]
    }
    evaluation_result = evaluate_dataset(
        sample_dataset, kile_predictions, lir_predictions, iou_threshold=0.9
    )
    _assert_metrics_at_least(evaluation_result, 1.0)


def test_evaluate_dataset_kile_missing_and_wrong_predictions(
    sample_dataset: Dataset,
    sample_dataset_docid: str,
) -> None:
    kile_predictions = {
        sample_dataset_docid: [
            replace(f, score=1) for f in sample_dataset[sample_dataset_docid].annotation.fields
        ]
    }
    # 1 missing and 3 wrong predictions
    kile_predictions[sample_dataset_docid].pop()
    kile_predictions[sample_dataset_docid][0] = replace(
        kile_predictions[sample_dataset_docid][0],
        page=1000,
        score=0.8,  # all other fields have default score 1.0
    )
    kile_predictions[sample_dataset_docid][1] = replace(
        kile_predictions[sample_dataset_docid][1],
        bbox=BBox(0, 0, 1, 1),
    )
    kile_predictions[sample_dataset_docid][2] = replace(
        kile_predictions[sample_dataset_docid][2],
        fieldtype="mock_fieldtype",
    )

    fields = len(sample_dataset[sample_dataset_docid].annotation.fields)
    false_negatives = 4
    true_positives = fields - false_negatives
    false_positives = 3
    recall = true_positives / fields
    precision = true_positives / (true_positives + false_positives)
    f1 = 2 * precision * recall / (precision + recall)

    # AP computation:
    # After sorting the predictions by score and prediction id, we get the following list:
    #       False, False, True, True, True, ..., True, False
    # The best precision is achieved for the highest recall which means it will be used also for
    # the smaller recall values (check average_precison.py for details).

    # the false prediction with the lowest score does not affect AP value
    ap_precision = true_positives / (true_positives + false_positives - 1)
    ap = recall * ap_precision

    evaluation_result = evaluate_dataset(sample_dataset, kile_predictions, {})
    assert evaluation_result.get_metrics("kile") == {
        "AP": pytest.approx(ap),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
    }


def test_evaluate_dataset_lir_missing_and_wrong_predictions(
    sample_dataset: Dataset,
    sample_dataset_docid: str,
) -> None:
    lir_predictions = {
        sample_dataset_docid: [
            replace(f, score=1)
            for f in deepcopy(sample_dataset[sample_dataset_docid].annotation.li_fields)
        ]
    }
    # missing prediction
    lir_predictions[sample_dataset_docid].pop()
    # wrong page
    lir_predictions[sample_dataset_docid][0] = replace(
        lir_predictions[sample_dataset_docid][0],
        page=1000,
    )
    # duplicated predictions
    lir_predictions[sample_dataset_docid].append(
        lir_predictions[sample_dataset_docid][1],
    )
    lir_predictions[sample_dataset_docid].append(
        lir_predictions[sample_dataset_docid][1],
    )
    # wrong fieldtype
    lir_predictions[sample_dataset_docid][2] = replace(
        lir_predictions[sample_dataset_docid][2],
        fieldtype="mock_fieldtype",
    )
    # assigned to wrong line item
    lir_predictions[sample_dataset_docid][3] = replace(
        lir_predictions[sample_dataset_docid][3],
        line_item_id=1000,
    )

    # change line item ids, does not influence results
    for i in range(len(lir_predictions[sample_dataset_docid])):
        if lir_predictions[sample_dataset_docid][i].line_item_id in [5, 6, 7, 8]:
            # swap line item ids 5<->8 & 6<->7
            new_line_item_id = 13 - lir_predictions[sample_dataset_docid][i].line_item_id
            lir_predictions[sample_dataset_docid][i] = replace(
                lir_predictions[sample_dataset_docid][i],
                line_item_id=new_line_item_id,
            )

    fields = len(sample_dataset[sample_dataset_docid].annotation.li_fields)
    false_negatives = 4
    true_positives = fields - false_negatives
    false_positives = 5
    recall = true_positives / fields
    precision = true_positives / (true_positives + false_positives)
    f1 = 2 * precision * recall / (precision + recall)

    # AP computation:
    # After sorting the predictions by score and prediction id, we get the following list:
    #       False, True, False, False, True, True, True, ..., True, False, False
    # The best precision is achieved for the highest recall which means it will be used also for
    # the smaller recall values (check average_precison.py for details).

    # The 2 extra predictions are ignored as they are last.
    ap_precision = true_positives / (true_positives + false_positives - 2)
    ap = recall * ap_precision

    evaluation_result = evaluate_dataset(sample_dataset, {}, lir_predictions)
    assert evaluation_result.get_metrics("lir") == {
        "AP": pytest.approx(ap),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
    }


def test_compute_metrics() -> None:
    field1 = Field(BBox(0, 0, 1, 1), page=0, score=1)
    field05 = Field(BBox(0, 0, 1, 1), page=0, score=0.5)
    field_ap_only = Field(BBox(0, 0, 1, 1), page=0, score=0.25, use_only_for_ap=True)
    docid_to_field_matching = {
        "a": FieldMatching(
            ordered_predictions_with_match=[
                (field_ap_only, field1),
                (field_ap_only, None),
                (field05, None),
                (field1, field1),
                (field05, field05),
                (field1, None),
            ],
            false_negatives=[],
        ),
        "b": FieldMatching(
            ordered_predictions_with_match=[(field1, field1)],
            false_negatives=[],
        ),
    }
    assert compute_metrics(docid_to_field_matching) == {
        "TP": 3,
        "FP": 2,
        "FN": 1,
        "precision": 3 / 5,
        "recall": 3 / 4,
        "f1": pytest.approx(2 / 3),
        # sorted predictions matched: [True, True, False, False, True, True, False]
        # (recall, precision) pairs: [(0.25, 1), (0.5, 1), (0.75, 0.6), (1, 4/6)]
        # after "filling gaps": [(0.5, 1), (1, 4/6)]
        "AP": pytest.approx(0.5 * 1 + 0.5 * 4 / 6),
    }


def test_validate_predictions(sample_dataset: Dataset, sample_dataset_docid: str) -> None:
    bbox = BBox(0, 0, 1, 1)
    with pytest.raises(
        PredictionsValidationError,
        match="You need to provide at least one prediction for at least one of the tasks.",
    ):
        _validate_predictions(sample_dataset, {})

    too_many_predictions = {"task": {sample_dataset_docid: [Field(bbox=bbox, page=2)] * 1001}}
    with pytest.raises(
        PredictionsValidationError,
        match=f"TASK: Exceeded limit of 1000 predictions per page for doc: {sample_dataset_docid}",
    ):
        _validate_predictions(sample_dataset, too_many_predictions)

    missing_fieldtype = {"task": {sample_dataset_docid: [Field(bbox=bbox, page=0)]}}
    with pytest.raises(
        PredictionsValidationError, match="TASK: Prediction is missing 'fieldtype'."
    ):
        _validate_predictions(sample_dataset, missing_fieldtype)

    with_line_item_id = {
        sample_dataset_docid: [Field(bbox=bbox, page=0, fieldtype="f", line_item_id=8)]
    }
    _validate_predictions(sample_dataset, {"lir": with_line_item_id})  # ok
    with pytest.raises(
        PredictionsValidationError, match="KILE: Prediction has extra 'line_item_id'."
    ):
        _validate_predictions(sample_dataset, {"kile": with_line_item_id})

    without_line_item_id = {sample_dataset_docid: [Field(bbox=bbox, page=0, fieldtype="f")]}
    _validate_predictions(sample_dataset, {"kile": without_line_item_id})
    with pytest.raises(
        PredictionsValidationError, match="LIR: Prediction is missing 'line_item_id'."
    ):
        _validate_predictions(sample_dataset, {"lir": without_line_item_id})

    only_part_with_scores = {
        sample_dataset_docid: [
            Field(bbox=bbox, page=0, fieldtype="f1"),
            Field(bbox=bbox, page=0, fieldtype="f2", score=1.0),
        ]
    }
    with pytest.raises(
        PredictionsValidationError,
        match="TASK: Either all or no predictions should have 'score' defined",
    ):
        _validate_predictions(sample_dataset, {"task": only_part_with_scores})

    extra_doc = {sample_dataset_docid: [], "mock-docid": []}
    with pytest.raises(
        PredictionsValidationError,
        match="TASK: Predictions provided for 1 documents not in the dataset sample-dataset:dev",
    ):
        _validate_predictions(sample_dataset, {"task": extra_doc})

    missing_doc = {}
    with pytest.raises(
        PredictionsValidationError,
        match=(
            "TASK: Predictions not provided for 1/1 documents. Pass an empty list of predictions "
            "for these documents if this was intended."
        ),
    ):
        _validate_predictions(sample_dataset, {"task": missing_doc})


def test_sort_predictions() -> None:
    bbox = BBox(0, 0, 1, 1)
    f_gold = Field(bbox=bbox, page=0)
    predictions_doc_a = [
        Field(bbox=bbox, text="a", page=0, score=0.4),
        Field(bbox=bbox, text="a", page=0, score=1),
        Field(bbox=bbox, text="a", page=0, score=0.8),
        Field(bbox=bbox, text="a", page=0, score=0.4),
    ]
    predictions_doc_b = [
        Field(bbox=bbox, text="b", page=0, score=0.8),
        Field(bbox=bbox, text="b", page=0, score=0.4),
        Field(bbox=bbox, text="b", page=0, score=0.5),
    ]
    docid_to_matching = {
        "a": FieldMatching(
            ordered_predictions_with_match=[
                (predictions_doc_a[0], f_gold),
                (predictions_doc_a[1], f_gold),
                (predictions_doc_a[2], f_gold),
                (predictions_doc_a[3], f_gold),
            ],
            false_negatives=[f_gold],
        ),
        "b": FieldMatching(
            ordered_predictions_with_match=[
                (predictions_doc_b[0], None),
                (predictions_doc_b[1], None),
                (predictions_doc_b[2], None),
            ],
            false_negatives=[],
        ),
    }

    actual_sorted_predictions = _sort_predictions(docid_to_matching)
    expected_sorted_predictions = [
        True,  # a[1], score=1
        False,  # b[0], score=0.8, pred_i=0
        True,  # a[2], score=0.8, pred_i=2
        False,  # b[2], score=0.5
        True,  # a[0], score=0.4, pred_i=0
        False,  # b[1], score=0.4, pred_i=1
        True,  # a[3], score=0.4, pred_i=3
    ]
    assert actual_sorted_predictions == expected_sorted_predictions


def test_sort_predictions_with_ap_only() -> None:
    bbox = BBox(0, 0, 1, 1)
    f_gold = Field(bbox=bbox, page=0)
    predictions_doc_a = [
        Field(bbox=bbox, text="a", page=0, score=0.4),
        Field(bbox=bbox, text="a", page=0, score=1),
        Field(bbox=bbox, text="a", page=0, score=0.8, use_only_for_ap=True),
        Field(bbox=bbox, text="a", page=0, score=0.4),
    ]
    predictions_doc_b = [
        Field(bbox=bbox, text="b", page=0, score=0.8),
        Field(bbox=bbox, text="b", page=0, score=0.4, use_only_for_ap=True),
        Field(bbox=bbox, text="b", page=0, score=0.5),
    ]
    docid_to_matching = {
        "a": FieldMatching(
            ordered_predictions_with_match=[
                (predictions_doc_a[0], f_gold),
                (predictions_doc_a[1], f_gold),
                (predictions_doc_a[2], f_gold),
                (predictions_doc_a[3], f_gold),
            ],
            false_negatives=[f_gold],
        ),
        "b": FieldMatching(
            ordered_predictions_with_match=[
                (predictions_doc_b[0], None),
                (predictions_doc_b[1], None),
                (predictions_doc_b[2], None),
            ],
            false_negatives=[],
        ),
    }

    # fields with use_only_for_ap=True will be last
    actual_sorted_predictions = _sort_predictions(docid_to_matching)
    expected_sorted_predictions = [
        True,  # a[1], score=1
        False,  # b[0], score=0.8, pred_i=0
        False,  # b[2], score=0.5
        True,  # a[0], score=0.4, pred_i=0
        True,  # a[3], score 0.4, pred_i=3
        True,  # a[2], use_only_for_ap=True, score=0.8, pred_i=2
        False,  # b[1], use_only_for_ap=True, score=0.4, pred_i=1
    ]
    assert actual_sorted_predictions == expected_sorted_predictions


def test_get_prediction_sort_key() -> None:
    a0 = _get_prediction_sort_key((False, -1.0), 0, "a")
    b0 = _get_prediction_sort_key((False, -1.0), 0, "b")
    a2 = _get_prediction_sort_key((True, -1.5), 2, "a")
    b2 = _get_prediction_sort_key((True, -1.5), 2, "b")
    assert a0 == ((False, -1.0), 0, "d95520d967275249")
    assert b0 == ((False, -1.0), 0, "0407a70fca4cc072")
    assert a2 == ((True, -1.5), 2, "55d1989cd656edd2")
    assert b2 == ((True, -1.5), 2, "77b295898f2dbda5")
    # The two pairs (a0,b0) and (a2,b2) have the same score and prediction_i but are in different
    # order for the two docs. This is thanks to the hashing of docid with prediction_i.
    assert b0 < a0
    assert a2 < b2
