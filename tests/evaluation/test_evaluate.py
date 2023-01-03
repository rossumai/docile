import random
from copy import deepcopy
from dataclasses import replace
from unittest import mock

import pytest

from docile.dataset import BBox, Dataset, Field
from docile.evaluation.evaluate import _get_sort_key_matched_pairs, evaluate_dataset
from docile.evaluation.pcc_field_matching import FieldMatching


def test_evaluate_dataset_perfect_predictions(
    sample_dataset: Dataset, sample_dataset_docid: str
) -> None:
    kile_predictions = {
        sample_dataset_docid: sample_dataset[sample_dataset_docid].annotation.fields
    }
    lir_predictions = {
        sample_dataset_docid: sample_dataset[sample_dataset_docid].annotation.li_fields
    }
    evaluation_dict = evaluate_dataset(sample_dataset, kile_predictions, lir_predictions)
    assert evaluation_dict == {"kile": 1.0, "lir": 1.0}


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
    evaluation_dict = evaluate_dataset(
        sample_dataset, kile_predictions, lir_predictions, iou_threshold=0.9
    )
    assert evaluation_dict == {"kile": 1.0, "lir": 1.0}


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

    # After sorting the predictions by score and prediction id, we get the following list:
    #       False, False, True, True, True, ..., True, False
    # The best precision is achieved for the highest recall which means it will be used also for
    # the smaller recall values (check average_precison.py for details).

    fields = len(sample_dataset[sample_dataset_docid].annotation.fields)
    true_positives = fields - 4
    recall = true_positives / fields
    # the prediction with score < 1 does not affect the result
    precision = true_positives / (fields - 2)
    kile = recall * precision
    evaluation_dict = evaluate_dataset(sample_dataset, kile_predictions, {})
    assert evaluation_dict == {"kile": pytest.approx(kile), "lir": 0.0}


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

    # After sorting the predictions by score and prediction id, we get the following list:
    #       False, True, False, False, True, True, True, ..., True, False, False
    # The best precision is achieved for the highest recall which means it will be used also for
    # the smaller recall values (check average_precison.py for details).

    fields = len(sample_dataset[sample_dataset_docid].annotation.li_fields)
    true_positives = fields - 4
    recall = true_positives / fields
    # The 2 extra predictions are ignored as they are last.
    precision = true_positives / (fields - 1)
    lir = recall * precision
    evaluation_dict = evaluate_dataset(sample_dataset, {}, lir_predictions)
    assert evaluation_dict == {"kile": 0.0, "lir": pytest.approx(lir)}


def test_get_sort_key_matched_pairs() -> None:
    bbox = BBox(0, 0, 1, 1)
    f_gold = Field(bbox=bbox, page=0)
    field_matching = FieldMatching(
        ordered_predictions_with_match=[
            (Field(bbox=bbox, page=0, score=1), f_gold),
            (Field(bbox=bbox, page=0, score=0.4), f_gold),
            (Field(bbox=bbox, page=0, score=0.4), None),
            (Field(bbox=bbox, page=0, score=0.8), None),
        ],
        false_negatives=[f_gold],
    )

    actual_score_matched_pairs = _get_sort_key_matched_pairs(field_matching, "mock_docid")
    expected_score_matched_pairs = [
        ((-1, 0, mock.ANY), True),
        ((-0.4, 1, mock.ANY), True),
        ((-0.4, 2, mock.ANY), False),
        ((-0.8, 3, mock.ANY), False),
    ]
    assert actual_score_matched_pairs == expected_score_matched_pairs
