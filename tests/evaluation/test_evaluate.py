import random
from dataclasses import replace

import pytest

from docile.dataset import BBox, Dataset, Field
from docile.evaluation.evaluate import evaluate_dataset


def test_evaluate_dataset_perfect_predictions(
    sample_dataset: Dataset, sample_dataset_docid: str
) -> None:
    kile_predictions = {
        sample_dataset_docid: sample_dataset[sample_dataset_docid].annotation.fields
    }
    # TODO: add LIR predictions to test # noqa
    evaluation_dict = evaluate_dataset(sample_dataset, kile_predictions, {})
    assert evaluation_dict == {"kile": 1.0, "lir": 0.0}


def test_evaluate_dataset_perfect_predictions_with_perturbations(
    sample_dataset: Dataset, sample_dataset_docid: str, random_seed: int = 402269889108107772
) -> None:
    """Test that changing the bboxes by 0.02% of the width/height does not influence the metric."""
    rng = random.Random(random_seed)
    max_change = 0.02 / 100
    kile_predictions = {
        sample_dataset_docid: [
            _field_perturbation(field, rng, max_change)
            for field in sample_dataset[sample_dataset_docid].annotation.fields
        ]
    }
    # TODO: add LIR predictions to test # noqa
    evaluation_dict = evaluate_dataset(sample_dataset, kile_predictions, {})
    assert evaluation_dict == {"kile": 1.0, "lir": 0.0}


def test_evaluate_dataset_kile_missing_and_wrong_predictions(
    sample_dataset: Dataset,
    sample_dataset_docid: str,
) -> None:
    kile_predictions = {
        sample_dataset_docid: [
            replace(f, score=1) for f in sample_dataset[sample_dataset_docid].annotation.fields
        ]
    }
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
    precision = (fields - 4) / (fields - 2)  # only predictions with score == 1 are counted
    recall = (fields - 4) / fields
    kile = precision * recall
    evaluation_dict = evaluate_dataset(sample_dataset, kile_predictions, {})
    assert evaluation_dict == {"kile": pytest.approx(kile), "lir": 0.0}


def _field_perturbation(field: Field, rng: random.Random, max_change: float) -> Field:
    """Change each bbox coordinate by -1 or 1 times 'max_change'."""
    new_bbox = BBox(*(x + rng.randrange(-1, 2, 2) * max_change for x in field.bbox.to_tuple()))
    return replace(field, bbox=new_bbox)
