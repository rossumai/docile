import pytest

from docile.evaluation.average_precision import compute_average_precision


def test_compute_average_precision_same_scores() -> None:
    """When all scores are equal the metric returns simply precision * recall."""
    predictions = [(1, True), (1, False), (1, False), (1, True), (1, True)]
    total_annotations = 10
    expected_average_precision = (3 / 5) * (3 / 10)  # precision * recall
    assert compute_average_precision(predictions, total_annotations) == pytest.approx(
        expected_average_precision
    )

    predictions_sorted_false = sorted(predictions, key=lambda sm: sm[1])
    predictions_sorted_true = sorted(predictions, key=lambda sm: not sm[1])
    assert compute_average_precision(predictions_sorted_false, total_annotations) == pytest.approx(
        expected_average_precision
    )
    assert compute_average_precision(predictions_sorted_true, total_annotations) == pytest.approx(
        expected_average_precision
    )


def test_compute_average_precision() -> None:
    # Recall 0.5 is achieved with perfect precision, recall 0.75 is achieved with precision 0.75
    # and higher recall cannot be achieved.
    predictions = [
        (0.0, False),
        (0.0, False),
        (0.5, False),
        (0.5, True),
        (1.0, True),
        (1.0, True),
    ]
    total_annotations = 4
    assert compute_average_precision(predictions, total_annotations) == pytest.approx(
        0.5 * 1.0 + 0.25 * 0.75
    )
    # Throwing out false predictions with lowest score does not influence the result.
    assert compute_average_precision(predictions[2:], total_annotations) == pytest.approx(
        0.5 * 1.0 + 0.25 * 0.75
    )
    # Throwing out false prediction if the same (or lower) score contains also true predictions
    # improves the result.
    assert compute_average_precision(predictions[3:], total_annotations) == pytest.approx(
        0.5 * 1.0 + 0.25 * 1.0
    )


def test_compute_average_precision_zig_zag() -> None:
    """If precision for lower score is better than for higher score, the better precision is used."""
    predictions = [
        (0, True),
        (0.5, False),
        (0.5, True),
    ]
    total_annotations = 2
    assert compute_average_precision(predictions, total_annotations) == pytest.approx(
        0.5 * 2 / 3 + 0.5 * 2 / 3
    )
