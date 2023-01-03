import pytest

from docile.evaluation.average_precision import compute_average_precision


def test_compute_average_precision() -> None:
    # Recall 0.5 is achieved with perfect precision, recall 0.75 is achieved with precision 0.75
    # and higher recall cannot be achieved.
    predictions = [True, True, False, True, False, False]
    total_annotations = 4
    assert compute_average_precision(predictions, total_annotations) == pytest.approx(
        0.5 * 1.0 + 0.25 * 0.75
    )
    # Throwing out false predictions with lowest score does not influence the result.
    assert compute_average_precision(
        [True, True, False, True], total_annotations
    ) == pytest.approx(0.5 * 1.0 + 0.25 * 0.75)
    # Throwing out false prediction if the same (or lower) score contains also true predictions
    # improves the result.
    assert compute_average_precision(
        [True, True, True, False, False], total_annotations
    ) == pytest.approx(0.5 * 1.0 + 0.25 * 1.0)


def test_compute_average_precision_fill_gaps() -> None:
    """Test the influence of "filling gaps"."""

    predictions = [True, False, False, True, True]
    # precision recall pairs:
    # recall -> precision
    #    0.1    1/1
    #    0.2    2/4, adjusted to 3/5 achievable for recall 0.3
    #    0.3    3/5
    total_annotations = 10
    expected_average_precision = 0.1 * 1 / 1 + 0.1 * 3 / 5 + 0.1 * 3 / 5  # 0.22
    assert compute_average_precision(predictions, total_annotations) == pytest.approx(
        expected_average_precision
    )

    predictions_sorted_false_first = sorted(predictions, key=lambda sm: sm)
    # precision recall pairs:
    # recall -> precision
    #    0.1    1/3, adjusted to 3/5 achievable for recall 0.3
    #    0.2    2/4, adjusted to 3/5 achievable for recall 0.3
    #    0.3    3/5
    expected_average_precision = 0.3 * 3 / 5  # 0.18
    assert compute_average_precision(
        predictions_sorted_false_first, total_annotations
    ) == pytest.approx(expected_average_precision)

    predictions_sorted_true_first = sorted(predictions, key=lambda sm: not sm)
    # precision recall pairs:
    # recall -> precision
    #    0.1    1
    #    0.2    1
    #    0.3    1
    assert compute_average_precision(
        predictions_sorted_true_first, total_annotations
    ) == pytest.approx(0.3)
