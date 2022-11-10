import itertools
from typing import Sequence, Tuple


def _sort_by_score(score_matched: Tuple[float, bool]) -> float:
    """Return -score to sort (score,match) pairs from the highest to the lowest score."""
    return -score_matched[0]


def compute_average_precision(
    predictions_score_matched: Sequence[Tuple[float, bool]],
    total_annotations: int,
) -> float:
    """
    Compute average precision (AP).

    There are multiple design decisions that influence the result:
    1.  All predictions with the same score are added in a batch. For the case when all scores are
        equal the metric is equivalent to 'precision * recall'.
    2.  When the precision-recall curve has a zig-zag pattern (higher recall does not mean lower
        precision), the "gaps are filled".
    3.  For two consecutive (recall,precision) pairs (r1,p1), (r2,p2) where r2>r1 we use the
        precision 'p2' in the interval [r1,r2] when computing the Average Precision.

    Point 1. influences how we construct the sequence of points (p,r) specifying the precision for
    some recall threshold. Points 2. and 3. can be also explained as computing the integral (from 0
    to 1) over function 'precision(r)' which is defined as:
        precision(r) == max{p' | there exists (p',r') with r' >= r}


    Parameters
    ----------
    predictions_score_matched
        List of predictions with their score (confidence) and whether a match was found for the
        prediction or not.
    total_annotations
        Total number of ground truth annotations, used to calculate the recall.

    Returns
    -------
    Average precision metric
    """
    recall_precision_pairs = [[0.0, 1.0]]  # the precision here is not used
    true_positives = 0
    observed_predictions = 0

    # All predictions with the same score are added in a batch.
    sorted_predictions = sorted(predictions_score_matched, key=_sort_by_score)
    for _score, group_it in itertools.groupby(sorted_predictions, key=_sort_by_score):
        group = list(group_it)
        true_positives += sum(1 for _score, matched in group if matched)
        observed_predictions += len(group)
        recall_precision_pairs.append(
            [(true_positives / total_annotations), (true_positives / observed_predictions)]
        )

    # Update precision to maximum precision for any larger recall
    for recall_precision, recall_precision_prev in zip(
        recall_precision_pairs[:0:-1], recall_precision_pairs[-2::-1]
    ):
        recall_precision_prev[1] = max(recall_precision_prev[1], recall_precision[1])

    average_precision = 0.0
    for recall_precision, recall_precision_next in zip(
        recall_precision_pairs[:-1], recall_precision_pairs[1:]
    ):
        # Notice that if there are multiple (recall,precision) pairs with the same recall, they are
        # sorted by precision (from highest to lowest). This means that only the first point (with
        # highest precision) influences the result (for the rest 'recall_diff == 0').
        recall_diff = recall_precision_next[0] - recall_precision[0]
        precision = recall_precision_next[1]
        average_precision += recall_diff * precision

    return average_precision
