from typing import Sequence


def compute_average_precision(
    sorted_predictions_matched: Sequence[bool],
    total_annotations: int,
) -> float:
    """
    Compute average precision (AP).

    There are some design decisions that influence the result of AP computation. These were done in
    line with how AP is computed in COCO evaluation for object detection.
    1.  When the precision-recall curve has a zig-zag pattern (precision increased for higher
        recall), the "gaps are filled".
    2.  For two consecutive (recall,precision) pairs (r1,p1), (r2,p2) where r2>r1 we use the
        precision 'p2' in the interval [r1,r2] when computing the Average Precision.

    Points 1. and 2. can be also explained as computing the integral (from 0 to 1) over a function
    'precision(r)' which is defined as:
        precision(r) == max{p' | there exists (p',r') with r' >= r}


    Parameters
    ----------
    sorted_predictions_matched
        An indicator for each prediction whether it was matched or not. Predictions should be
        sorted by score (from the highest to the lowest).
    total_annotations
        Total number of ground truth annotations, used to calculate the recall.

    Returns
    -------
    Average precision metric
    """
    if total_annotations == 0:
        return 0.0

    recall_precision_pairs = [[0.0, 1.0]]  # the precision here is not used
    true_positives = 0
    observed_predictions = 0

    # Iteratively update precision and recall.
    for matched in sorted_predictions_matched:
        true_positives += 1 if matched else 0
        observed_predictions += 1
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
