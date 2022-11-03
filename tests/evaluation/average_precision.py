import itertools
from typing import Sequence, Tuple


def _sort_by_score(score_matched: Tuple[float, bool]) -> float:
    return -score_matched[0]


def compute_average_precision(
    predictions_score_matched: Sequence[Tuple[float, bool]],
    total_annotations: int,
) -> float:
    recall_precision_pairs = [[0.0, 1.0]]
    true_positives = 0
    observed_predictions = 0
    sorted_predictions = sorted(predictions_score_matched, key=_sort_by_score)
    for _score, group_it in itertools.groupby(sorted_predictions, key=_sort_by_score):
        group = list(group_it)
        true_positives += sum(1 for _score, matched in group if matched)
        observed_predictions += len(group)
        recall_precision_pairs.append(
            [(true_positives / total_annotations), (true_positives / observed_predictions)]
        )

    # Update precision to maximum precision for any larger recall
    for rp_pair, rp_pair_prev in zip(
        recall_precision_pairs[:0:-1], recall_precision_pairs[-2::-1]
    ):
        rp_pair_prev[1] = max(rp_pair_prev[1], rp_pair[1])

    if recall_precision_pairs[-1][0] < 1:
        recall_precision_pairs.append([1.0, 0.0])

    average_precision = 0.0
    for rp_pair, rp_pair_next in zip(recall_precision_pairs[:-1], recall_precision_pairs[1:]):
        recall_diff = rp_pair_next[0] - rp_pair[0]
        precision = rp_pair[1]
        average_precision += recall_diff * precision

    return average_precision
