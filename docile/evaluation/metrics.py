from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List, NamedTuple, Optional, Set

from docile.dataset.field import BBOX, PCC, Field


class ComparisonPair(NamedTuple):
    score: float
    gold: Optional[Field]
    pred: Optional[Field]


def pccs_covered(sorted_x_pccs: List[PCC], sorted_y_pccs: List[PCC], bbox: BBOX) -> Set[PCC]:
    """Obtain which PCCs are under the given bbox."""
    l, t, r, b = bbox

    i_l = bisect_left(sorted_x_pccs, l, key=lambda p: p.x)  # type: ignore
    i_r = bisect_right(sorted_x_pccs, r, key=lambda p: p.x)  # type: ignore
    x_subset = set(sorted_x_pccs[i_l:i_r])

    i_t = bisect_left(sorted_y_pccs, t, key=lambda p: p.y)  # type: ignore
    i_b = bisect_right(sorted_y_pccs, b, key=lambda p: p.y)  # type: ignore
    y_subset = set(sorted_y_pccs[i_t:i_b])

    return x_subset.intersection(y_subset)


def pccs_iou(
    sorted_x_pccs: List[PCC], sorted_y_pccs: List[PCC], gold_bbox: BBOX, pred_bbox: BBOX
) -> float:
    """Calculate IOU over Pseudo Character Boxes."""
    golds = pccs_covered(sorted_x_pccs, sorted_y_pccs, gold_bbox)
    preds = pccs_covered(sorted_x_pccs, sorted_y_pccs, pred_bbox)

    return len(golds.intersection(preds)) / len(golds.union(preds))


def get_comparisons(
    annotation: List[Field], pccs: List[PCC], predictions: List[Field]
) -> List[ComparisonPair]:
    """Heuristic that attempts to find the best match between annotations and predictions."""
    sorted_x_pccs = sorted(pccs, key=lambda p: p.x)
    sorted_y_pccs = sorted(pccs, key=lambda p: p.y)

    annotations_by_key = defaultdict(list)
    for a in annotation:
        annotations_by_key[a.fieldtype].append(a)

    predictions_by_key = defaultdict(list)
    for p in predictions:
        predictions_by_key[p.fieldtype].append(p)

    comparison_pairs: List[ComparisonPair] = []

    # FIXME (mskl): The following matching is not guaranteed to be optimal
    all_fieldtypes = set(annotations_by_key.keys()).union(predictions_by_key.keys())
    for fieldtype in all_fieldtypes:
        for ig, gold in enumerate(annotations_by_key[fieldtype]):
            best_i = 0
            best_iou = 0.0
            for ip, pred in enumerate(predictions_by_key[fieldtype]):
                iou = pccs_iou(sorted_x_pccs, sorted_y_pccs, gold.bbox, pred.bbox)
                if iou > best_iou:
                    best_i = ip
                    best_iou = iou
            if best_iou > 0:
                # Only construct the pair if any match was found
                best_pred = predictions_by_key[fieldtype].pop(best_i)
                comparison_pairs.append(ComparisonPair(score=best_iou, gold=gold, pred=best_pred))
                # Also remove the gold field
                annotations_by_key[fieldtype].pop(ig)

    # Remaining unmatched annotations
    for _fieldtype, fields in annotations_by_key.items():
        for field in fields:
            comparison_pairs.append(ComparisonPair(score=0.0, gold=field, pred=None))

    # Remaining unmatched predictions
    for _fieldtype, fields in predictions_by_key.items():
        for field in fields:
            comparison_pairs.append(ComparisonPair(score=0.0, gold=None, pred=field))

    return comparison_pairs
