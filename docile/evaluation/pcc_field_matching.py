from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from docile.dataset import PCC, BBox, Field

# Small value for robust >= on floats.
EPS = 1e-6


@dataclass(frozen=True)
class MatchedPair:
    gold: Field
    pred: Field


@dataclass(frozen=True)
class FieldMatching:
    matches: Sequence[MatchedPair]
    extra: Sequence[Field]  # not matched predictions
    misses: Sequence[Field]  # not matched annotations


def pccs_covered(
    sorted_x_pccs: Sequence[PCC], sorted_y_pccs: Sequence[PCC], bbox: BBox
) -> Set[PCC]:
    """Obtain which PCCs are under the given bbox."""

    i_l = bisect_left(sorted_x_pccs, bbox.left, key=lambda p: p.x)
    i_r = bisect_right(sorted_x_pccs, bbox.right, key=lambda p: p.x)
    x_subset = set(sorted_x_pccs[i_l:i_r])

    i_t = bisect_left(sorted_y_pccs, bbox.top, key=lambda p: p.y)
    i_b = bisect_right(sorted_y_pccs, bbox.bottom, key=lambda p: p.y)
    y_subset = set(sorted_y_pccs[i_t:i_b])

    return x_subset.intersection(y_subset)


def pccs_iou(
    sorted_x_pccs: Sequence[PCC], sorted_y_pccs: Sequence[PCC], gold_bbox: BBox, pred_bbox: BBox
) -> float:
    """Calculate IOU over Pseudo Character Boxes."""
    if not gold_bbox.intersects(pred_bbox):
        return 0

    golds = pccs_covered(sorted_x_pccs, sorted_y_pccs, gold_bbox)
    preds = pccs_covered(sorted_x_pccs, sorted_y_pccs, pred_bbox)

    if len(golds) == len(preds) == 0:
        return 1

    return len(golds.intersection(preds)) / len(golds.union(preds))


def get_matches(
    predictions: Sequence[Field],
    annotations: Sequence[Field],
    pccs: Iterable[PCC],
    iou_threshold: float = 1,
) -> FieldMatching:
    """
    Find matching between predictions and annotations.

    Parameters
    ----------
    predictions
        Either KILE fields from one page/document or LI fields from one line item. Notice
        that one line item can span multiple pages. These predictions are being matched in the
        sorted order by 'score' and in the original order if no score is given (or is equal for
        several predictions).
    annotations
        KILE or LI gold fields for the same page/document.
    pccs
        Pseudo-Character-Centers (PCCs) covering all pages that have any of the
        predictions/annotations fields.
    iou_threshold
        Necessary 'intersection / union' to accept a pair of fields as a match. The official
        evaluation uses threshold 1.0 but lower thresholds can be used for debugging.
    """
    have_scores = sum(1 for f in predictions if f.score is not None)
    if have_scores > 0 and have_scores < len(predictions):
        raise ValueError("Either all or no predictions need to have scores")

    page_to_pccs = defaultdict(list)
    for pcc in pccs:
        page_to_pccs[pcc.page].append(pcc)
    page_to_sorted_x_pccs = {
        page: sorted(page_pccs, key=lambda p: p.x) for page, page_pccs in page_to_pccs.items()
    }
    page_to_sorted_y_pccs = {
        page: sorted(page_pccs, key=lambda p: p.y) for page, page_pccs in page_to_pccs.items()
    }

    key_page_to_annotations = defaultdict(lambda: defaultdict(list))
    for a in annotations:
        key_page_to_annotations[a.fieldtype][a.page].append(a)

    key_to_predictions = defaultdict(list)
    for p in predictions:
        key_to_predictions[p.fieldtype].append(p)

    matched_pairs: List[MatchedPair] = []
    extra: List[Field] = []

    all_fieldtypes = set(key_page_to_annotations.keys()).union(key_to_predictions.keys())
    for fieldtype in all_fieldtypes:
        for pred in sorted(key_to_predictions[fieldtype], key=lambda pred: -(pred.score or 1)):
            for gold_i, gold in enumerate(key_page_to_annotations[fieldtype][pred.page]):
                iou = pccs_iou(
                    sorted_x_pccs=page_to_sorted_x_pccs[pred.page],
                    sorted_y_pccs=page_to_sorted_y_pccs[pred.page],
                    gold_bbox=gold.bbox,
                    pred_bbox=pred.bbox,
                )
                if iou > iou_threshold - EPS:
                    matched_pairs.append(MatchedPair(gold=gold, pred=pred))
                    key_page_to_annotations[fieldtype][pred.page].pop(gold_i)
                    break
            else:
                extra.append(pred)

    misses = [
        field
        for page_to_fields in key_page_to_annotations.values()
        for fields in page_to_fields.values()
        for field in fields
    ]

    return FieldMatching(matches=matched_pairs, extra=extra, misses=misses)
