from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence

from docile.dataset import BBox, Field
from docile.evaluation.pcc import PCCSet

# Small value for robust >= on floats.
EPS = 1e-6


@dataclass(frozen=True)
class MatchedPair:
    gold: Field
    pred: Field


@dataclass(frozen=True)
class FieldMatching:
    matches: Sequence[MatchedPair]
    false_positives: Sequence[Field]  # not matched predictions
    false_negatives: Sequence[Field]  # not matched annotations


def pccs_iou(pcc_set: PCCSet, gold_bbox: BBox, pred_bbox: BBox, page: int) -> float:
    """Calculate IOU over Pseudo Character Centers."""
    if not gold_bbox.intersects(pred_bbox):
        return 0

    golds = pcc_set.get_covered_pccs(gold_bbox, page)
    preds = pcc_set.get_covered_pccs(pred_bbox, page)

    if len(golds) == len(preds) == 0:
        return 1

    return len(golds.intersection(preds)) / len(golds.union(preds))


def get_matches(
    predictions: Sequence[Field],
    annotations: Sequence[Field],
    pcc_set: PCCSet,
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
    pcc_set
        Pseudo-Character-Centers (PCCs) covering all pages that have any of the
        predictions/annotations fields.
    iou_threshold
        Necessary 'intersection / union' to accept a pair of fields as a match. The official
        evaluation uses threshold 1.0 but lower thresholds can be used for debugging.
    """
    have_scores = sum(1 for f in predictions if f.score is not None)
    if have_scores > 0 and have_scores < len(predictions):
        raise ValueError("Either all or no predictions need to have scores")

    key_page_to_annotations = defaultdict(lambda: defaultdict(list))
    for a in annotations:
        key_page_to_annotations[a.fieldtype][a.page].append(a)

    key_to_predictions = defaultdict(list)
    for p in predictions:
        key_to_predictions[p.fieldtype].append(p)

    matched_pairs: List[MatchedPair] = []
    false_positives: List[Field] = []

    all_fieldtypes = set(key_page_to_annotations.keys()).union(key_to_predictions.keys())
    for fieldtype in all_fieldtypes:
        for pred in sorted(key_to_predictions[fieldtype], key=lambda pred: -(pred.score or 1)):
            for gold_i, gold in enumerate(key_page_to_annotations[fieldtype][pred.page]):
                iou = pccs_iou(
                    pcc_set=pcc_set,
                    gold_bbox=gold.bbox,
                    pred_bbox=pred.bbox,
                    page=pred.page,
                )
                if iou > iou_threshold - EPS:
                    matched_pairs.append(MatchedPair(gold=gold, pred=pred))
                    key_page_to_annotations[fieldtype][pred.page].pop(gold_i)
                    break
            else:
                false_positives.append(pred)

    false_negatives = [
        field
        for page_to_fields in key_page_to_annotations.values()
        for fields in page_to_fields.values()
        for field in fields
    ]

    return FieldMatching(
        matches=matched_pairs, false_positives=false_positives, false_negatives=false_negatives
    )
