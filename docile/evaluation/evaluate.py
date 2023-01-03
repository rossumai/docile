import logging
import operator
from typing import Dict, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from docile.dataset import Dataset, Field
from docile.evaluation.average_precision import compute_average_precision
from docile.evaluation.line_item_matching import get_lir_matches
from docile.evaluation.pcc import get_document_pccs
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches

logger = logging.getLogger(__name__)


PredictionSortKey = Tuple[float, int, int]


def evaluate_dataset(
    dataset: Dataset,
    docid_to_kile_predictions: Mapping[str, Sequence[Field]],
    docid_to_lir_predictions: Mapping[str, Sequence[Field]],
    with_text: bool = False,
    iou_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate the dataset on KILE and LIR using the given predictions.

    If evaluating only on one of these metrics, simply provide no predictions for the second metric.

    Parameters
    ----------
    dataset
        Dataset with gold annotations to evaluate on.
    docid_to_kile_predictions
        Mapping from doc ids (in the 'dataset') to KILE predictions.
    docid_to_lir_predictions
        Mapping from doc ids (in the 'dataset') to LIR predictions.
    with_text
        If True, evaluate (also) by comparing the read-out text.
    iou_threshold
        Necessary 'intersection / union' to accept a pair of fields as a match. The official
        evaluation uses threshold 1.0 but lower thresholds can be used for debugging.

    Returns
    -------
    Dictionary from metric name to the float value.
    """
    if with_text:
        raise NotImplementedError("Comparing the read-out text is not implemented yet.")

    metric_to_predictions = {
        "kile": docid_to_kile_predictions,
        "lir": docid_to_lir_predictions,
    }
    for metric, docid_to_predictions in metric_to_predictions.items():
        have_scores = sum(
            sum(1 for f in fields if f.score is not None)
            for fields in docid_to_predictions.values()
        )
        if have_scores > 0 and have_scores < sum(
            len(fields) for fields in docid_to_predictions.values()
        ):
            raise ValueError("Either all or no predictions need to have scores")

        extra = len(set(docid_to_predictions.keys()).difference(dataset.docids))
        missing = len(set(dataset.docids).difference(docid_to_predictions.keys()))
        if extra:
            logger.warning(
                f"{metric}: Found predictions for {extra} documents not in the dataset {dataset}, "
                "they will be ignored."
            )
        if missing:
            logger.warning(
                f"{metric}: Did not find any predictions for {missing}/{len(dataset)} documents"
            )

    metric_to_sort_key_matched_pairs = {"kile": [], "lir": []}
    if with_text:
        metric_to_sort_key_matched_pairs.update({"kile_with_text": [], "lir_with_text": []})

    metric_to_total_annotations = {metric: 0 for metric in metric_to_sort_key_matched_pairs.keys()}

    for document in tqdm(dataset, desc="Run matching for documents"):
        pcc_set = get_document_pccs(document)

        kile_annotations = document.annotation.fields
        kile_predictions = docid_to_kile_predictions.get(document.docid, [])
        lir_annotations = document.annotation.li_fields
        lir_predictions = docid_to_lir_predictions.get(document.docid, [])

        # TODO: pass use_text to matching
        for use_text in [False, True] if with_text else [False]:
            metric = "kile_with_text" if use_text else "kile"
            kile_matching = get_matches(
                predictions=kile_predictions,
                annotations=kile_annotations,
                pcc_set=pcc_set,
                iou_threshold=iou_threshold,
            )
            metric_to_sort_key_matched_pairs[metric].extend(
                _get_sort_key_matched_pairs(kile_matching, document.docid)
            )
            metric_to_total_annotations[metric] += len(kile_annotations)

            metric = "lir_with_text" if use_text else "lir"
            lir_matching, _line_item_matching = get_lir_matches(
                predictions=lir_predictions,
                annotations=lir_annotations,
                pcc_set=pcc_set,
                iou_threshold=iou_threshold,
            )
            metric_to_sort_key_matched_pairs[metric].extend(
                _get_sort_key_matched_pairs(lir_matching, document.docid)
            )
            metric_to_total_annotations[metric] += len(lir_annotations)

    metric_to_average_precision = {}
    for metric, sort_key_matched_pairs in metric_to_sort_key_matched_pairs.items():
        sorted_predictions_matched = [
            matched
            for _sort_key, matched in sorted(sort_key_matched_pairs, key=operator.itemgetter(0))
        ]
        metric_to_average_precision[metric] = compute_average_precision(
            sorted_predictions_matched=sorted_predictions_matched,
            total_annotations=metric_to_total_annotations[metric],
        )
    return metric_to_average_precision


def _get_prediction_sort_key(
    score: Optional[float], prediction_i: int, docid: str
) -> PredictionSortKey:
    """
    Get a sort key for a prediction.

    For evaluation purposes, predictions are sorted by these criteria (sorted by importance):
    1.  Score from the highest to the lowest.
    2.  Original order in which the predictions were passed in.
    3.  The document id. Document id is hashed together with the prediction_i to make sure
        documents are not always sorted in the same order (for different prediction indices) which
        would make some documents more important for the evaluation than others.
    """
    if score is None:
        score = 1
    return (-score, prediction_i, hash((docid, prediction_i)))


def _get_sort_key_matched_pairs(
    field_matching: FieldMatching, docid: str
) -> Sequence[Tuple[PredictionSortKey, bool]]:
    """For each prediction return its sort key and an indicator whether it was matched."""
    return [
        (_get_prediction_sort_key(pred.score, pred_i, docid), gold is not None)
        for pred_i, (pred, gold) in enumerate(field_matching.ordered_predictions_with_match)
    ]
