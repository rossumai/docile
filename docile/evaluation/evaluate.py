import itertools
import logging
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from tqdm import tqdm

from docile.dataset import Dataset, Field
from docile.evaluation.average_precision import compute_average_precision
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset: Dataset,
    docid_to_kile_predictions: Mapping[str, Sequence[Field]],
    docid_to_lir_predictions: Mapping[str, Sequence[Field]],
    with_text: bool = False,
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

    metric_to_score_matched_pairs = {"kile": [], "lir": []}
    if with_text:
        metric_to_score_matched_pairs.update({"kile_with_text": [], "lir_with_text": []})

    metric_to_total_annotations = {metric: 0 for metric in metric_to_score_matched_pairs.keys()}

    for document in tqdm(dataset, desc="Run matching for documents"):
        pccs = itertools.chain(
            *(document.ocr.get_all_pccs(page) for page in range(document.page_count))
        )

        kile_annotations = document.annotation.fields
        kile_predictions = docid_to_kile_predictions[document.docid]
        li_annotations = document.annotation.li_fields

        # TODO: pass use_text to matching and compute LIR metric as well # noqa
        for use_text in [False, True] if with_text else [False]:
            metric = "kile_with_text" if use_text else "kile"
            metric_to_score_matched_pairs[metric].extend(
                _get_score_matched_pairs(
                    get_matches(
                        predictions=kile_predictions, annotations=kile_annotations, pccs=pccs
                    )
                )
            )
            metric_to_total_annotations[metric] += len(kile_annotations)
            metric = "lir_with_text" if use_text else "lir"
            metric_to_score_matched_pairs[metric].extend([])
            metric_to_total_annotations[metric] += len(li_annotations)

    metric_to_average_precision = {
        metric: compute_average_precision(
            predictions_score_matched=score_matched_pairs,
            total_annotations=metric_to_total_annotations[metric],
        )
        for metric, score_matched_pairs in metric_to_score_matched_pairs.items()
    }
    return metric_to_average_precision


def _get_score_matched_pairs(field_matching: FieldMatching) -> Iterable[Tuple[float, bool]]:
    return itertools.chain(
        (
            (match.pred.score if match.pred.score is not None else 1, True)
            for match in field_matching.matches
        ),
        ((pred.score if pred.score is not None else 1, False) for pred in field_matching.extra),
    )
