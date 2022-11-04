import itertools
import logging
from enum import Enum, auto
from typing import Mapping, Sequence

from tqdm import tqdm

from docile.dataset import Dataset, Field
from docile.evaluation.average_precision import compute_average_precision
from docile.evaluation.pcc_field_matching import get_matches

logger = logging.getLogger(__name__)


class Metric(Enum):
    KILE = auto()
    LIR = auto()
    KILE_WITH_TEXT = auto()
    LIR_WITH_TEXT = auto()


def evaluate_dataset(
    dataset: Dataset,
    docid_to_predictions: Mapping[str, Sequence[Field]],
    metric: Metric,
) -> float:
    extra = set(docid_to_predictions.keys()).difference(dataset.docids)
    missing = set(dataset.docids).difference(docid_to_predictions.keys())
    if extra:
        logger.warning(
            f"Trying to evaluate predictions for {len(extra)} not in the dataset {dataset}"
        )
    if missing:
        logger.warning(f"Did not find any predictions for {len(missing)}/{len(dataset)} documents")

    if metric == Metric.KILE:
        predictions_score_matched = []
        total_annotations = 0
        for docid, predictions in tqdm(docid_to_predictions.items()):
            document = dataset[docid]
            annotations = document.annotation.fields
            pccs = itertools.chain(
                *(document.ocr.get_all_pccs(page) for page in range(document.page_count))
            )
            field_matching = get_matches(
                predictions=predictions, annotations=annotations, pccs=pccs
            )
            predictions_score_matched.extend(
                (match.pred.score if match.pred.score is not None else 1, 1)
                for match in field_matching.matches
            )
            predictions_score_matched.extend(
                (pred.score if pred.score is not None else 1, 0) for pred in field_matching.extra
            )
            total_annotations += len(annotations)

        average_precision = compute_average_precision(
            predictions_score_matched=predictions_score_matched,
            total_annotations=total_annotations,
        )
        return average_precision
    else:
        raise NotImplementedError
