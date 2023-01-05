import hashlib
import logging
import operator
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union, cast

from tabulate import tabulate
from tqdm import tqdm

from docile.dataset import KILE_FIELDTYPES, LIR_FIELDYPES, Dataset, Field
from docile.evaluation.average_precision import compute_average_precision
from docile.evaluation.line_item_matching import get_lir_matches
from docile.evaluation.pcc import get_document_pccs
from docile.evaluation.pcc_field_matching import FieldMatching, get_matches

logger = logging.getLogger(__name__)


PredictionSortKey = Tuple[float, int, str]

TASK_TO_PRIMARY_METRIC_NAME = {"kile": "AP", "lir": "f1"}
METRIC_NAMES = ["AP", "f1", "precision", "recall", "TP", "FP", "FN"]


@dataclass(frozen=True)
class EvaluationReport:
    """
    Class with the evaluation result.

    It stores the matching between predictions and annotations which can be used to (quickly)
    compute different metrics. The following options are supported:
    * Unmatch predictions whose text differs from the ground truth text (in the primary metric
      this is not required).
    * Filter predictions and annotations to a specific fieldtype.
    * Compute metrics for a single document
    """

    task_to_docid_to_matching: Mapping[str, Mapping[str, FieldMatching]]
    dataset_name: str  # name of evaluated Dataset
    iou_threshold: float  # which value was used to generate this report

    def get_primary_metric(self, task: str) -> float:
        """Return the primary metric used for DocILE'23 benchmark competition."""
        metric = TASK_TO_PRIMARY_METRIC_NAME[task]
        return self.get_metrics(task)[metric]

    def get_metrics(
        self, task: str, same_text: bool = False, fieldtype: str = "", docid: str = ""
    ) -> Dict[str, float]:
        """Get metrics based on several filters.

        Parameters
        ----------
        task
            Task name for which to return the metrics, should be "kile" or "lir".
        same_text
            Require predictions to have exactly the same text as the ground truth in the
            annotation. Note that matching is done based on the location only and this is then just
            used to unmatch predictions with wrong text. This means it can happen that a correct
            prediction is not counted as true positive if there is another prediction in the same
            location with wrong text that was matched to the annotation first.
        fieldtype
            If non-empty, restrict the predictions and annotations to this fieldtype.
        docid
            If non-empty, evaluate only on the single document.

        Returns
        -------
        Dictionary from metric name to the metric value.
        """
        docid_to_matching = self.task_to_docid_to_matching[task]
        if docid != "":
            docid_to_matching = {docid: docid_to_matching[docid]}
        docid_to_filtered_matching = {
            docid: matching.filter(same_text=same_text, fieldtype=fieldtype)
            for docid, matching in docid_to_matching.items()
        }
        return compute_metrics(docid_to_filtered_matching)

    def print_report(
        self,
        docid: str = "",
        include_fieldtypes: bool = True,
        include_same_text: bool = False,
        tablefmt: str = "github",
        floatfmt: str = ".3f",
    ) -> str:
        """
        Return a string with a detailed evaluation report.

        Parameters
        ----------
        docid
            Only restrict to this single document.
        include_fieldtypes
            Also show metrics for each fieldtype separately.
        include_same_text
            Also show results if exact text match is required.
        tablefmt
            Format in which the table should be printed. With 'github' (default) the whole report
            can be stored as a markdown file. You can also use 'latex' to generate a LaTeX table
            definition and other options as defined in the `tabulate` package.
        floatfmt
            Formatting option for floats in tables. Check `tabulate` package for details.

        Returns
        -------
        Multi-line string with the human-readable report.
        """
        if docid == "":
            report = [f"Evaluation report for {self.dataset_name}"]
        else:
            report = [f"Evaluation report for '{docid}' from {self.dataset_name}"]

        iou_threshold_str = ""
        if self.iou_threshold < 1:
            iou_threshold_str = f" [IoU threshold for PCCs = {self.iou_threshold}]"
        report[-1] += iou_threshold_str
        report.append("=" * len(report[-1]))

        for task in sorted(self.task_to_docid_to_matching.keys()):
            same_text_choices = [False, True] if include_same_text else [False]
            for same_text in same_text_choices:
                task_name = task.upper()
                if same_text:
                    task_name += " (with text comparison)"
                report.append(task_name)
                report.append("-" * len(report[-1]))
                summary_metrics = self.get_metrics(task=task, same_text=same_text, docid=docid)
                primary_metric_name = TASK_TO_PRIMARY_METRIC_NAME[task]
                primary_metric = summary_metrics[primary_metric_name]
                report.append(f"Primary metric ({primary_metric_name}): {primary_metric}")
                report.append("")

                assert set(summary_metrics.keys()) == set(METRIC_NAMES)
                headers = ["fieldtype"] + METRIC_NAMES
                rows = [["**-> micro average**"] + [summary_metrics[m] for m in METRIC_NAMES]]

                if include_fieldtypes:
                    fieldtypes = KILE_FIELDTYPES if task == "kile" else LIR_FIELDYPES
                    for fieldtype in fieldtypes:
                        metrics = self.get_metrics(
                            task=task, same_text=same_text, fieldtype=fieldtype, docid=docid
                        )
                        rows.append([fieldtype] + [metrics[m] for m in METRIC_NAMES])

                table = tabulate(rows, headers, tablefmt=tablefmt, floatfmt=floatfmt)
                report.extend(table.splitlines())
                report.append("")
        return "\n".join(report)


def evaluate_dataset(
    dataset: Dataset,
    docid_to_kile_predictions: Mapping[str, Sequence[Field]],
    docid_to_lir_predictions: Mapping[str, Sequence[Field]],
    iou_threshold: float = 1.0,
) -> EvaluationReport:
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
    iou_threshold
        Necessary 'intersection / union' to accept a pair of fields as a match. The official
        evaluation uses threshold 1.0 but lower thresholds can be used for debugging.

    Returns
    -------
    Evaluation report containing the matched predictions. Use its `print_metrics()` method to get
    the metrics.
    """
    _validate_predictions(dataset, docid_to_kile_predictions, docid_to_lir_predictions)

    tasks = [
        task
        for task, docid_to_predictions in [
            ("kile", docid_to_kile_predictions),
            ("lir", docid_to_lir_predictions),
        ]
        if sum(len(predictions) for predictions in docid_to_predictions.values()) > 0
    ]
    task_to_docid_to_matching = {task: {} for task in tasks}
    for document in tqdm(dataset, desc="Run matching for documents"):
        pcc_set = get_document_pccs(document)

        if "kile" in tasks:
            kile_matching = get_matches(
                predictions=docid_to_kile_predictions.get(document.docid, []),
                annotations=document.annotation.fields,
                pcc_set=pcc_set,
                iou_threshold=iou_threshold,
            )
            task_to_docid_to_matching["kile"][document.docid] = kile_matching

        if "lir" in tasks:
            lir_matching, _line_item_matching = get_lir_matches(
                predictions=docid_to_lir_predictions.get(document.docid, []),
                annotations=document.annotation.li_fields,
                pcc_set=pcc_set,
                iou_threshold=iou_threshold,
            )
            task_to_docid_to_matching["lir"][document.docid] = lir_matching

    return EvaluationReport(
        task_to_docid_to_matching=task_to_docid_to_matching,
        dataset_name=dataset.name,
        iou_threshold=iou_threshold,
    )


def compute_metrics(
    docid_to_matching: Mapping[str, FieldMatching]
) -> Dict[str, Union[int, float]]:
    """Compute different metrics for the given matchings between predictions and annotations."""
    ap = compute_average_precision(
        sorted_predictions_matched=_sort_predictions(docid_to_matching),
        total_annotations=sum(
            len(matching.annotations) for matching in docid_to_matching.values()
        ),
    )

    # Remove all predictions that were only for AP computation
    matchings_no_ap = [
        matching.filter(exclude_only_for_ap=True) for matching in docid_to_matching.values()
    ]
    total_predictions = sum(len(matching.predictions) for matching in matchings_no_ap)
    total_annotations = sum(len(matching.annotations) for matching in matchings_no_ap)

    true_positives = sum(len(matching.matches) for matching in matchings_no_ap)
    false_positives = sum(len(matching.false_positives) for matching in matchings_no_ap)
    false_negatives = sum(len(matching.false_negatives) for matching in matchings_no_ap)

    precision = true_positives / total_predictions if total_predictions else 0.0
    recall = true_positives / total_annotations if total_annotations else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "AP": ap,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
    }


def _validate_predictions(
    dataset: Dataset,
    docid_to_kile_predictions: Mapping[str, Sequence[Field]],
    docid_to_lir_predictions: Mapping[str, Sequence[Field]],
) -> None:
    """Run basic checks on the provided predictions."""
    metric_to_predictions = {
        "kile": docid_to_kile_predictions,
        "lir": docid_to_lir_predictions,
    }
    if all(
        0 == sum(len(predictions) for predictions in docid_to_predictions.values())
        for docid_to_predictions in metric_to_predictions.values()
    ):
        raise ValueError(
            "You need to provide at least one prediction for at least one of the tasks."
        )

    if any(
        pred.fieldtype is None
        for docid_to_predictions in metric_to_predictions.values()
        for predictions in docid_to_predictions.values()
        for pred in predictions
    ):
        raise ValueError("Some prediction is missing 'fieldtype'.")

    if any(
        pred.line_item_id is not None
        for predictions in docid_to_kile_predictions.values()
        for pred in predictions
    ):
        raise ValueError("Some KILE prediction has extra 'line_item_id'.")

    if any(
        pred.line_item_id is None
        for predictions in docid_to_lir_predictions.values()
        for pred in predictions
    ):
        raise ValueError("Some LIR prediction is missing 'line_item_id'.")

    for metric, docid_to_predictions in metric_to_predictions.items():
        have_scores = sum(
            sum(1 for f in fields if f.score is not None)
            for fields in docid_to_predictions.values()
        )
        if have_scores > 0 and have_scores < sum(
            len(fields) for fields in docid_to_predictions.values()
        ):
            raise ValueError("Either all or no predictions need to have scores")

        if have_scores > 0:
            max_ap_only_score = max(
                (
                    cast(float, pred.score)  # this is guaranteed by have_scores > 0
                    for predictions in docid_to_predictions.values()
                    for pred in predictions
                    if pred.use_only_for_ap and pred.score is not None
                ),
                default=0,
            )
            min_not_ap_only_score = max(
                (
                    cast(float, pred.score)  # this is guaranteed by have_scores > 0
                    for predictions in docid_to_predictions.values()
                    for pred in predictions
                    if not pred.use_only_for_ap and pred.score is not None
                ),
                default=1,
            )
            if max_ap_only_score > min_not_ap_only_score:
                logger.warning(
                    "Found a prediction with use_only_for_ap=True that has a higher score "
                    f"({max_ap_only_score}) than another prediction with use_only_for_ap=False "
                    f"({min_not_ap_only_score}). Note that all predictions with "
                    "use_only_for_ap=True will be used (matched, counted in AP) only after all of "
                    "the predictions with use_only_for_ap=False anyway."
                )

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


def _sort_predictions(docid_to_matching: Mapping[str, FieldMatching]) -> Sequence[bool]:
    """
    Collect and sort predictions from the given field matchings.

    Returns
    -------
    Indicator for each prediction whether it was matched, sorted by the criteria explained in
    `_get_prediction_sort_key`.
    """
    sort_key_prediction_matched: List[Tuple[PredictionSortKey, bool]] = []
    total_annotations = 0
    for docid, matching in docid_to_matching.items():
        for pred_i, (pred, gold) in enumerate(matching.ordered_predictions_with_match):
            sort_key_prediction_matched.append(
                (_get_prediction_sort_key(pred.normalized_score, pred_i, docid), gold is not None)
            )
        total_annotations += len(matching.annotations)

    return [
        matched
        for _sort_key, matched in sorted(sort_key_prediction_matched, key=operator.itemgetter(0))
    ]


def _get_prediction_sort_key(score: float, prediction_i: int, docid: str) -> PredictionSortKey:
    """
    Get a sort key for a prediction.

    For evaluation purposes, predictions are sorted by these criteria (sorted by importance):
    1.  Score from the highest to the lowest.
    2.  Original order in which the predictions were passed in.
    3.  The document id. Document id is hashed together with the prediction_i to make sure
        documents are not always sorted in the same order (for different prediction indices) which
        would make some documents more important for the evaluation than others.

    Parameters
    ----------
    score
        Prediction score (confidence).
    prediction_i
        The original rank of the prediction for the document as given on the input.
    docid
        Document ID

    Returns
    -------
    A tuple whose ordering corresponds to the criteria described above.
    """
    hashed_docid = hashlib.sha1(docid.encode())
    hashed_docid.update(prediction_i.to_bytes(8, "little"))
    return (-score, prediction_i, hashed_docid.hexdigest()[:16])
