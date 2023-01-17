from pathlib import Path
from typing import Optional

import click

from docile.dataset import CachingConfig, Dataset, load_predictions
from docile.evaluation import EvaluationResult, evaluate_dataset


@click.command("Evaluate predictions on DocILE dataset")
@click.option(
    "-t",
    "--task",
    type=click.Choice(["KILE", "LIR"]),
    required=True,
    help="whether to evaluate KILE or LIR task",
)
@click.option(
    "-d",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="path to the zip with dataset or unzipped dataset",
)
@click.option(
    "-s",
    "--split",
    type=str,
    required=True,
    default="val",
    help="name of the dataset split to evaluate on",
    show_default=True,
)
@click.option(
    "-p",
    "--predictions",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="path to the json file with predictions",
)
@click.option(
    "--store-evaluation-result",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="path to a json file where to store the evaluation result",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=1.0,
    help="IoU threshold for PCC matching, can be useful for experiments",
)
@click.option(
    "--evaluate-also-text",
    is_flag=True,
    help="if used, also show metrics that require exact text match for predictions",
)
@click.option(
    "--primary-metric-only",
    is_flag=True,
    help="if used, the script prints only the primary metric instead of a full evaluation report",
)
def evaluate(
    task: str,
    dataset_path: Path,
    split: str,
    predictions: Path,
    store_evaluation_result: Optional[Path],
    iou_threshold: float,
    evaluate_also_text: bool,
    primary_metric_only: bool,
) -> None:
    dataset = Dataset(split, dataset_path, cache_images=CachingConfig.OFF)
    docid_to_predictions = load_predictions(predictions)
    if task == "KILE":
        evaluation_result = evaluate_dataset(
            dataset,
            docid_to_kile_predictions=docid_to_predictions,
            docid_to_lir_predictions={},
            iou_threshold=iou_threshold,
        )
    elif task == "LIR":
        evaluation_result = evaluate_dataset(
            dataset,
            docid_to_kile_predictions={},
            docid_to_lir_predictions=docid_to_predictions,
            iou_threshold=iou_threshold,
        )
    else:
        raise ValueError(f"Unknown task {task}")

    if store_evaluation_result is not None:
        evaluation_result.to_file(store_evaluation_result)

    if primary_metric_only:
        metric_value = evaluation_result.get_primary_metric(task.lower())
        print(metric_value)  # noqa T201
    else:
        report = evaluation_result.print_report(include_same_text=evaluate_also_text)
        print(report)  # noqa T201


@click.command("Print evaluation previously done on DocILE dataset")
@click.option(
    "--evaluation-result-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="path to the json file with evaluation result",
)
@click.option(
    "--evaluate-also-text",
    is_flag=True,
    help="if used, also show metrics that require exact text match for predictions",
)
def print_evaluation_report(
    evaluation_result_path: Path,
    evaluate_also_text: bool,
) -> None:
    evaluation_result = EvaluationResult.from_file(evaluation_result_path)
    report = evaluation_result.print_report(include_same_text=evaluate_also_text)
    print(report)  # noqa T201


if __name__ == "__main__":
    evaluate()
