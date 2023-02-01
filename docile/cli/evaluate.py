from pathlib import Path
from typing import List, Optional, Sequence

import click

from docile.dataset import CachingConfig, Dataset, load_predictions
from docile.evaluation import (
    EvaluationResult,
    NamedRange,
    evaluate_dataset,
    get_evaluation_subsets,
)


class NamedRangesParamType(click.ParamType):
    """Parameter for list of ranges."""

    name = "named_range"

    def convert(self, value: str, param: click.Option, ctx: click.Context) -> List[NamedRange]:
        """
        Convert the input value into list of named ranges.

        Parameters
        ----------
        value
            List of ranges in the format 'range1,range2,...' One range can be one of 'x', 'x-y'
            or 'x+', corresponding to ranges [x, x], [x, y] and [x, infinity).
        param
            Click option.
        ctx
            Click context.

        Returns
        -------
        named_ranges
            List of named ranges, i.e., tuples of string (range name) and range. Range is either
            Tuple[int, int] or Tuple[int, None] if there is no upper bound.
        """
        if value == "":
            return []
        parsed_named_ranges: List[NamedRange] = []
        for range_name in value.split(","):
            size_range = range_name.split("-")
            if range_name.isdigit():
                parsed_named_ranges.append((range_name, (int(range_name), int(range_name))))
            elif range_name[-1] == "+" and range_name[:-1].isdigit():
                parsed_named_ranges.append((range_name, (int(range_name[:-1]), None)))
            elif len(size_range) == 2 and all(x.isdigit() for x in size_range):
                parsed_named_ranges.append((range_name, (int(size_range[0]), int(size_range[1]))))
            else:
                self.fail(
                    f"Cannot parse range name {range_name}. Options are 'x', 'x-y' or 'x+'",
                    param,
                    ctx,
                )
        return parsed_named_ranges


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
    "--evaluate-x-shot-subsets",
    type=NamedRangesParamType(),
    default="0,1-3,4+",
    help=(
        "evaluate on subsets of x-shot layout clusters. Pass empty string to turn it off. "
        "Format: 'range1,range2,...' where range is one of 'x', 'x-y' or 'x+'."
    ),
    show_default=True,
)
@click.option(
    "--evaluate-synthetic-subsets",
    is_flag=True,
    help="if used, evaluate also on subsets belonging to layout clusters with synthetic data",
)
@click.option(
    "--evaluate-fieldtypes",
    is_flag=True,
    help="show breakdown per fiedltype",
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
    evaluate_x_shot_subsets: Sequence[NamedRange],
    evaluate_synthetic_subsets: bool,
    evaluate_fieldtypes: bool,
    evaluate_also_text: bool,
    primary_metric_only: bool,
) -> None:
    dataset = Dataset(split, dataset_path, cache_images=CachingConfig.OFF)
    docid_to_predictions = load_predictions(predictions)
    subsets = get_evaluation_subsets(dataset, evaluate_x_shot_subsets, evaluate_synthetic_subsets)
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
        report = evaluation_result.print_report(
            subsets=subsets,
            include_fieldtypes=evaluate_fieldtypes,
            include_same_text=evaluate_also_text,
        )
        print(report)  # noqa T201


@click.command("Print evaluation previously done on DocILE dataset")
@click.option(
    "--evaluation-result-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="path to the json file with evaluation result",
)
@click.option(
    "--evaluate-x-shot-subsets",
    type=NamedRangesParamType(),
    default="0,1-3,4+",
    help=(
        "evaluate on subsets of x-shot layout clusters. Pass empty string to turn it off. "
        "Format: 'range1,range2,...' where range is one of 'x', 'x-y' or 'x+'."
    ),
    show_default=True,
)
@click.option(
    "--evaluate-synthetic-subsets",
    is_flag=True,
    help="if used, evaluate also on subsets belonging to layout clusters with synthetic data",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "if --evaluate-x-shot-subsets (used by default) or --evaluate-synthetic-subsets are used, "
        "you need to pass a path to the dataset"
    ),
)
@click.option(
    "--evaluate-fieldtypes",
    is_flag=True,
    help="show breakdown per fiedltype",
)
@click.option(
    "--evaluate-also-text",
    is_flag=True,
    help="if used, also show metrics that require exact text match for predictions",
)
def print_evaluation_report(
    evaluation_result_path: Path,
    evaluate_x_shot_subsets: Sequence[NamedRange],
    evaluate_synthetic_subsets: bool,
    dataset_path: Optional[Path],
    evaluate_fieldtypes: bool,
    evaluate_also_text: bool,
) -> None:
    evaluation_result = EvaluationResult.from_file(evaluation_result_path)
    subsets: List[Dataset] = []
    if len(evaluate_x_shot_subsets) > 0 or evaluate_synthetic_subsets:
        if dataset_path is None:
            raise ValueError(
                "You need to provide --dataset-path when --evaluate-x-shot-subsets (used by "
                "default) or --evaluate-synthetic-subsets are used."
            )
        test_split_name: Optional[str] = None
        for split_name in ["test", "val"]:
            if evaluation_result.dataset_name.endswith(split_name):
                test_split_name = split_name
        if test_split_name is None:
            raise ValueError(
                f"Unknown dataset {evaluation_result.dataset_name}, cannot find x-shot subsets"
            )
        test = Dataset(
            test_split_name, dataset_path, load_ocr=False, cache_images=CachingConfig.OFF
        )
        subsets = get_evaluation_subsets(test, evaluate_x_shot_subsets, evaluate_synthetic_subsets)
    report = evaluation_result.print_report(
        subsets=subsets,
        include_fieldtypes=evaluate_fieldtypes,
        include_same_text=evaluate_also_text,
    )
    print(report)  # noqa T201


if __name__ == "__main__":
    evaluate()
