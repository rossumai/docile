import argparse
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple, Union

from tabulate import tabulate

from docile.evaluation import EvaluationResult
from docile.evaluation.evaluate import TASK_TO_PRIMARY_METRIC_NAME


def _highlight_best_numbers(
    main_metric: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Union[str, int]]],
    tablefmt: str,
    floatfmt: str,
) -> Tuple[List[str], List[List[Union[str, int]]]]:
    """Return updated headers and rows, highlighting the main metric with its best numbers."""
    if tablefmt != "github":
        raise NotImplementedError("Highlight only works for github style tables")
    main_metric_col_is = [i for i, h in enumerate(headers) if h.endswith(main_metric)]
    headers_task = list(headers)
    rows_task = [list(row) for row in rows]
    for col_i in main_metric_col_is:
        headers_task[col_i] = f"<ins>{headers_task[col_i]}</ins>"  # underline in github
        max_i = max(range(len(rows_task)), key=lambda i: rows_task[i][col_i])
        for i in range(len(rows_task)):
            rows_task[i][col_i] = f"{rows_task[i][col_i]:{floatfmt}}"
        rows_task[max_i][col_i] = f"**{rows_task[max_i][col_i]}**"
    return headers_task, rows_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help=(
            "Path to directory with evaluation results of your models. It should contain a "
            "subdirectory for each model with {split}_results_KILE.json and/or "
            "{split}_results_LIR.json files."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help='you can pass multiple splits separated by commas, e.g., --split="val,test"',
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="github",
        help="table format such as 'github' or 'latex', see `tabulate` package for more options",
    )
    parser.add_argument("--floatfmt", type=str, default=".3f")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="models to include in the table (in the given order), separated by commas",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="show counts of True Positive and False Positive/Negative predictions",
    )
    parser.add_argument(
        "--highlight-best-numbers",
        action="store_true",
        help=(
            "highlight the best numbers for the main metric, only implemented for "
            "--table-format=github"
        ),
    )
    args = parser.parse_args()

    splits = args.split.split(",")

    metric_names = ["AP", "f1", "precision", "recall"]
    if args.show_counts:
        metric_names.extend(["TP", "FP", "FN"])
    headers = ["model"]
    for split in splits:
        prefix = f"{split}-" if len(splits) > 1 else ""
        headers.extend([f"{prefix}{m}" for m in metric_names])

    rows = {"KILE": [], "LIR": []}
    models_paths = list(args.predictions_dir.iterdir())
    if args.models != "":
        models_paths = [args.predictions_dir / m for m in args.models.split(",")]
    for model_dir in models_paths:
        for task in ["KILE", "LIR"]:
            row = [model_dir.name]
            for split in splits:
                results_path = model_dir / f"{split}_results_{task}.json"
                metrics: Mapping[str, Union[str, float]] = {m: "-" for m in metric_names}
                if results_path.exists():
                    eval_result = EvaluationResult.from_file(results_path)
                    metrics = eval_result.get_metrics(task.lower())
                row.extend([metrics[m] for m in metric_names])
            rows[task].append(row)

    report = []
    for task in ["KILE", "LIR"]:
        headers_task, rows_task = headers, rows[task]
        if args.highlight_best_numbers:
            main_metric = TASK_TO_PRIMARY_METRIC_NAME[task.lower()]
            headers_task, rows_task = _highlight_best_numbers(
                main_metric, headers_task, rows_task, args.tablefmt, args.floatfmt
            )
        report.append(task)
        report.append("=" * len(task))
        report.append("")
        table = tabulate(rows_task, headers_task, tablefmt=args.tablefmt, floatfmt=args.floatfmt)
        report.extend(table.splitlines())
        report.append("")

    print("\n".join(report))  # noqa T201
