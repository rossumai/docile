# DocILE: Document Information Localization and Extraction Benchmark
[![Tests](https://github.com/rossumai/docile/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rossumai/docile/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository to work with the [DocILE dataset and benchmark](https://docile.rossum.ai/), used in the DocILE'23 CLEF Lab and ICDAR Competition.
The competition deadline is on May 10, 2023 and comes with a **$9000 prize pool**.

The repository consists of:
* A python library, `docile`, making it easy to load the dataset, work with its annotations, pdfs and pre-computed OCR and run the evaluation.
* An interactive [dataset browser notebook](docile/tools/dataset_browser.ipynb) to visualize the document annotations, predictions and evaluation results.
* Training and inference scripts for [Baseline methods](baselines/) provided together with pretrained checkpoints and results on test and validation sets.
* [Tutorials](tutorials/) to get quickly started with the repo.

Table of Contents:
* [Download the dataset](#download-the-dataset)
* [Installation](#installation)
* [Predictions format and running evaluation](#predictions-format-and-running-evaluation)
* [Tasks and evaluation metrics](#tasks-and-evaluation-metrics)
* [Pre-computed OCR](#pre-computed-ocr)
* [Development instructions](#development-instructions)
* [Dataset and benchmark paper](#dataset-and-benchmark-paper)

## Download the dataset

First you need to obtain a secret token by following the instructions at https://docile.rossum.ai/. Then download and unzip the dataset by running:
```bash
./download_dataset.sh TOKEN annotated-trainval data/docile --unzip
./download_dataset.sh TOKEN synthetic data/docile --unzip
./download_dataset.sh TOKEN unlabeled data/docile --unzip
```

Run `./download_dataset.sh --help` for more options, including how to only show urls (to download
with a different tool than curl), how to download smaller unlabeled/synthetic chunks or unlabeled
dataset without pdfs (with pre-computed OCR only).

You can also work with the zipped datasets when you turn off image caching (check [Load and sample dataset](tutorials/load_and_sample_dataset.md) tutorial for details).

## Installation

### Option 1: Install as a library
Install the library with:
```shell
pip install docile-benchmark
```

To convert pdfs into images, the library uses https://github.com/Belval/pdf2image. On linux you might need to install:
```shell
apt install poppler-utils
```
And on macOS:

```shell
brew install poppler
```

Now you have all dependencies to work with the dataset annotations, pdfs, pre-comptued OCR and to run the evaluation. You can install extra dependencies by running the following (although using one of the provided dockerfiles, as explained below, might be easier in this case):
```shell
pip install "docile-benchmark[interactive]"
pip install "docile-benchmark[ocr]"
```

The first line installs additional dependencies allowing you to use the interactive dataset browser in [docile/tools/dataset_browser.py](docile/tools/dataset_browser.py) and the [tutorials](tutorials/). The second line let's you rerun the OCR predictions from scratch (e.g., if you'd like to run it with different parameters) but to make it work, you might need additional dependencies on your system. Check https://github.com/mindee/doctr for the installation instructions (for pytorch).

### Option 2: Use docker
There are two Dockerfiles available with the dependencies preinstalled:

* `Dockerfile` is a lighter, CPU-only, version with all necessary dependencies to use the dataset with the pre-computed OCR and interactive browser.
* `Dockerfile.gpu` has CUDA GPU support and contains additional dependencies needed to recompute OCR predictions from scratch (not needed for standard usage).

You can use docker compose to manage the docker images. First update the settings in `docker-compose.yml` and the port for jupyter lab in `.env`. Then build the image with:
```shell
docker compose build jupyter[-gpu]
```
where `jupyter` uses `Dockerfile` and `jupyter-gpu` uses `Dockerfile.gpu`. You can then start the jupyter server:
```shell
docker compose up -d jupyter[-gpu]
```

Jupyter lab can be then accessed at `https://127.0.0.1:${JUPYTER_PORT}` (retrieve the token from logs with `docker compose logs jupyter[-gpu]`). You can also login to the container with:
```shell
docker compose exec jupyter bash
```
After that run `poetry shell` to activate the virtual environment with the `docile` library and its dependencies installed.

## Predictions format and running evaluation

To evaluate predictions for tasks KILE or LIR on the test set, you have to make a submission to the benchmark on the [Robust Reading Competition](https://rrc.cvc.uab.es/?ch=26) website. To evaluate on the validation set, use the following command:
```bash
docile_evaluate \
  --task LIR \
  --dataset-path path/to/dataset[.zip] \
  --split val \
  --predictions path/to/predictions.json \
  --evaluate-x-shot-subsets "0,1-3,4+" \  # default, show evaluation for 0-shot, few-shot and many-shot layout clusters
  --evaluate-synthetic-subsets \  # optional, show evaluation on layout clusters with available synthetic data
  --evaluate-fieldtypes \  # optional, show breakdown per fieldtype
  --evaluate-also-text \  # optional, evaluate if the text prediction is correct
  --store-evaluation-result LIR_val_eval.json  # optional, it can be loaded in the dataset browser
```

Run `docile_evaluate --help` for more information on the options. You can also run `docile_print_evaluation_report --evaluation-result-path LIR_val_eval.json` to print the results of a previously computed evaluation. It is also possible to run evaluation or load the evaluation result directly from code which provides even more options, such as computing metrics just for a single document, specific subset of documents (i.e., documents belonging to the same layout cluster) etc. Check [docile/evaluation/evaluate.py](docile/evaluation/evaluate.py) module for more details.

Predictions need to be stored in a single json file (for each task separately) containing a mapping from `docid` to the predictions for that document, i.e.:
```json
{
    "docid1": [
        {
            "page": 0,
            "bbox": [0.2, 0.1, 0.4, 0.5],
            "fieldtype": "line_item_order_id",
            "line_item_id": 3,
            "score": 0.8,
            "text": "Order 38",
            "use_only_for_ap": true
        },
        "..."
    ],
    "docid2": [{"...": "..."}, "..."],
    "..."
}
```
Explanation of the individual fields of the predictions:
  * `page`: page index (from zero) the prediction belongs to
  * `bbox`: relative coordinates (from 0 to 1) representing the `left`, `top`, `right`, `bottom` sides of the bbox, respectively
  * `fieldtype`: the fieldtype (sometimes called category or key) of the prediction
  * `line_item_id`: ID of the line item. This should be a different number for each line item, the order does not matter. Omit for KILE predictions.
  * `score` [optional]: the confidence for this prediction, can be omitted (in that case predictions are taken in the order in which they are stored in the list)
  * `text` [optional]: text of the prediction, evaluated in a secondary metric only (when `--evaluate-also-text` is used)
  * `use_only_for_ap` [optional, default is False]: only use the prediction for AP metric computation, not for f1, precision and recall (useful for less confident predictions).

You can use [Field class](docile/dataset/field.py) to work with the predictions in code, in which case you can store them in a json file in the required format by using the `docile.dataset.store_predictions` function. You can also use the [BBox class](docile/dataset/bbox.py) to easily convert between relative and absolute coordinates (relative coordinates are used everywhere by default).

Use [docile/tools/print_results.py](docile/tools/print_results.py) to display results of your predictions formatted in a table (choosing from table styles like github, latex or csv).

## Tasks and evaluation metrics

The DocILE benchmark comes with two tracks, Key Information Localization and Extraction (KILE) and Line Item Recognition (LIR), as described in the [dataset and benchmark paper](#dataset-and-benchmark-paper). In both tracks, the goal is to localize key information of pre-defined categories, i.e., for each document, detect fields with their `fieldtype`, location (`page` and `bbox`) and optionally `text`. For LIR, fields have to be additionally grouped into line items. A Line Item (LI) is a tuple of fields (e.g., description, quantity, and price) describing a single object instance to be extracted, e.g., a row in a table.

Evaluated metrics are Average Precision (AP), F1, precision and recall, computed over all fields across all documents and field types (micro-average). A predicted and a ground truth field can be matched if they have the same `fieldtype` and if they are in the same location, which is decided by the overlap of their bounding boxes with the [provided OCR words](#pre-computed-ocr), described more precisely in the [dataset and benchmark paper](#dataset-and-benchmark-paper):

<img width="500" alt="Definition of Pseudo-Character-Centers and correct localization." src="https://user-images.githubusercontent.com/1220288/222190727-e851a1d9-9c91-438e-bb8d-7ca2066620a4.png">

For LIR, the fields have to additionally belong to corresponding Line Items. The mapping between `line_item_id` for predicted and ground truth fields is decided by taking the maximum matching which maximizes the total number of matched fields, when considering only fields with `use_only_for_ap=False` (because `use_only_for_ap=True` can be used to include also less confident predictions for AP computation which could negatively affect this matching).

For both KILE and LIR, the metrics are then computed like this:

* For each document page, find matching between predicted and ground truth fields. If more predicted fields match the same ground truth field, the one with higher `score` is used (but primarily the one with `use_only_for_ap=False`).
* Order all predictions by descending `score` (but again, predictions with `use_only_for_ap=True` come last). Break ties by the original order in which predictions were provided. See [evaluate.py](docile/evaluation/evaluate.py) for the precise ordering rules. Notice that the order does not influence F1, precision and recall but it influences AP.
* Compute F1, precision and recall by counting number of true positives and false positives/negatives and AP by iteratively adding predictions and updating precision and recall. We use the COCO style of AP where "gaps are filled", i.e., the precision-recall curve becomes non-increasing.

## Pre-computed OCR

Pre-computed OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library. On top of that, word boxes were snapped to text (check the code in [docile/dataset/document_ocr.py](docile/dataset/document_ocr.py)). These snapped word boxes are used in evaluation.

To get the OCR words for a single page, call `document.ocr.get_all_words(page)`, optionally passing `snapped=True` to get the snapped bounding boxes. In some rare cases, the list of words might be empty (when the page is blank). Also notice the predicted words are grouped into blocks. For some models it might be beneficial to re-order the words primarily by lines.

While it should not be needed, it is possible to (re)generate OCR from scratch (including the snapping) with the provided `Dockerfile.gpu`. Just delete `DATASET_PATH/ocr` directory and then access the ocr for each document and page with `document.ocr.get_all_words(page, snapped=True)`.

## Development instructions

For development, install [poetry](https://python-poetry.org/docs/) and run `poetry install`. Start a shell with the virtual environment activated with `poetry shell`. No other dependencies are needed to run pre-commit and the tests. It's recommended to use docker (as explained above) if you need the extra (interactive or ocr) dependencies.

Install pre-commit with `pre-commit install` (don't forget you need to prepend all commands with `poetry run ...` if you did not run `poetry shell` first).

Run tests by calling `pytest tests`.

## Dataset and benchmark paper
The dataset, the benchmark tasks and the evaluation criteria are described in detail in the [dataset paper](https://arxiv.org/abs/2302.05658). To cite the dataset, please use the following BibTeX entry:
```
@misc{simsa2023docile,
    title={{DocILE} Benchmark for Document Information Localization and Extraction},
    author={{\v{S}}imsa, {\v{S}}t{\v{e}}p{\'a}n and {\v{S}}ulc, Milan and U{\v{r}}i{\v{c}}{\'a}{\v{r}}, Michal and Patel, Yash and Hamdi, Ahmed and Koci{\'a}n, Mat{\v{e}}j and Skalick{\`y}, Maty{\'a}{\v{s}} and Matas, Ji{\v{r}}{\'\i} and Doucet, Antoine and Coustaty, Micka{\"e}l and Karatzas, Dimosthenis},
    url = {https://arxiv.org/abs/2302.05658},
    journal={arXiv preprint arXiv:2302.05658},
    year={2023}
}
```
