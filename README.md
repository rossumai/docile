# DocILE: Document Information Localization and Extraction Benchmark
[![Tests](https://github.com/rossumai/docile/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rossumai/docile/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository to work with the [DocILE dataset and benchmark](https://docile.rossum.ai/), used in the DocILE'23 CLEF Lab and ICDAR Competition.
The competition deadline is on May 10, 2023 and comes with a **$9000 prize pool**.

The repository consists of:
* A python library, `docile`, making it easy to load the dataset, work with its annotations, pdfs and pre-computed OCR and run the evaluation.
* An interactive [dataset browser notebook](docile/tools/dataset_browser.ipynb) to visualize the document annotations, predictions and evaluation results.
* Baseline methods (will appear soon).

Table of Contents:
* [Download the dataset](#download-the-dataset)
* [Installation](#installation)
* [Predictions format and running evaluation](#predictions-format-and-running-evaluation)
* [Pre-computed OCR](#pre-computed-ocr)
* [Development instructions](#development-instructions)

Also check [Tutorials](tutorials/) to get quickly started with the repo.

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
To install just the library, download the wheel from the [latest release on github](https://github.com/rossumai/docile/releases) and run:
```shell
pip install docile-0.1.0-py3-none-any.whl
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
pip install "docile-0.1.0-py3-none-any.whl[interactive]"
pip install "docile-0.1.0-py3-none-any.whl[ocr]"
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

To evaluate predictions for tasks KILE or LIR, use the following command:
```bash
docile_evaluate \
  --task LIR \
  --dataset-path path/to/dataset[.zip] \
  --split val \
  --predictions path/to/predictions.json \
  --evaluate-also-text \  # optional
  --store-evaluation-result LIR_val_eval.json  # optional, it can be loaded in the dataset browser
```

Run `docile_evaluate --help` for more information on the options. You can also run `docile_print_evaluation_report --evaluation-result-path LIR_val_eval.json` to print the results of a previously computed evaluation.

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

You can use `docile.dataset.store_predictions` to store predictions represented with the `docile.dataset.Field` class to a json file with the required format.

## Pre-computed OCR

Pre-computed OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library. On top of that, word boxes were snapped to text (check the code in [docile/dataset/document_ocr.py](docile/dataset/document_ocr.py)). These snapped word boxes are used in evaluation (description of the evaluation is coming soon).

While this should not be needed, it is possible to (re)generate OCR from scratch (including the snapping) with the provided `Dockerfile.gpu`. Just delete `DATASET_PATH/ocr` directory and then access the ocr for each document and page with `doc.ocr.get_all_words(page, snapped=True)`.

## Development instructions

For development, install [poetry](https://python-poetry.org/docs/) and run `poetry install`. Start a shell with the virtual environment activated with `poetry shell`. No other dependencies are needed to run pre-commit and the tests. It's recommended to use docker (as explained above) if you need the extra (interactive or ocr) dependencies.

Install pre-commit with `pre-commit install` (don't forget you need to prepend all commands with `poetry run ...` if you did not run `poetry shell` first).

Run tests by calling `pytest tests`.
