# DocILE: Document Information Localization and Extraction Benchmark
[![Tests](https://github.com/rossumai/docile/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rossumai/docile/actions/workflows/tests.yml)

# Evaluate predictions

To evaluate predictions for tasks KILE or LIR, use the following command:
```bash
poetry run evaluate --task KILE --dataset-path path/to/dataset[.zip] --split val --predictions path/to/predictions.json --evaluate-also-text
```

Run `poetry run evaluate --help` for more information on the options.

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

# Development instructions

## Installation

The following installation assumes you want to use the pre-computed OCR. If you want to (re)compute the OCR, you need to do additional installation steps described in [Pre-computed OCR](#pre-computed-ocr) section or [Run in docker](#run-in-docker).

TLDR: On Mac OS X you can simply use:
```bash
make install
```

Otherwise you can follow these steps:

- Ubuntu
```bash
apt-get install poppler-utils -y
```
- Mac OS X:
```bash
brew install poppler
```

We use https://python-poetry.org/docs/ for dependency management. Follow the official instructions to install Poetry. You can use `brew install poetry` on Mac OS X. To install the poetry dependencies, use:

```bash
poetry install --with test --with dev --without doctr
```

Note that you can use `poetry shell` to spawn a new shell with this virtual environment activated.


## Run in docker
You can use the provided Dockerfile and docker-compose with docile repo installed, including DocTR
(see [Pre-computed OCR](#pre-computed-ocr) for more info on the re-computing OCR). Build it with
`docker-compose build` and start it with `docker-compose up -d`. You can then access jupyterlab at
`https://127.0.0.1:${JUPYTER_PORT}` (the port is defined in  `.env/`) or login to the container
with:
```docker-compose exec --privileged -it docile-jupyter-1 bash```

**Note:** docker needs to be run with `--privileged` flag when running DocTR because otherwise
operation `pkey_mprotect` (which is called somewhere inside) is not allowed.

If your data is not `data/` do not forget to mount it to the docker container.

## Running Tests

```bash
make test
```

# Data

Download dataset from [google drive](https://drive.google.com/file/d/1I4sf75dSEgnVEWE7MUZQX7BG98ivAYk6/view?usp=share_link) and unzip it into the `data/` folder or work with the zipped dataset directly in a read-only mode (i.e., caching on disk for images must be turned off).

## Pre-computed OCR

OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library.

If you wish to (re)generate OCR from scratch (e.g., on a different dataset), delete `DATASET_PATH/ocr` directory and install DocTR. On Mac OS X this can be done using `make install-with-ocr` (resp. `make install-with-ocr-m1` on M1 mac) or by following the official [installation instructions](https://github.com/mindee/doctr#installation).

# Example usage

Assuming you have your dataset in `data/docile/` with two splits `data/docile/train.json` and `data/docile/test.json`:

```python
from docile.dataset import Dataset
from docile.evaluation import evaluate_dataset

# example: take only 10 documents for debugging
dataset_train = Dataset("train", "data/docile").sample(10, seed=42)
# equivalently you can load the dataset directly from zip
dataset_train = Dataset("train", "data/real.zip").sample(10, seed=42)

for document in dataset_train:
    kile_fields = document.annotation.fields
    li_fields = document.annotation.li_fields
    for page in range(document.page_count):
        img = document.page_image(page)
        kile_fields_page = [field for field in kile_fields if field.page == page]
        li_fields_page = [field for field in li_fields if field.page == page]
        ocr = document.ocr.get_all_words(page)
        # ...Add to training set...

# ...Train here...

dataset_test = Dataset("test", "data/docile")
docid_to_kile_predictions = {}
docid_to_lir_predictions = {}
for document in dataset_test:
    # kile_predictions = ... predict KILE fields
    # lir_predictions = ... predict LIR fields
    docid_to_kile_predictions[document.docid] = kile_predictions
    docid_to_lir_predictions[document.docid] = lir_predictions

evaluation_result = evaluate_dataset(dataset_test, docid_to_kile_predictions, docid_to_lir_predictions)
print(evaluation_result.print_report(include_same_text=True))
```

When working with a large dataset, such as the unlabeled dataset, do not load the whole dataset to memory:

```python
from docile.dataset import CachingConfig, Dataset

dataset_unlabeled = Dataset(
    split_name="pretraining-all",
    dataset_path="data/docile-unlabeled",
    load_annotations=False,
    load_ocr=False,
    cache_images=CachingConfig.OFF,
)
for document in dataset_unlabeled:
    # temporarily cache document resources in memory.
    with document:
        # process document here
```
