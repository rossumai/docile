# DocILE: Document Information Localization and Extraction Benchmark
[![Tests](https://github.com/rossumai/docile/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rossumai/docile/actions/workflows/tests.yml)

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

Download dataset from [google drive](https://drive.google.com/file/d/1I4sf75dSEgnVEWE7MUZQX7BG98ivAYk6/view?usp=share_link) and unzip it into the `data/` folder.

## Pre-computed OCR

OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library.

If you wish to (re)generate OCR from scratch (e.g., on a different dataset), delete `DATASET_PATH/ocr` directory and install DocTR. On Mac OS X this can be done using `make install-with-ocr` (resp. `make install-with-ocr-m1` on M1 mac) or by following the official [installation instructions](https://github.com/mindee/doctr#installation).

# Example usage

Assuming you have your dataset in `data/docile/` with two splits `data/docile/train.json` and `data/docile/test.json`:

```python
from pathlib import Path
from docile.dataset import Dataset
from docile.evaluation import evaluate_dataset

DATASET_PATH = Path("data/docile")
# example: take only first 10 documents for debugging
dataset_train = Dataset.from_file("train", DATASET_PATH)[:10]

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

dataset_test = Dataset.from_file("test", DATASET_PATH)
docid_to_kile_predictions = {}
docid_to_lir_predictions = {}
for document in dataset_test:
    # kile_predictions = ... predict KILE fields
    # lir_predictions = ... predict LIR fields
    docid_to_kile_predictions[document.docid] = kile_predictions
    docid_to_lir_predictions[document.docid] = lir_predictions

evaluation_report = evaluate_dataset(dataset_test, docid_to_kile_predictions, docid_to_lir_predictions)
print(evaluation_report.print_report(include_same_text=True))
```
