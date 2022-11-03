# DocILE
[![Tests](https://github.com/rossumai/docile/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rossumai/docile/actions/workflows/tests.yml) DocILE: Document Information Localization and Extraction Benchmark

# Installation
You need to install some dependencies for PDF handling:

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
poetry install
```

Note that you can use `poetry shell` to spawn a new shell with this virtual environment activated.

# Development instructions

## Installation

TLDR: If you are on Intel Mac OS X just use the provided makefile to install all dependencies.
```bash
make install
```

## Running Tests

```bash
make test
```

# Data

Download dataset from [google drive](https://drive.google.com/file/d/1I4sf75dSEgnVEWE7MUZQX7BG98ivAYk6/view?usp=share_link) and unzip it into the `data/` folder.

## Pre-computed OCR

OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library.

If you wish to generate OCR from scratch (e.g., on a different dataset), delete `DATASET_PATH/ocr` directory and install DocTR by following the [tutorial](https://github.com/mindee/doctr#installation).

# Example usage

Assuming you have your dataset in `data/docile/` with two splits `data/docile/train.json` and
`data/docile/test.json` (you might have to make this split if working with the dev set):

```python
from pathlib import Path
from docile.dataset import Dataset
from docile.evaluation.evaluate import Metric, evaluate_dataset

DATASET_PATH = Path("data/docile")
dataset_train = Dataset.from_file("train", DATASET_PATH)

for document in dataset_train:
    kile_fields = document.annotation.fields
    li_fields = document.annotation.li_fields
    for page in range(document.page_count):
        img = document.page_image(page)
        kile_fields_page = [field for field in kile_fields if k.page == page]
        li_fields_page = [field for field in li_fields if k.page == page]
        ocr = document.ocr.get_all_words(page)
        # ...Add to training set...

# ...Train here...

dataset_test = Dataset.from_file("test", DATASET_PATH)
docid_to_kile_predictions = {}
for document in dataset_test:
    # kile_predictions = ... predict kile_fields
    docid_to_kile_predictions[document.docid] = kile_predictions

average_precision = evaluate_dataset(dataset_test, docid_to_kile_predictions, Metric.KILE)
print(f"Test AP={average_precision}")
```
