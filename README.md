# DocILE
DocILE: Document Information Localization and Extraction Benchmark

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

## Pre-computed OCR

OCR is provided with the dataset. The prediction was done using the [DocTR](https://github.com/mindee/doctr) library.

If you wish to generate OCR from scratch (e.g., on a different dataset), delete `DATASET_PATH/ocr` directory and install DocTR by following the [tutorial](https://github.com/mindee/doctr#installation).
