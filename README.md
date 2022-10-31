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

# Development instructions
We use https://python-poetry.org/docs/ for dependency management. Follow the official instructions to install Poetry. To install the dependencies, use:

```bash
poetry install
```

Note that you can use `poetry shell` to spawn a new shell with this virtual environment activated.

You will also need https://pre-commit.com/ for pre-commit checks. If you have done the steps above, it is already installed in your virtual environment. To activate the pre-commit, use:

```bash
pre-commit install
```


# Data
TODO

## Pre-computed OCR

OCR is provided with the dataset, it is pre-computed using DocTR.

If you wish to generate OCR from scratch (e.g., on a different dataset), delete `DATASET\_PATH/ocr` directory and install DocTR by following the tutorial at https://github.com/mindee/doctr#installation, or by using these instructions for Mac:

- Mac OS X:
```bash
brew install cairo pango gdk-pixbuf libffi

poetry install "python-doctr[tf]"
```
