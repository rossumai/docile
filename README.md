# DocILE: Document Information Localization and Extraction Benchmark

Repository to work with the [DocILE dataset and benchmark](https://docile.rossum.ai/), used in the DocILE'23 CLEF Lab and ICDAR Competition.

The repository consists of:
* A python library, `docile`, making it easy to load the dataset, work with its annotations, pdfs and pre-computed OCR and run the evaluation.

Table of Contents:
* [Download the dataset](#download-the-dataset)
* [Installation](#installation)
* [Tutorials](tutorials/) 

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



