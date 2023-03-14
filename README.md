# Dataset based on DocILE: Document Information Localization and Extraction Benchmark (research purposes)

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

## Installation

Install the library with:
```shell
pip install docile-benchmark
```

## Tutorials

I made a tutorial in a Jupyter Notebook to visualize both real and synthetic documents.
