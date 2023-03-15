# Dataset aiming to represent business documents. 

Organisations have multiple sources of knowledge: documentation, code, contacts, meetings, events, power points… And even each source may be diverse in format and on how to interact with it.

Proposing here written documentation as starting point, therefore containing only text.

Dataset currently based on:

(1) DocILE: Document Information Localization and Extraction Benchmark (research purposes) (https://github.com/rossumai/docile)

(2) Open4Business: An Open Access Dataset for Summarizing Business Documents (https://github.com/amanpreet692/Open4Business)

Table of Contents:
* [Download the DocILE dataset](#download-the-docile-dataset)
* [Installation of the python library 'docile'](#installation-of-docile)
* [Download the O4B dataset](#download-the-o4b-dataset)
* [Tutorials](tutorials/) 

## Download the DocILE dataset

First you need to obtain a secret token by following the instructions at https://docile.rossum.ai/. Then download and unzip the dataset by running:
```bash
./download_dataset.sh TOKEN annotated-trainval data/docile --unzip
./download_dataset.sh TOKEN synthetic data/docile --unzip
./download_dataset.sh TOKEN unlabeled data/docile --unzip
```

## Installation of 'docile'

Install the library with:
```shell
pip install docile-benchmark
```

## Download the O4B dataset

The current version of the dataset can be downloaded from: [O4B Download](https://drive.google.com/file/d/1w5mc6vxXrHIPRbRpoOxbUo8yTdVkW6l5/view?usp=sharing).

Steps to use the dataset:

 1. Download the zip from the URL given above and extract it.
 2. The extracted directory will contain 7 files - 1 source and 1 target file for each of the splits, namely train, dev and test. For instance, for training set the file names will be train.source and train.target. The additional file called refs.bib consist of the bibtex reference for the articles used for creating O4B. 
 3. In both the source and target files, each line represents 1 record. 

## Tutorials

I made interactive tutorials in Jupyter Notebooks for each dataset source:

* [DocILE dataset notebook](tutorials/Visualizing&#32;documents.ipynb) to load and visualize both real and synthetic documents + test the text extraction from the loaded pdfs.
* [O4B dataset notebook](tutorials/Open4Business&#32;Dataset&#32;Visualization.ipynb) to load and visualize the full-text articles and its summaries.
