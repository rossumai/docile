# DocILE baselines

The DocILE benchmark comes with several baselines to help you get started and to have something to compare your models against. The baselines are described in the [dataset and benchmark paper](../README.md#dataset-and-benchmark-paper).

## Download checkpoints and predictions

> :warning: The predictions on the validation set are not yet included, they will be added in the next few days.

Checkpoints of various trained models are provided with predictions on the validation set. You can download them with the same [download script](../download_dataset.sh) that is provided for downloading the dataset in the root of this repository.

First you need to obtain a secret token by following the instructions at https://docile.rossum.ai/. Then run this from the root of this repository:
```bash
./download_dataset.sh TOKEN BASELINE data/baselines --unzip
```
Here TOKEN is the secret token you obtained and BASELINE is one of the baselines available for download from the list below. You can also download all baselines with one command:
```bash
./download_dataset.sh TOKEN baselines data/baselines --unzip
```

The baselines available for download are below and contain the PyTorch checkpoint and predictions on `val` set (the two models pretrained on the `unlabeled` dataset do not contain any predictions).
* `baselines-roberta-base`: RoBERTa model trained from the `roberta-base` Hugging Face dataset.
* `baselines-roberta-ours`: RoBERTa model trained from our pretrained model (`baselines-roberta-pretraining`)
* `baselines-layoutlmv3-ours`: LayoutLMv3 model trained from our pretrained model (`baselines-layoutlmv3-pretraining`)
* `baselines-detr`: DETR model trained to detect tables and Line Items for the LIR task.
* `baselines-roberta-pretraining`: Unsupervised pre-training for RoBERTa on the `unlabeled` set.
* `baselines-layoutlmv3-pretraining`: Unsupervised pre-training for LayoutLMv3 on the `unlabeled` set.

## Code to run trainings and inference

We provide code to reproduce all results that can be found in the [paper](../README.md#dataset-and-benchmark-paper). Some small errors were found and fixed in the training and inference scripts that influence the numbers presented in the paper v1 version. These should be fixed in an updated version that we aim to make available on arXiv in April 2023.

The code is structured into three subfolders:
* [NER](NER/) contains most of the baselines code, including training code for RoBERTa, LayoutLMv3 and RoBERTa pretraining, and the inference code.
* [layoutlmv3_pretraing](layoutlmv3_pretrain/) [coming soon] contains code for LayoutLMv3 pretraining
* [table-transformer](table-transformer/) [coming soon] contains code for DETR used for table and Line Item detection.

## Results on the validation set

Coming soon, results of the baselines on the validation set will appear here.
