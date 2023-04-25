# DocILE baselines

The DocILE benchmark comes with several baselines to help you get started and to have something to compare your models against. The baselines are described in the [dataset and benchmark paper](../README.md#dataset-and-benchmark-paper).

## Download checkpoints and predictions

**Info:** New checkpoints were provided on April 27, 2023. Older checkpoints can be downloaded temporarily with the `download_dataset.sh` script by using the string `baselines-20230315` instead of `baselines`. To see results of the previously shared checkpoints on the test and validation sets, check the history of this file. The new trainings were all run with the currently present version of the code and with hyperparameters set to the same values (for NER-based models).

**Warning:** Furthermore a [bug was fixed](https://github.com/rossumai/docile/pull/64) on April 24, 2023 in the NER baselines training code (inference and pretraining were not affected). This bug did not affect any of the shared baselines as it was introduced during refactorings after trainings of the originally shared checkpoints have already finished and before the new trainings have started.

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
* `baselines-roberta-pretraining`: Unsupervised pre-training for RoBERTa on the `unlabeled` set.
* `baselines-layoutlmv3-pretraining`: Unsupervised pre-training for LayoutLMv3 on the `unlabeled` set.
* `baselines-detr`: DETR model trained to detect tables and Line Items for the LIR task. Provided predictions on the validation set include table and LI predictions and also predictions of `roberta-base` model that uses these DETR predictions during inference.
* `baselines-roberta-base-with-synthetic-pretraining`, `baselines-roberta-ours-with-synthetic-pretraining`, `baselines-layoutlmv3-ours-with-synthetic-pretraining`: Versions of the RoBERTa and LayoutLMv3 models that were first pretrained on the synthetic data before being trained on the annotated data. All three baselines come with two checkpoints, one after synthetic pretraining and one after the final training, and predictions on validation set after the final training.

## Code to run trainings and inference

We provide code to reproduce all results that can be found in the [paper](../README.md#dataset-and-benchmark-paper). Some small errors were found and fixed in the training and inference scripts that influence the numbers presented in the paper v1 version. These should be fixed in an updated version that we aim to make available on arXiv in April 2023. You can see the updated comparison of the provided baselines in the following section.

The code is structured into three subfolders:
* [NER](NER/) contains most of the baselines code, including training code for RoBERTa, LayoutLMv3 and RoBERTa pretraining, and the inference code.
* [layoutlmv3_pretraing](layoutlmv3_pretrain/) contains code for LayoutLMv3 pretraining.
* [table-transformer](table-transformer/) contains code for DETR used for table and Line Item detection.

## Results of the provided baselines

While results on the test set better compare which models work better, results on the validation set can be used to easily compare your models against the baselines, since annotations for test set are not publicly available and the only way to run the evaluations on the test set is to make a submission to the benchmark on the [RRC website](https://rrc.cvc.uab.es/?ch=26).

### KILE

The main benchmark metric for KILE is `AP`, the best results on AP are bold in the table.

| model                                      | <ins>val-AP</ins>   |   val-F1 |   val-precision |   val-recall | <ins>test-AP</ins>   |   test-F1 |   test-precision |   test-recall |
|--------------------------------------------|---------------------|----------|-----------------|--------------|----------------------|-----------|------------------|---------------|
| roberta_base                               | 0.552               |    0.688 |           0.681 |        0.694 | 0.534                |     0.664 |            0.658 |         0.671 |
| roberta_ours                               | 0.537               |    0.671 |           0.661 |        0.682 | 0.515                |     0.645 |            0.634 |         0.656 |
| layoutlmv3_ours                            | 0.513               |    0.657 |           0.651 |        0.662 | 0.507                |     0.639 |            0.636 |         0.641 |
| roberta_base_with_synthetic_pretraining    | **0.566**           |    0.689 |           0.680 |        0.698 | **0.539**            |     0.664 |            0.659 |         0.669 |
| roberta_ours_with_synthetic_pretraining    | 0.542               |    0.677 |           0.672 |        0.682 | 0.527                |     0.652 |            0.648 |         0.656 |
| layoutlmv3_ours_with_synthetic_pretraining | 0.532               |    0.674 |           0.680 |        0.668 | 0.512                |     0.655 |            0.662 |         0.648 |

### LIR

The main benchmark metric for LIR is `F1`, the best results on F1 are bold in the table.

| model                                      |   val-AP | <ins>val-F1</ins>   |   val-precision |   val-recall |   test-AP | <ins>test-F1</ins>   |   test-precision |   test-recall |
|--------------------------------------------|----------|---------------------|-----------------|--------------|-----------|----------------------|------------------|---------------|
| roberta_base                               |    0.552 | 0.688               |           0.709 |        0.668 |     0.576 | 0.686                |            0.695 |         0.678 |
| roberta_ours                               |    0.538 | 0.662               |           0.676 |        0.649 |     0.570 | 0.686                |            0.693 |         0.678 |
| layoutlmv3_ours                            |    0.546 | 0.666               |           0.688 |        0.645 |     0.531 | 0.661                |            0.682 |         0.641 |
| roberta_base_with_synthetic_pretraining    |    0.567 | **0.701**           |           0.721 |        0.683 |     0.583 | **0.698**            |            0.710 |         0.687 |
| roberta_ours_with_synthetic_pretraining    |    0.549 | 0.682               |           0.703 |        0.662 |     0.559 | 0.675                |            0.696 |         0.655 |
| layoutlmv3_ours_with_synthetic_pretraining |    0.564 | 0.681               |           0.704 |        0.659 |     0.582 | 0.691                |            0.709 |         0.673 |
| roberta_base_detr_table                    |    0.553 | 0.682               |           0.719 |        0.648 |     0.560 | 0.682                |            0.706 |         0.660 |
| roberta_base_detr_tableLI                  |    0.427 | 0.613               |           0.661 |        0.572 |     0.407 | 0.594                |            0.632 |         0.560 |
