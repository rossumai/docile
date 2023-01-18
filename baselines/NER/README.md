# NER baselines

Our baselines build upon Named Entity Recognition (NER). Here we provide the code to construct the targets for NER from the DocILE dataset, train RoBERTa and LayoutLMv3 models (including RoBERTa pretraining on the unlabeled part of dataset) and run inference, combining the model predictions for individual OCR tokens into the predicted fields. See [dataset and benchmark paper](../../README.md#dataset-and-benchmark-paper) for more details.

Notice that some checkpoints and predictions on validation set are provided, see [../README.md](../README.md) for more info.

To run training (including pretraining) and inference, follow the [run_training.sh](run_training.sh), resp. [run_inference.sh](run_inference.sh) scripts. They assume you downloaded the dataset and baselines checkpoints according to the instructions in [main readme file](../../README.md#download-the-dataset) and [baselines readme file](../README.md#download-checkpoints-and-predictions).
