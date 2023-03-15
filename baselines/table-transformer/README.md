# table-transformer baseline

The original repository: https://github.com/microsoft/table-transformer

We use the checkpoints and code from HuggingFace:
https://huggingface.co/microsoft/table-transformer-detection

As table-transformer is simply a DETR pretrained on [PubTables-1M](https://github.com/microsoft/table-transformer) and pretraining on additional document data is not allowed for DocILE, we start the training from a general [DETR checkpoint](https://huggingface.co/facebook/detr-resnet-50).

## Running training and evaluation

We use [pytorch-lightning](https://pytorch-lightning.readthedocs.io/) CLI for running the experiments:

```
CUDA_VISIBLE_DEVICES=0 poetry run python main.py fit --config table_detection_config.yaml
```

```
CUDA_VISIBLE_DEVICES=0 poetry run python main.py fit --config table_line_item_detection_config.yaml

```

To get table bounding box predictions, run e.g.

```
CUDA_VISIBLE_DEVICES=0 poetry run python predict.py --checkpoint_path /app/data/baselines/checkpoints/detr_table_20140.ckpt --prediction_type table_detection --dataset_path /app/data/docile/ --split val --output_json_path /app/data/baselines/line_item_detection/table_transformer/predictions/val/detr_table_detection.json
```

and e.g.

```
CUDA_VISIBLE_DEVICES=0 poetry run python predict.py --checkpoint_path /app/data/baselines/checkpoints/detr_LI_170100.ckpt --prediction_type table_line_item_detection --dataset_path /app/data/docile/ --split val --output_json_path /app/data/baselines/line_item_detection/table_transformer/predictions/val/detr_table_line_item_detection.json --table_detection_predictions_pickle /app/data/baselines/line_item_detection/table_transformer/predictions/val/detr_table_detection.pickle
```

to get line item bounding boxes. These can then be passed to [../NER/docile_inference_NER_multilabel.py](../NER/docile_inference_NER_multilabel.py) as `--crop_bboxes_filename` and `--line_item_bboxes_filename`, respectively.
