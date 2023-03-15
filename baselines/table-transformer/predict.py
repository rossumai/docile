import argparse
import json
import logging
import pickle
from collections import defaultdict

import table_transformer
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import docile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--prediction_type",
        choices=["table_detection", "table_line_item_detection"],
        required=True,
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--output_json_path",
        required=True,
        help="A dictionary doc_id->page_n->predictions. With --prediction_type "
        "table_detection, a pickle file with a list of bboxes (one for each "
        "page) will be saved beside the json file.",
    )
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_workers", default=16)
    parser.add_argument(
        "--table_detection_predictions_pickle",
        default=None,
        help="Path to a file created by this script for a table detection model.",
    )
    args = parser.parse_args()

    if (
        args.prediction_type == "table_line_item_detection"
        and args.table_detection_predictions_pickle is None
    ):
        raise ValueError(
            "--prediction_type table_line_item_detection requires --table_detection_predictions_pickle"
        )

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        logging.warning("CUDA not available, predicting on CPU")
        accelerator = "cpu"

    table_detr = table_transformer.TableDetr.load_from_checkpoint(args.checkpoint_path)

    docile_dataset = docile.dataset.Dataset(args.split, args.dataset_path)

    crop_bboxes = None
    if args.table_detection_predictions_pickle is not None:
        with open(args.table_detection_predictions_pickle, "rb") as fin:
            crop_bboxes = pickle.load(fin)
            dataset_not_cropped = table_transformer.TableTransformerDataset(
                docile_dataset=docile_dataset, extractor=table_detr.extractor
            )

    dataset = table_transformer.TableTransformerDataset(
        docile_dataset=docile_dataset, extractor=table_detr.extractor, crop_bboxes=crop_bboxes
    )

    evaluator = Trainer(accelerator=accelerator, devices=1)
    _res = evaluator.predict(
        table_detr,
        DataLoader(
            dataset,
            collate_fn=table_detr.collate_fn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
    )
    res_detection = []
    for r in _res:
        res_detection.extend(r)

    bbox_dict = defaultdict(lambda: defaultdict(list))

    if args.prediction_type == "table_detection":
        res_detection = [list(a[0]) if len(a) else None for a in res_detection]
        with open(args.output_json_path.strip(".json") + ".pickle", "wb") as fout:
            pickle.dump(res_detection, fout)
        for ((doc_id, page_n), _), bbox in zip(dataset.coco_annots, res_detection):
            bbox_dict[doc_id][page_n] = [round(x) for x in bbox] if bbox else None

    elif args.prediction_type == "table_line_item_detection":
        assert len(res_detection) == len(dataset.crop_bboxes)
        for bboxes, crop_bbox in zip(res_detection, dataset.crop_bboxes):
            for bbox in bboxes:
                # left offset
                bbox[0] += crop_bbox[0]
                bbox[2] += crop_bbox[0]
                # top offset
                bbox[1] += crop_bbox[1]
                bbox[3] += crop_bbox[1]

        # output predictions for all pages, not only for those with predictions
        for (doc_id, page_n), _ in dataset_not_cropped.coco_annots:
            bbox_dict[doc_id][page_n] = []

        for ((doc_id, page_n), _), bboxes in zip(dataset.coco_annots, res_detection):
            bbox_dict[doc_id][page_n] = [[round(x) for x in bbox] for bbox in bboxes]

    with open(args.output_json_path, "w") as fout:
        json.dump(bbox_dict, fout)


if __name__ == "__main__":
    main()
