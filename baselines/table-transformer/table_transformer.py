import json
import logging
import os
import pickle
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection
from transformers.models.detr.feature_extraction_detr import DetrFeatureExtractor
from transformers.models.detr.modeling_detr import DetrForObjectDetection
from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerForObjectDetection,
    TableTransformerObjectDetectionOutput,
)

import docile.dataset


def get_bbox_predictions(
    batch: Dict,
    out: TableTransformerObjectDetectionOutput,
    extractor: DetrFeatureExtractor,
    allowed_label_ids: Optional[Iterable] = None,
    threshold: float = 0.5,
):
    """Get bboxes from model output

    Parameters
    ----------
    batch
        A batch from DataLoader
    out :
        Batched model output
    allowed_label_ids
        Only return predictions of this category
    threshold
        Don't return predictions with lower confidence

    Returns
    -------
    bbox_predictions : List[np.ndarray]
        A list of bbox predictions arrays, one for each input example
    """
    orig_sizes = torch.concat(
        [labels["orig_size"].detach().unsqueeze(0) for labels in batch["labels"]]
    )
    postprocessed_outputs = extractor.post_process_object_detection(
        out, threshold=threshold, target_sizes=orig_sizes
    )

    res = []
    for i in range(len(out.logits)):
        scores = postprocessed_outputs[i]["scores"].detach().cpu().numpy()
        bboxes_scaled = (
            postprocessed_outputs[i]["boxes"].detach().cpu().numpy()[scores.argsort()[::-1]]
        )

        if allowed_label_ids is not None:
            bboxes_scaled = bboxes_scaled[
                np.array(
                    [li in allowed_label_ids for li in postprocessed_outputs[i]["labels"]],
                    dtype=bool,
                )
            ]
            areas = (bboxes_scaled[:, 2] - bboxes_scaled[:, 0]).astype(np.float32) * (
                bboxes_scaled[:, 3] - bboxes_scaled[:, 1]
            )
            bboxes_scaled = bboxes_scaled[areas > 4]

        res.append(bboxes_scaled)
    return res


def load_table_bboxes(path):
    """Load table bboxes from a saved COCO dataset

    Output bboxes are in the (left, top, right, bottom) format (suitable for
    TableDetr)
    """
    with open(path) as fin:
        labels = json.load(fin)

    def coco_to_detr_bbox(coco_bbox):
        left, top, width, height = coco_bbox
        return [left, top, left + width, top + height]

    return [
        coco_to_detr_bbox(annot["bbox"])
        for annot in labels["annotations"]
        if annot["category_id"] == 1
    ]


class TableTransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        docile_dataset,
        extractor,
        filter_category_ids: Optional[Sequence[int]] = None,
        crop_bboxes: Optional[Sequence[Sequence[float]]] = None,
        load_ground_truth_crop_bboxes: bool = False,
        num_workers: int = 10,
    ):
        """Dataset for table-transformer/DETR

        Parameters
        ----------
        extractor
            HF table-transformer feature extractor
        filter_category_ids
            keep only these category ids (0 for tables, 2 for line items)
        crop_bboxes
            Crop bboxes, one for each image, in the (left, top, right, bottom)
            and values in pixels
        load_ground_truth_crop_bboxes
            Use table bboxes as crop_bboxes
        num_workers
            Number of processes to use for dataset preparation
        """
        self.docile_dataset = docile_dataset
        self.extractor = extractor
        self.filter_category_ids = filter_category_ids
        self.num_workers = num_workers

        if crop_bboxes is not None and load_ground_truth_crop_bboxes:
            raise ValueError(
                "both `crop_bboxes` and `load_ground_truth_crop_bboxes` were specified"
            )
        self.crop_bboxes = crop_bboxes

        self.coco_annots, gt_crop_bboxes = self._get_coco_annots_and_crop_bboxes()
        if load_ground_truth_crop_bboxes:
            self.crop_bboxes = gt_crop_bboxes

        if self.crop_bboxes is not None:
            assert len(self.coco_annots) == len(self.crop_bboxes)
            self.coco_annots = [
                a
                for a, crop_bbox in zip(self.coco_annots, self.crop_bboxes)
                if crop_bbox is not None
            ]
            self.crop_bboxes = [
                crop_bbox for crop_bbox in self.crop_bboxes if crop_bbox is not None
            ]

    def _get_coco_annots_and_crop_bboxes(self):
        """
        Returns
        -------
        coco_annots, crop_bboxes
            Both lists contain one element for each page in the dataset
        """

        logging.info("Converting annotations to the COCO format")
        with ThreadPool(self.num_workers) as pool:

            def f(i):
                doc = self.docile_dataset[i]
                coco_annots = []
                crop_bboxes = []
                with doc:
                    doc_coco_annots = self.get_coco_annotations(doc)
                    for page_n, coco_annot in enumerate(doc_coco_annots):
                        crop_bboxes.append(self.get_table_bbox(doc, page_n))
                        if self.filter_category_ids is not None:
                            coco_annot = [
                                a
                                for a in coco_annot
                                if a["category_id"] in self.filter_category_ids
                            ]
                        coco_annots.append(((doc.docid, page_n), coco_annot))
                return coco_annots, crop_bboxes

            res = list(
                tqdm.tqdm(
                    pool.imap(f, range(len(self.docile_dataset))), total=len(self.docile_dataset)
                )
            )
        coco_annots = []
        crop_bboxes = []
        for ca, cb in res:
            coco_annots.extend(ca)
            crop_bboxes.extend(cb)
        return coco_annots, crop_bboxes

    @staticmethod
    def get_table_bbox(doc, page_n):
        """
        Returns
        -------
        bbox
            Table bbox in the (left, top, right, bottom) format or None if
            there is no table on the page
        """
        table_grid = doc.annotation.get_table_grid(page_n)
        if table_grid is None:
            return None
        img = doc.page_image(page_n)
        table_bbox = table_grid.bbox.to_absolute_coords(width=img.width, height=img.height)
        return [table_bbox.left, table_bbox.top, table_bbox.right, table_bbox.bottom]

    @staticmethod
    def _get_table_bbox(coco_annot):
        """Deprecated"""
        table_bbox = None
        if coco_annot:
            table = [a for a in coco_annot if a["category_id"] == 0][0]
            left, top, width, height = table["bbox"]
            right = left + width
            bottom = top + height
            table_bbox = [left, top, right, bottom]
        return table_bbox

    def __len__(self):
        return len(self.coco_annots)

    def get_img_and_target(self, idx):
        (doc_id, page_n), coco_annot = self.coco_annots[idx]
        doc = self.docile_dataset[doc_id]
        img = doc.page_image(page_n)

        if self.crop_bboxes is not None:
            crop_bbox = self.crop_bboxes[idx]
            img, coco_annot = self.crop_example(img, coco_annot, crop_bbox)

        target = {"image_id": -1, "annotations": coco_annot}
        return img, target

    def get_img(self, idx):
        """Return PIL image of the example at index `idx`"""
        return self.get_img_and_target(idx)[0]

    def get_coco_target(self, idx):
        """Return COCO target of the example at index `idx`"""
        return self.get_img_and_target(idx)[1]

    def __getitem__(self, idx):
        img, target = self.get_img_and_target(idx)
        encoding = self.extractor(images=img, annotations=target, return_tensors="pt")
        return encoding["pixel_values"].squeeze(), encoding["labels"][0]

    def crop_example(self, img: PIL.Image, coco_annot: List, crop_bbox: Sequence[float]):
        """Crop input image and annotations

        The annotations are cropped to their intersection with the image crop
        (and their bbox coordinates are changed to be relative to the crop top
        left corner)

        Parameters
        ----------
        coco_annot
            COCO target. A list of dicts with "bbox" (top, left, width,
            height), "area" etc. keys
        crop_bbox
            (top, left, right, bottom) (as returned by TableDetr)
        """
        new_img = img.crop(crop_bbox)

        crop_left, crop_top, crop_right, crop_bottom = crop_bbox
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top

        new_coco_annot = []
        for t in coco_annot:
            new_t = t.copy()

            left, top, width, height = new_t["bbox"]
            right = left + width
            bottom = top + height

            new_left = min(crop_width, max(0, left - crop_left))
            new_top = min(crop_height, max(0, top - crop_top))
            new_right = min(crop_width, max(0, right - crop_left))
            new_bottom = min(crop_height, max(0, bottom - crop_top))

            new_width = new_right - new_left
            new_height = new_bottom - new_top
            new_t["bbox"] = [new_left, new_top, new_width, new_height]
            new_t["area"] = new_width * new_height
            new_coco_annot.append(new_t)

        return new_img, new_coco_annot

    @classmethod
    def get_coco_annotations(cls, doc):
        page_to_lies = defaultdict(list)
        for lie in doc.annotation.content["line_item_extractions"]:
            page_to_lies[lie["page"]].append(lie)

        line_items_list = []
        for page_n in range(doc.page_count):
            img = doc.page_image(page_n)
            line_items = {}

            for lie in page_to_lies[page_n]:
                left, top, right, bottom = lie["bbox"]

                left *= img.width
                right *= img.width
                top *= img.height
                bottom *= img.height

                if lie["line_item_id"] in line_items:
                    li_left, li_top, li_width, li_height = line_items[lie["line_item_id"]]["bbox"]
                    li_right = li_left + li_width
                    li_bottom = li_top + li_height

                    left = min(left, li_left)
                    top = min(top, li_top)
                    right = max(right, li_right)
                    bottom = max(bottom, li_bottom)
                width = right - left
                height = bottom - top

                line_items[lie["line_item_id"]] = {
                    "image_id": -1,
                    "bbox": [left, top, width, height],
                    "category_id": 2,
                    "area": width * height,
                    # "iscrowd": 0
                    "docid": doc.docid,
                    "page_n": page_n,
                }
            line_items = list(line_items.values())
            if line_items:
                table_coco_annot = cls.get_table_coco_annot(doc, page_n)
                if table_coco_annot is None:
                    logging.warning(
                        f"Line items, but no table grid ({doc.docid, page_n}). "
                        "Using line item bboxes union."
                    )
                    table_coco_annot = cls._get_table_coco_annot(doc.docid, page_n, line_items)
                line_items.append(table_coco_annot)
            line_items_list.append(line_items)
        return line_items_list

    @staticmethod
    def _get_table_coco_annot(doc_id, page_n, line_items):
        """Deprecated"""
        table_left, table_top, table_width, table_height = line_items[0]["bbox"]
        table_right = table_left + table_width
        table_bottom = table_top + table_height

        for line_item in line_items:
            left, top, width, height = line_item["bbox"]
            right = left + width
            bottom = top + height
            table_left = min(table_left, left)
            table_top = min(table_top, top)
            table_right = max(table_right, right)
            table_bottom = max(table_bottom, bottom)

        table_width = table_right - table_left
        table_height = table_bottom - table_top
        return {
            "image_id": -1,
            "bbox": [table_left, table_top, table_width, table_height],
            "category_id": 0,
            "area": table_width * table_height,
            # "iscrowd": 0
            "docid": doc_id,
            "page_n": page_n,
        }

    @classmethod
    def get_table_coco_annot(cls, doc, page_n):
        table_bbox = cls.get_table_bbox(doc, page_n)
        if table_bbox is None:
            return None
        left, top, right, bottom = table_bbox
        width = right - left
        height = bottom - top
        return {
            "image_id": -1,
            "bbox": [left, top, width, height],
            "category_id": 0,
            "area": (bottom - top) * (right - left),
            # "iscrowd": 0
            "docid": doc.docid,
            "page_n": page_n,
        }


class TableDetr(pl.LightningModule):
    def __init__(
        self,
        description: str,
        train_dataset_name: str,
        val_dataset_name: str,
        task: str = None,
        initial_checkpoint: str = None,
        config_hf_id: str = None,
        dataset_path: str = "/app/data/docile/",
        load_ground_truth_crop_bboxes: bool = False,
        predictions_root_dir: str = "/app/data/baselines/line_item_detection/"
        "table_transformer/predictions/",
        crop_bboxes_filename: Optional[str] = None,
        lr: float = 3e-5,
        lr_backbone: float = 3e-7,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 16,
        threshold: float = 0.5,
    ):
        """Table transformer Lightning Module

        Modified from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

        Primarily created to try out table-transformer pretrained checkpoints:
        https://huggingface.co/microsoft/table-transformer-structure-recognition
        https://huggingface.co/microsoft/table-transformer-detection

        Parameters
        ----------
        description
            A short experiment description
        train_dataset_name
            Saved COCO dataset directory name (should exist in `dataset_root_dir`)
        val_dataset_name
            Saved COCO dataset directory name (should exist in `dataset_root_dir`)
        task
            Either table-line-item-detection or table-detection
        initial_checkpoint
            A checkpoint used to initialize model weights. Either a path to a
            .ckpt file, a HuggingFace id or None (then a default pretrained
            model will be used)
        config_hf_id
            Use this HF model id to load model (useful when starting from a
            local checkpoint)
        dataset_path
            Path to a directory with the DocILE dataset
        load_ground_truth_crop_bboxes
            Train on ground truth table crops (for structure recognition)
        predictions_root_dir
            Directory with saved crop_bboxes
        crop_bboxes_filename
            Name of the pickle file with saved crop bboxes
        lr
            Learning rate of the Detr transformer
        lr_backbone
            Learning rate of the ResNet-18 backbone
        num_workers
            DataLoader num_workers (both for train and val)
        threshold
            Only return detections with at least this confidence during inference
        """
        super().__init__()

        allowed_tasks = ["table-detection", "table-line-item-detection"]
        if task not in allowed_tasks:
            raise ValueError(f"task {task} not in {allowed_tasks}")

        if load_ground_truth_crop_bboxes and crop_bboxes_filename is not None:
            raise ValueError(
                "both load_ground_truth_crop_bboxes and crop_bboxes_filename specified"
            )
        self.crop_bboxes_filename = crop_bboxes_filename
        self.predictions_root_dir = predictions_root_dir

        if task == "table-line-item-detection":
            hf_id = "microsoft/table-transformer-structure-recognition"
            allowed_labels = ["table row"]
        elif task == "table-detection":
            hf_id = "microsoft/table-transformer-detection"
            allowed_labels = ["table"]

        if initial_checkpoint is None:
            initial_checkpoint = hf_id
        self.initial_checkpoint = initial_checkpoint
        if initial_checkpoint.endswith(".ckpt"):
            cls = TableTransformerForObjectDetection
            if config_hf_id is not None:
                hf_id = config_hf_id
                if hf_id == "facebook/detr-resnet-50":
                    cls = DetrForObjectDetection
            config = AutoConfig.from_pretrained(hf_id)
            self.model = cls(config)
            ckpt = torch.load(initial_checkpoint)
            self.model.load_state_dict(
                {
                    k.replace("model.model.", "model.")
                    .replace("model.class_labels_classifier", "class_labels_classifier")
                    .replace("model.bbox_predictor", "bbox_predictor"): v
                    for k, v in ckpt["state_dict"].items()
                }
            )
        else:
            if config_hf_id is not None and config_hf_id != initial_checkpoint:
                raise ValueError("Loading model from HF, but different config_hf_id specified")
            self.model = AutoModelForObjectDetection.from_pretrained(initial_checkpoint)
        # values from table-transformer (force for DETR)
        self.model.config.label2id["table"] = 0
        self.model.config.id2label[0] = "table"
        self.model.config.label2id["table row"] = 2
        self.model.config.id2label[2] = "table row"

        self.extractor = AutoFeatureExtractor.from_pretrained(hf_id)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allowed_labels = allowed_labels
        self.threshold = threshold
        self.description = description
        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = val_dataset_name
        self.load_ground_truth_crop_bboxes = load_ground_truth_crop_bboxes
        self.dataset_path = dataset_path

        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx, split):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        batch_size = batch["pixel_values"].shape[0]
        self.log(f"{split}_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"{split}_" + k, v.item(), batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        out = self.model(**batch)
        return get_bbox_predictions(
            batch,
            out,
            self.extractor,
            allowed_label_ids=[self.model.config.label2id[label] for label in self.allowed_labels],
            threshold=self.threshold,
        )

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    def get_dataloader(self, split: str):
        crop_bboxes = None
        if self.crop_bboxes_filename:
            crop_bboxes_path = os.path.join(
                self.predictions_root_dir,
                {"train": self.train_dataset_name, "val": self.val_dataset_name}[split],
                self.crop_bboxes_filename,
            )
            with open(crop_bboxes_path, "rb") as fin:
                crop_bboxes = pickle.load(fin)

        docile_dataset = docile.dataset.Dataset(
            split_name={"train": self.train_dataset_name, "val": self.val_dataset_name}[split],
            dataset_path=Path(self.dataset_path),
        )

        dataset = TableTransformerDataset(
            docile_dataset=docile_dataset,
            extractor=self.extractor,
            crop_bboxes=crop_bboxes,
            load_ground_truth_crop_bboxes=self.load_ground_truth_crop_bboxes,
            filter_category_ids=[
                self.model.config.label2id[label] for label in self.allowed_labels
            ],
        )
        dataloader = DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(split == "train"),
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")
