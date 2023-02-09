#!/usr/bin/env python
# coding=utf-8
import logging
import time
from math import ceil, floor
from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
from transformers.utils import check_min_version

from docile.dataset import Dataset

# check min transfomers version
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


class LayoutLMModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        processor,
        model_name="microsoft/layoutlmv3-base",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.5,
        warmup=0.03,
        # critetion,
        optimizer="AdamW",
        trainer=None,
        datamodule=None,
        ckpt_dir=None,
    ):
        super(LayoutLMModel, self).__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.optimizer = optimizer
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_dir = ckpt_dir

        self._init_model()
        self._init_criterion()
        print(self.model)
        self._epoch_time = time.time()

    def _init_model(self) -> None:
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.model = self.randomize_model(self.model)
        self.model.mlm_pred = torch.nn.Linear(
            in_features=self.config.hidden_size, out_features=self.config.vocab_size
        )

    def randomize_model(self, model):
        for module_ in model.named_modules():
            if isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model

    def _init_criterion(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def _compute_criterion(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = preds.contiguous().view(-1, self.config.vocab_size)
        targets = targets.contiguous().view(-1)
        return self.criterion(preds, targets)

    def configure_optimizers(self) -> None:
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer '{self.optimizer}'")

        total_steps = self.datamodule.train_dataset_size
        total_batch_size = (
            self.datamodule.batch_size * self.trainer.num_devices * self.trainer.num_nodes
        )
        max_steps = ceil(total_steps / total_batch_size) * self.trainer.max_epochs
        if self.warmup is None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max_steps
            )
        else:
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                max_epochs=max_steps,
                warmup_epochs=floor(max_steps * self.warmup),
                eta_min=self.learning_rate * 0.001,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def _log_progress_bar_metrics(self, loss: Tensor, key: str = "train") -> None:
        _loss = loss.detach()
        self.log(
            key + "/loss",
            _loss.cpu().item(),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

    def _compute_forward(
        self, text_masked: Tensor, images: Tensor, bbox: Tensor, attn_mask: Tensor
    ) -> Tensor:
        model_output = self.model(
            input_ids=text_masked,
            bbox=bbox,
            pixel_values=images,
            attention_mask=attn_mask,
        )
        last_hidden_state = model_output.last_hidden_state
        mlm_preds = self.model.mlm_pred(last_hidden_state)
        return mlm_preds

    def _get_outputs_and_losses(
        self,
        text_masked: Tensor,
        text_target: Tensor,
        images: Tensor,
        bbox: Tensor,
        attn_mask: Tensor,
        key: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        outputs = self._compute_forward(text_masked, images, bbox, attn_mask)
        outputs = outputs[:, : text_target.shape[1], :]
        loss = self._compute_criterion(outputs, text_target)
        return loss, outputs

    def _get_tensors(
        self,
        batch: Tensor,
    ) -> Tensor:
        text_masked = []
        text_target = []
        images = []
        bbox = []
        attention_mask = []
        for given_sample in batch:
            text_masked.append(given_sample["masked_ids"])
            text_target.append(given_sample["input_ids"])
            images.append(given_sample["pixel_values"])
            bbox.append(given_sample["bbox"])
            attention_mask.append(given_sample["attention_mask"])
        text_masked = torch.cat(text_masked, dim=0).to(torch.long)
        text_target = torch.cat(text_target, dim=0).to(torch.long)
        images = torch.cat(images, dim=0)
        bbox = torch.cat(bbox, dim=0).to(torch.long)
        attention_mask = torch.cat(attention_mask, dim=0)
        return text_masked, text_target, images, bbox, attention_mask

    def training_step(self, batch: Tensor, batch_idx: int, key: str = "train") -> Tensor:
        text_masked, text_target, images, bbox, attention_mask = self._get_tensors(batch)
        self.model.train()
        loss, outputs = self._get_outputs_and_losses(
            text_masked=text_masked,
            text_target=text_target,
            images=images,
            bbox=bbox,
            attn_mask=attention_mask,
            key=key,
        )
        self._log_progress_bar_metrics(
            loss,
            key=key,
        )
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int, key: str = "val") -> Tensor:
        text_masked, text_target, images, bbox, attention_mask = self._get_tensors(batch)
        self.model.train()
        loss, outputs = self._get_outputs_and_losses(
            text_masked=text_masked,
            text_target=text_target,
            images=images,
            bbox=bbox,
            attn_mask=attention_mask,
            key=key,
        )
        self._log_progress_bar_metrics(
            loss,
            key=key,
        )
        return loss

    def test_step(self, batch: Tensor, batch_idx: int, key: str = "test") -> Tensor:
        text_masked, text_target, images, bbox, attention_mask = self._get_tensors(batch)
        self.model.eval()
        loss, outputs = self._get_outputs_and_losses(
            text_masked=text_masked,
            text_target=text_target,
            images=images,
            bbox=bbox,
            attn_mask=attention_mask,
            key=key,
        )
        self._log_progress_bar_metrics(
            loss,
            key=key,
        )
        return loss


class DocDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        split,
        model_name,
        batch_size: int = 32,
        num_workers: int = 6,
        img_size: int = 224,
        img_scale: int = [1.0, 1.0],  # noqa B006
        use_sorted=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset = Dataset(
            split,
            dataset_path,
            load_ocr=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name, apply_ocr=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.img_scale = img_scale
        self.use_sorted = use_sorted
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.img_size, scale=self.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.img_size, scale=self.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        self.train_dataset = DataLoaderWrapper(
            self.dataset,
            tokenizer=self.tokenizer,
            processor=self.processor,
            transform=self.train_transforms,
            use_sorted=self.use_sorted,
        )
        self.val_dataset = DataLoaderWrapper(
            self.dataset,
            tokenizer=self.tokenizer,
            processor=self.processor,
            transform=self.val_transforms,
            use_sorted=self.use_sorted,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_wokers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: x,
        )


class DataLoaderWrapper:
    def __init__(
        self,
        dataset,
        tokenizer,
        processor,
        transform,
        image_size=(224, 224),
        max_words=512,
        mlm_probability=0.15,
        use_sorted=False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.transform = transform
        self.image_size = image_size
        self.max_words = max_words
        self.mlm_probability = mlm_probability
        self.use_sorted = use_sorted

    def get_all_words(self, page_json):
        all_text = []
        for i in range(len(page_json)):
            given_text = page_json[i].text.lower()
            all_text.append(given_text)
        return all_text

    def get_all_tokens_bb(self, dict_ocr, max_words):
        l_tokens = []
        l_bb = []
        for given_word in dict_ocr:
            word = given_word.text.lower()
            bbox = given_word.bbox
            x0 = bbox.left * self.image_size[0]
            y0 = bbox.top * self.image_size[1]
            x1 = bbox.right * self.image_size[0]
            y1 = bbox.bottom * self.image_size[1]
            bbox = [int(x0), int(y0), int(x1), int(y1)]
            l_tokens.append(word)
            l_bb.append(bbox)
        assert len(l_tokens) == len(l_bb)
        if len(l_tokens) > self.max_words:
            l_tokens = l_tokens[: self.max_words]
            l_bb = l_bb[: self.max_words]
        return l_tokens, l_bb

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = (
            torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
            .to(device)
            .to(input_ids.dtype)
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        targets[input_ids == None] = -100  # noqa E711
        input_ids[input_ids == None] = self.tokenizer.pad_token_id  # noqa E711
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def adjust_inputs(self, encodings):
        temp_input_ids = torch.full([1, self.max_words], self.tokenizer.pad_token_id)
        temp_input_ids[
            :, : min(encodings.input_ids.shape[1], self.max_words)
        ] = encodings.input_ids[:, : self.max_words]
        temp_bbox = torch.full([1, self.max_words, 4], 0.0)
        temp_bbox[:, : min(encodings.input_ids.shape[1], self.max_words), :] = encodings.bbox[
            :, : self.max_words, :
        ]
        temp_attn_mask = torch.full([1, self.max_words], 0)
        temp_attn_mask[
            :, : min(encodings.input_ids.shape[1], self.max_words)
        ] = encodings.attention_mask[:, : self.max_words]
        encodings["bbox"] = temp_bbox
        encodings["input_ids"] = temp_input_ids
        encodings["attention_mask"] = temp_attn_mask
        return encodings

    def get_center_line_clusters(self, line_item):
        centers = np.array([x.bbox.centroid[1] for x in line_item])
        heights = np.array([x.bbox.height for x in line_item])
        n_bins = len(centers)
        if n_bins < 1:
            return {}

        hist_h, bin_edges_h = np.histogram(heights, bins=n_bins)
        bin_centers_h = bin_edges_h[:-1] + np.diff(bin_edges_h) / 2
        idxs_h = np.where(hist_h)[0]
        heights_cluster_centers = np.unique(bin_centers_h[idxs_h].astype(np.int32))
        heights_cluster_centers.sort()

        groups_heights = {}
        for field in line_item:
            g = np.array(
                list(
                    map(lambda height: np.abs(field.bbox.height - height), heights_cluster_centers)
                )
            ).argmin()
            gid = heights_cluster_centers[g]
            if gid not in groups_heights:
                groups_heights[gid] = [field]
            else:
                groups_heights[gid].append(field)

        hist, bin_edges = np.histogram(centers, bins=n_bins)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        idxs = np.where(hist)[0]
        y_center_clusters = bin_centers[idxs]
        y_center_clusters.sort()
        line_item_height = y_center_clusters.max() - y_center_clusters.min()

        if line_item_height < heights_cluster_centers[0]:
            return {0: y_center_clusters.mean()}
        else:
            clusters = {}
            cnt = 0
            yc_prev = y_center_clusters[0]
            for yc in y_center_clusters:
                if np.abs(yc_prev - yc) < heights_cluster_centers[0]:
                    flag = True
                else:
                    flag = False
                if flag:
                    if cnt not in clusters:
                        clusters[cnt] = [yc]
                    else:
                        clusters[cnt].append(yc)
                else:
                    cnt += 1
                    clusters[cnt] = [yc]
                yc_prev = yc
            for k, v in clusters.items():
                clusters[k] = np.array(v).mean()
        return clusters

    def split_fields_by_text_lines(self, line_item):
        clusters = self.get_center_line_clusters(line_item)
        for ft in line_item:
            g = np.array(
                list(map(lambda y: np.abs(ft.bbox.to_tuple()[1] - y), clusters.values()))
            ).argmin()
            ft.groups = [g]
        return line_item, clusters

    def get_sorted_field_candidates(self, original_fields):
        fields = []
        line_item, clusters = self.split_fields_by_text_lines(original_fields)
        line_item.sort(key=lambda x: x.groups)
        groups = {}
        for ft in line_item:
            gid = str(ft.groups)
            if gid not in groups.keys():
                groups[gid] = [ft]
            else:
                groups[gid].append(ft)
        for gid, fs in groups.items():
            fs.sort(key=lambda x: x.bbox.centroid[0])
            for f in fs:
                lid_str = f"{f.line_item_id:04d}" if f.line_item_id else "-001"
                f.groups = [f"{lid_str}{int(gid.strip('[]')):>04d}"]
            fields.extend(fs)
        return fields, clusters

    def __getitem__(self, idx):
        document = self.dataset[idx]
        num_pages = document.page_count
        page_num = np.random.randint(low=0, high=num_pages, size=1)[0]
        page_json = document.ocr.get_all_words(page_num, snapped=True)
        if self.use_sorted:
            page_json, _ = self.get_sorted_field_candidates(page_json)
        page_img = document.page_image(page_num, self.image_size)
        page_img = self.transform(page_img)
        all_words, all_bb = self.get_all_tokens_bb(page_json, max_words=self.max_words)
        encoding = self.processor(page_img, all_words, boxes=all_bb, return_tensors="pt")
        encoding = self.adjust_inputs(encoding)
        input_ids = encoding.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(
            input_ids=input_ids,
            vocab_size=self.tokenizer.vocab_size,
            device=input_ids.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )
        encoding["input_ids"] = labels
        encoding["masked_ids"] = input_ids
        return encoding

    def __len__(self):
        return len(self.dataset)


def main():
    dataset_path = Path("/app/data/docile")
    split = "unlabeled"
    model_name = "microsoft/layoutlmv3-base"
    batch_size = 128
    num_workers = 0
    learning_rate = 0.0006
    momentum = 0.99
    weight_decay = 0.5
    gradient_norm_clip_val = 5.0
    optimizer = "AdamW"
    ckpt_dir = "/app/data/baselines/trainings/layoutlmv3_pretraining"
    num_gpus = 8
    num_nodes = 1
    max_epochs = 30
    use_sorted = False
    resume_from_checkpoint = None

    logger.info("Pre-training LayoutLMv3")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=-1, save_on_train_epoch_end=True
    )
    f_trainer = pl.Trainer(
        callbacks=[lr_monitor, checkpoint_callback],
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        check_val_every_n_epoch=max_epochs,
        log_every_n_steps=1,
        default_root_dir=ckpt_dir,
        gradient_clip_val=gradient_norm_clip_val,
        resume_from_checkpoint=resume_from_checkpoint,
        strategy="ddp",
    )

    datamodule = DocDataModule(
        dataset_path=dataset_path,
        split=split,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        use_sorted=use_sorted,
    )
    datamodule.train_dataset_size = len(datamodule.dataset)
    datamodule.val_dataset_size = len(datamodule.dataset)
    datamodule.test_dataset_size = len(datamodule.dataset)

    model = LayoutLMModel(
        tokenizer=datamodule.tokenizer,
        processor=datamodule.processor,
        model_name=model_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        trainer=None,
        datamodule=datamodule,
        ckpt_dir=ckpt_dir,
    )

    f_trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
