import dataclasses
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from helpers import FieldWithGroups
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from docile.dataset import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ScriptArguments:
    """
    Arguments which aren't included in the GaudiTrainingArguments
    """

    docile_path: Path = dataclasses.field(
        default=Path("/storage/pif_documents/dataset_exports/docile221221-0/"),
        # default=Path("/storage/pif_documents/dataset_exports/docile_pretraining_v1_2022_12_22/"),
        metadata={"help": ""},
    )
    split: str = dataclasses.field(
        default="train",
        # default="unlabeled",
        metadata={"help": ""},
    )
    dataset_id: str = dataclasses.field(
        default=None,
        metadata={"help": "The repository id of the dataset to use (via the datasets library)."},
    )
    output_dir: str = dataclasses.field(
        default=Path("/storage/table_extraction/pretrain_roberta/"),
        metadata={
            "help": "The repository id where the model will be saved or loaded from for further pre-training."
        },
    )
    model_config_id: Optional[str] = dataclasses.field(
        # default="bert-base-uncased", metadata={"help": "Pretrained config name or path if not the same as model_name"}
        default="roberta-base",
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    per_device_train_batch_size: Optional[int] = dataclasses.field(
        default=16,
        metadata={"help": "The Batch Size per HPU used during training"},
    )
    dataloader_num_workers: Optional[int] = dataclasses.field(
        default=0,
        metadata={"help": "Num workers for dataloader"},
    )
    gradient_accumulation_steps: Optional[int] = dataclasses.field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    max_steps: Optional[int] = dataclasses.field(
        default=1_000_000,
        metadata={"help": "The Number of Training steps to perform."},
    )
    learning_rate: Optional[float] = dataclasses.field(
        default=1e-4, metadata={"help": "Learning Rate for the training"}
    )
    mlm_probability: Optional[float] = dataclasses.field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    re_order_ocr_boxes: Optional[bool] = dataclasses.field(
        default=False, metadata={"help": "Flag if the OCR boxes should be re-ordered."}
    )
    warmup_steps: Optional[int] = dataclasses.field(
        default=5000, metadata={"help": "Warmup steps."}
    )


def get_center_line_clusters(line_item):
    # get centers of text boxes (y-axis only)
    centers = np.array([x.bbox.centroid[1] for x in line_item])
    heights = np.array([x.bbox.height for x in line_item])

    n_bins = len(centers)
    if n_bins < 1:
        return {}

    hist_h, bin_edges_h = np.histogram(heights, bins=n_bins)
    bin_centers_h = bin_edges_h[:-1] + np.diff(bin_edges_h) / 2
    idxs_h = np.where(hist_h)[0]
    # heights_cluster_centers = bin_centers_h[idxs_h]
    heights_cluster_centers = np.unique(bin_centers_h[idxs_h].astype(np.int32))
    heights_cluster_centers.sort()

    # group text boxes by heights
    groups_heights = {}
    for field in line_item:
        g = np.array(
            list(map(lambda height: np.abs(field.bbox.height - height), heights_cluster_centers))
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
        # there is probably just 1 cluster
        return {0: y_center_clusters.mean()}
    else:
        #  estimate the number of lines by looking at the cluster centers
        clusters = {}
        cnt = 0
        yc_prev = y_center_clusters[0]
        for yc in y_center_clusters:
            # if np.abs(yc_prev - yc) < 5:
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


def split_fields_by_text_lines(line_item):
    clusters = get_center_line_clusters(line_item)
    new_line_item = []
    for ft in line_item:
        g = np.array(
            # list(map(lambda y: (ft.bbox.centroid[1] - y) ** 2, clusters.values()))
            list(map(lambda y: np.abs(ft.bbox.to_tuple()[1] - y), clusters.values()))
        ).argmin()
        updated_ft = dataclasses.replace(ft, groups=[g])
        new_line_item.append(updated_ft)
    return new_line_item, clusters


def get_sorted_field_candidates(lineitem_fields):
    lineitem_fields = [FieldWithGroups.from_dict(field.to_dict()) for field in lineitem_fields]
    fields = []
    # for lid, line_item in lineitem_fields.items():
    for line_item in [lineitem_fields]:
        # clustering of text boxes in a given line item into individual text lines (stored in fieldlabel.groups)
        line_item, clusters = split_fields_by_text_lines(line_item)

        # sort text boxes by
        line_item.sort(key=lambda x: x.groups)

        # group by lines:
        groups = {}
        for ft in line_item:
            gid = str(ft.groups)
            if gid not in groups.keys():
                groups[gid] = [ft]
            else:
                groups[gid].append(ft)

        # lid_str = f"{lid:04d}" if lid else "-001"

        for gid, fs in groups.items():
            # sort by x-axis (since we are dealing with a single line)
            fs.sort(key=lambda x: x.bbox.centroid[0])
            for f in fs:
                lid_str = f"{f.line_item_id:04d}" if f.line_item_id else "-001"
                updated_f = dataclasses.replace(
                    f,
                    # groups = [f"{lid:04d}{int(gid.strip('[]')):>04d}"]
                    groups=[f"{lid_str}{int(gid.strip('[]')):>04d}"],
                )
                fields.append(updated_f)
    return fields, clusters


class DataLoaderWrapper(TorchDataset):
    def __init__(self, dataset, tokenizer, re_order_ocr_boxes=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.re_order_ocr_boxes = re_order_ocr_boxes

    def __getitem__(self, idx):
        document = self.dataset[idx]
        num_pages = document.page_count
        # document.page_count = num_pages
        page_num = np.random.randint(low=0, high=num_pages, size=1)[0]
        ocr = document.ocr.get_all_words(page_num, snapped=True)
        # NOTE: re-order the OCR tokens, so it is top-to-bottom left-to-right
        if self.re_order_ocr_boxes:
            sorted_field, _ = get_sorted_field_candidates(ocr)
            tokens = [x.text for x in sorted_field]
        else:
            tokens = [x.text for x in ocr]
        # encoding["tokens"] = tokens
        # return {"text": tokens}
        tokenized_inputs = self.tokenizer(
            tokens,
            return_special_tokens_mask=True,
            truncation=True,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            is_split_into_words=True,
        )
        return tokenized_inputs

    def __len__(self):
        return len(self.dataset)


def run_mlm():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"Script parameters {script_args}")

    # set seed for reproducibility
    seed = 34
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_config_id,
        add_prefix_space=True if script_args.model_config_id == "roberta-base" else False,
    )

    dataset = Dataset(
        script_args.split, script_args.docile_path, load_annotations=False, load_ocr=False
    )

    train_dataset = DataLoaderWrapper(dataset, tokenizer, script_args.re_order_ocr_boxes)

    # load model from config (for training from scratch)
    logger.info("Training new model from scratch")
    config = AutoConfig.from_pretrained(script_args.model_config_id)

    model = AutoModelForMaskedLM.from_config(config)

    logger.info(f"Resizing token embedding to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=script_args.mlm_probability, pad_to_multiple_of=8
    )

    # define our hyperparameters
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        seed=seed,
        max_steps=script_args.max_steps,
        # logging & evaluation strategies
        logging_dir=f"{script_args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=5_000,
        save_total_limit=2,
        report_to="tensorboard",
        # pretraining
        ddp_find_unused_parameters=True,
        # throughput_warmup_steps=2,
        # warmup_steps=5000,
        warmup_steps=script_args.warmup_steps,
        dataloader_num_workers=script_args.dataloader_num_workers,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # train the model
    trainer.train()


if __name__ == "__main__":
    run_mlm()
