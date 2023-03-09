import argparse
import dataclasses
import json
import math
import os
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from data_collator import MyMLDataCollatorForTokenClassification
from datasets import Dataset as ArrowDataset
from datasets import concatenate_datasets
from helpers import FieldWithGroups, show_summary
from my_roberta_multilabel import MyXLMRobertaMLForTokenClassification
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models.roberta.configuration_roberta import RobertaConfig

from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES, Dataset

classes = (
    [
        # 'background',
        # special background classes
        "KILE_background",
        "LI_background",
        "LIR_background",
        # LI class
        # 'LI',  #  NOTE: will be added separately to unique_entities
    ]
    + KILE_FIELDTYPES
    + LIR_FIELDTYPES
)


def tag_fields_with_entities(fields, unique_entities=[]):  # noqa: B006
    # assumes that tokens are FieldLabels and already sorted (by text lines, i.e. vertically) and horizontaly (by x-axis)
    # hash map for determining entity type (B, I)
    if len(unique_entities) < 1:
        entity_map = {x: False for x in classes}
    else:
        entity_map = {x[2:]: False for x in unique_entities}

    tokens_with_entities = []

    prev_lid = None

    for token in fields:
        fts = token.fieldtype if isinstance(token.fieldtype, list) else None
        lid = token.line_item_id

        other = (token.bbox.to_tuple(),)

        if not fts and lid is None:  # 0 0
            # complete O class
            tokens_with_entities.append((token.text, ["O-KILE", "O-LIR", "O-LI"], other))
        elif not fts and lid:  # 0 1
            # can be just B-LI, I-LI, or E-LI
            li_label = "B-LI" if prev_lid is None or prev_lid != lid else "I-LI"
            tokens_with_entities.append((token.text, ["O-KILE", "O-LIR", li_label], other))
        elif fts and lid is None:  # 1 0
            # add all classes from fts
            label = []
            for ft in fts:
                if entity_map[ft]:
                    label.append(f"I-{ft}")
                else:
                    label.append(f"B-{ft}")
                    entity_map[ft] = True
            label.append("O-LI")
            tokens_with_entities.append((token.text, label, other))
        else:  # 1 1
            li_label = "B-LI" if prev_lid is None or prev_lid != lid else "I-LI"
            # add all classes from fts
            label = []
            for ft in fts:
                if entity_map[ft]:
                    label.append(f"I-{ft}")
                else:
                    label.append(f"B-{ft}")
                    entity_map[ft] = True
            label.append(li_label)
            tokens_with_entities.append((token.text, label, other))

        # reset line_item labels in entity_map if there is a transition to a different line_item
        if prev_lid != lid:
            for k in entity_map.keys():
                if k.startswith("line_item_"):
                    entity_map[k] = False

        # possibly correct last but one entry (because it should have [x, E-LI] instead of [x, I-LI], unless it is O-LI)
        if prev_lid != lid:
            if len(tokens_with_entities) > 1:
                if tokens_with_entities[-2][1][-1][0] != "O":
                    tokens_with_entities[-2][1][-1] = "E-LI"

        prev_lid = lid

    return tokens_with_entities


class NERDataMaker:
    # NER Data Maker class for processing Full Table
    def __init__(self, data, metadata, unique_entities=None, use_BIO_format=True) -> None:
        if not unique_entities:
            have_unique_entities = False
            self.unique_entities = []
        else:
            have_unique_entities = True
            self.unique_entities = unique_entities
        self.processed_tables = []

        temp_processed_tables = []
        for page_data in tqdm(data, desc="Processing tables 1/2"):
            tokens_with_entities = tag_fields_with_entities(page_data, self.unique_entities)
            if tokens_with_entities:
                if not have_unique_entities:
                    for _, ent, _ in tokens_with_entities:
                        if ent not in self.unique_entities:
                            self.unique_entities.append(ent)
                temp_processed_tables.append(tokens_with_entities)

        if not have_unique_entities:
            self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")

        for tokens_with_entities in tqdm(temp_processed_tables, desc="Processing tables 2/2"):
            self.processed_tables.append(
                [
                    # (t, self.unique_entities.index(ent), info)
                    (t, [self.unique_entities.index(e) for e in ent], info)
                    for t, ent, info in tokens_with_entities
                ]
            )

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_tables)

    def __getitem__(self, idx):
        def _process_tokens_for_one_page(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            infos = []
            # bboxes = []
            for t, ent, info in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)
                infos.append(info)

            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens,
                "infos": infos,
            }

        tokens_with_encoded_entities = self.processed_tables[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_page(idx, tokens_with_encoded_entities)
        else:
            return [
                _process_tokens_for_one_page(i + idx.start, tee)
                for i, tee in enumerate(tokens_with_encoded_entities)
            ]

    def as_hf_dataset(self, tokenizer, tag_everything=False, stride=0):
        from datasets import Array2D
        from datasets import Dataset as ArrowDataset
        from datasets import Features, Sequence, Value

        def tokenize_and_align_labels_unbatched(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                stride=stride,
                padding=True,
                max_length=512,
                return_overflowing_tokens=True,  # important !!!
                return_length=True,
                verbose=True,
            )

            labels = []
            bboxes = []

            i = 0
            label = examples["ner_tags"]
            bbox = examples["bboxes"]
            overflowing = tokenized_inputs[i].overflowing
            for i in range(0, 1 + len(overflowing)):
                word_ids = tokenized_inputs[i].word_ids
                previous_word_idx = None
                label_ids = []
                bboxes_tmp = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(np.zeros_like(label[0]))
                        bboxes_tmp.append(np.array([0, 0, 0, 0], dtype=np.int32))
                    elif (word_idx != previous_word_idx) or (tag_everything):
                        label_ids.append(label[word_idx])
                        bboxes_tmp.append(bbox[word_idx])
                    else:
                        label_ids.append(np.zeros_like(label[0]))
                        bboxes_tmp.append(np.array([0, 0, 0, 0], dtype=np.int32))
                    previous_word_idx = word_idx
                labels.append(label_ids)
                bboxes.append(bboxes_tmp)
            tokenized_inputs["labels"] = labels
            tokenized_inputs["bboxes"] = bboxes
            return tokenized_inputs

        ids, ner_tags, tokens, infos = [], [], [], []
        bboxes = []

        def make_labels(x, N):
            tmp = np.zeros(N, dtype=bool)
            tmp[x] = 1
            return tmp

        for i, pt in enumerate(self.processed_tables):
            ids.append(i)
            pt_tokens, pt_tags, pt_info = list(zip(*pt))
            ner_tags.append(tuple([make_labels(x, len(self.unique_entities)) for x in pt_tags]))
            tokens.append(pt_tokens)
            infos.append(pt_info)
            bboxes.append(
                [np.array([d[0][0], d[0][1], d[0][2], d[0][3]], dtype=np.int32) for d in pt_info]
            )
        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens,
            "bboxes": bboxes,
        }
        features = Features(
            {
                "tokens": Sequence(Value("string")),
                "ner_tags": Array2D(shape=(None, len(self.unique_entities)), dtype="bool"),
                "id": Value("int32"),
                "bboxes": Array2D(shape=(None, 4), dtype="int32"),
            }
        )
        ds = ArrowDataset.from_dict(data, features)
        tokenized_ds = ds.map(tokenize_and_align_labels_unbatched, batched=False)
        return tokenized_ds


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


def get_sorted_field_candidates(original_fields):
    fields = []
    # for lid, line_item in original_fields.items():

    # clustering of text boxes in a given line item into individual text lines (stored in fieldlabel.groups)
    # line_item, clusters = split_fields_by_text_lines(line_item)
    line_item, clusters = split_fields_by_text_lines(original_fields)

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


def get_data_from_docile(dataset, overlap_thr=0.5):
    data = []
    metadata = []

    for document in tqdm(dataset, desc=f"Generating data from {dataset}"):
        doc_id = document.docid
        # page_to_table_grids = document.annotation.content["metadata"]["page_to_table_grids"]

        kile_fields = [
            FieldWithGroups.from_dict(field.to_dict()) for field in document.annotation.fields
        ]
        li_fields = [
            FieldWithGroups.from_dict(field.to_dict()) for field in document.annotation.li_fields
        ]
        for page in range(document.page_count):
            img = document.page_image(page)
            W, H = img.size
            kile_fields_page = [field for field in kile_fields if field.page == page]
            li_fields_page = [field for field in li_fields if field.page == page]
            kile_fields_page = [
                dataclasses.replace(field, bbox=field.bbox.to_absolute_coords(W, H))
                for field in kile_fields_page
            ]
            li_fields_page = [
                dataclasses.replace(field, bbox=field.bbox.to_absolute_coords(W, H))
                for field in li_fields_page
            ]

            ocr = [
                FieldWithGroups.from_dict(word.to_dict())
                for word in document.ocr.get_all_words(page, snapped=True)
            ]
            ocr = [
                dataclasses.replace(
                    ocr_field, bbox=ocr_field.bbox.to_absolute_coords(W, H), fieldtype=[]
                )
                for ocr_field in ocr
            ]

            # 0. Get table grid
            table_grid = document.annotation.get_table_grid(page)
            tables_bbox = table_grid.bbox.to_absolute_coords(W, H) if table_grid else None

            # 1. Tag ocr fields with fieldtypes from kile_fields + li_fields
            # We sort the kile+lir fields by top coordinate and then for each ocr field we performr
            # binary search to find only the kile+lir fields overlapping vertically.
            # Note: original index is kept to preserve original behaviour.
            kile_li_fields_page = kile_fields_page + li_fields_page
            kile_li_fields_page_sorted = sorted(
                enumerate(kile_li_fields_page),
                key=lambda i_f: i_f[1].bbox.top,
            )
            fields_top_coords = [field.bbox.top for _, field in kile_li_fields_page_sorted]
            # Max bottom coordinate is needed to have a sorted array for binary search. This means
            # some extra fields will be included in the found range, causing a very minor slowdown.
            fields_bottom_coords_max = [
                field.bbox.bottom for _, field in kile_li_fields_page_sorted
            ]
            for i in range(1, len(kile_li_fields_page_sorted)):
                fields_bottom_coords_max[i] = max(
                    fields_bottom_coords_max[i],
                    fields_bottom_coords_max[i - 1],
                )
            # Indexes to original kile_li_fields_page array
            fields_idxs = [idx for idx, _ in kile_li_fields_page_sorted]

            updated_ocr = []
            for ocr_field in ocr:
                new_ocr_field = dataclasses.replace(ocr_field, groups="")
                # take only fields with bottom coord after ocr_field.bbox.top
                i_l = bisect_right(fields_bottom_coords_max, ocr_field.bbox.top)
                # take only fields with top coord before ocr_field.bbox.bottom
                i_r = bisect_left(fields_top_coords, ocr_field.bbox.bottom)
                for idx in sorted(fields_idxs[i_l:i_r]):
                    field = kile_li_fields_page[idx]
                    if ocr_field.bbox and field.bbox:
                        if (
                            field.bbox.intersection(ocr_field.bbox).area / ocr_field.bbox.area
                            >= overlap_thr
                        ):
                            if field.fieldtype not in ocr_field.fieldtype:
                                new_ocr_field.fieldtype.append(field.fieldtype)
                            new_ocr_field = dataclasses.replace(
                                new_ocr_field, line_item_id=field.line_item_id
                            )
                updated_ocr.append(new_ocr_field)
            ocr = updated_ocr

            # Re-Order OCR boxes
            sorted_fields, _ = get_sorted_field_candidates(ocr)

            tables_ocr = []
            if tables_bbox:
                for field in sorted_fields:
                    if tables_bbox.intersection(field.bbox).area / field.bbox.area >= overlap_thr:
                        tables_ocr.append(field)

            # # 2. Split into individual lines, group by line item id
            # for table_i, table_fields in enumerate(tables_ocr):
            text_lines = {}
            # for field in page_fields:
            for field in tables_ocr:
                gid = field.groups[0][4:]
                if gid not in text_lines:
                    text_lines[gid] = [field]
                else:
                    text_lines[gid].append(field)
            # now there should be only 1 line_item_id (or first 04d in groups) per each text_lines
            # we need to merge text_lines, if there are several of them assigned to the same line_item_id
            line_items = {}
            # prev_id = 0 + 1000*table_i
            prev_id = 0 + 1000 * page
            for _, fields in text_lines.items():
                line_item_ids = [x.line_item_id for x in fields if x.line_item_id is not None]
                prev_id = line_item_ids[0] if line_item_ids else prev_id
                if prev_id not in line_items:
                    line_items[prev_id] = fields
                else:
                    line_items[prev_id].extend(fields)
            # 3. Append to data, which will be then used to construct NER Dataset
            new_line_items = {}
            for lid, fields in line_items.items():
                if lid > 0:
                    new_fields = []
                    for field in fields:
                        gid = field.groups[0]
                        new_field = dataclasses.replace(
                            field, line_item_id=lid, groups=[f"{lid:04d}{gid[4:]}"]
                        )
                        new_fields.append(new_field)
                    new_line_items[lid] = new_fields
                else:
                    new_line_items[lid] = fields
            line_items = new_line_items

            # append data and metadata
            metadata.append(
                {
                    "i": len(data),
                    "doc_id": doc_id,
                    "page_n": page,
                    # "table_n": table_i,
                    # "row_separators": row_sep[table_i]
                }
            )
            data.append(sorted_fields)
    return data, metadata


def load_metadata(src: Path):
    out = []
    with open(src, "r") as json_file:
        out = json.load(json_file)
    return out


def store_metadata(dest: Path, metadata):
    with open(dest, "w") as json_file:
        json.dump(metadata, json_file)


def load_data(src: Path):
    out = []
    with open(src, "r") as json_file:
        A = json.load(json_file)
    for table_data in A:
        out.append([])
        for field in table_data:
            out[-1].append(FieldWithGroups.from_dict(field))
    return out


def store_data(dest: Path, data):
    out = []
    for table_data in data:
        out.append([])
        for field in table_data:
            out[-1].append(
                {
                    "fieldtype": field.fieldtype if field.fieldtype else "background",
                    "bbox": field.bbox.to_tuple(),
                    "groups": field.groups,
                    "line_item_id": field.line_item_id,
                    "page": field.page,
                    "score": field.score,
                    "text": field.text,
                }
            )
    with open(dest, "w") as json_file:
        json.dump(out, json_file)


def _arrow_dataset_path(arrow_format_path, docile_dataset):
    return arrow_format_path / docile_dataset.split_name


def prepare_hf_dataset(
    docile_dataset,
    tokenizer,
    overlap_thr,
    arrow_format_path,
    preprocessed_dataset_path,
    chunk_size=100000,
):
    if len(docile_dataset) > chunk_size:
        if not arrow_format_path:
            raise NotImplementedError(
                f"You need to set --arrow-format-path because {docile_dataset} has more than 10000 documents"
            )
        num_chunks = math.ceil(len(docile_dataset) / chunk_size)
        dataset_chunks = []
        for chunk in range(num_chunks):
            chunk_dataset = docile_dataset[chunk * chunk_size : (chunk + 1) * chunk_size]
            chunk_dataset.split_name = f"{docile_dataset.split_name}_chunk_{chunk}_of_{num_chunks}"
            # make sure the chunk is stored to disk
            prepare_hf_dataset(
                chunk_dataset,
                tokenizer,
                overlap_thr,
                arrow_format_path,
                preprocessed_dataset_path,
                chunk_size,
            )
            # load it from disk
            dataset_chunk = ArrowDataset.load_from_disk(
                _arrow_dataset_path(arrow_format_path, chunk_dataset)
            )
            dataset_chunks.append(dataset_chunk)
        return concatenate_datasets(dataset_chunks)

    if arrow_format_path:
        try:
            load_path = _arrow_dataset_path(arrow_format_path, docile_dataset)
            print(f"Loading dataset in arrow format from path {load_path}")
            return ArrowDataset.load_from_disk(load_path)
        except Exception:
            print(f"Could not load {docile_dataset.split_name} in arrow format, regenerating.")

    if preprocessed_dataset_path:
        dataset_name = docile_dataset.data_paths.name
        preprocessed_path = preprocessed_dataset_path / dataset_name
        print(
            f"Loading preprocessed {docile_dataset.split_name} data from path {preprocessed_path}"
        )
        try:
            data = load_data(
                preprocessed_path / f"{docile_dataset.split_name}_multilabel_preprocessed.json"
            )
            metadata = load_metadata(
                preprocessed_path / f"{docile_dataset.split_name}_multilabel_metadata.json"
            )
        except Exception:
            print(f"Could not load preprocessed {docile_dataset.split_name}, regenerating.")
            data, metadata = get_data_from_docile(docile_dataset, overlap_thr=overlap_thr)
            print(
                f"Storing preprocessed {docile_dataset.split_name} data to {preprocessed_dataset_path}"
            )
            os.makedirs(preprocessed_dataset_path / dataset_name, exist_ok=True)
            store_data(
                preprocessed_path / f"{docile_dataset.split_name}_multilabel_preprocessed.json",
                data,
            )
            store_metadata(
                preprocessed_path / f"{docile_dataset.split_name}_multilabel_metadata.json",
                metadata,
            )
    else:
        data, metadata = get_data_from_docile(docile_dataset, overlap_thr=overlap_thr)

    data_maker = NERDataMaker(data, metadata, unique_entities=unique_entities, use_BIO_format=True)
    print("Converting dataset to HuggingFace format")
    dataset = data_maker.as_hf_dataset(
        tokenizer=tokenizer, stride=args.stride, tag_everything=args.tag_everything
    )
    if arrow_format_path:
        store_path = _arrow_dataset_path(arrow_format_path, docile_dataset)
        print(
            f"Storing HuggingFace Dataset for {docile_dataset.split_name} in arrow format to: {store_path}"
        )
        dataset.save_to_disk(store_path)
        print(f"HuggingFace Dataset for {docile_dataset.split_name} in arrow format stored")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docile_path",
        type=Path,
        default=Path("/app/data/docile/"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        # default="synthetic",
    )
    parser.add_argument(
        "--overlap_thr",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output_dir", type=str, help="Where to store the outputs.")
    parser.add_argument(
        "--train_bs",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--test_bs",
        type=int,
        default=8,
        help="Testing batch size",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="HuggingFace model to fine-tune",
    )
    parser.add_argument(
        "--use_bert",
        action="store_true",
    )
    parser.add_argument(
        "--use_roberta",
        action="store_true",
    )
    parser.add_argument(
        "--save_total_limit",
        default=None,
        type=int,
        help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="",
    )
    parser.add_argument(
        "--num_epochs",
        default=40,
        type=int,
        help="",
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help="",
    )
    parser.add_argument(
        "--use_BIO_format",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--use_2d_positional_embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--use_1d_positional_embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--use_2d_concat",
        action="store_true",
    )
    parser.add_argument(
        "--stride",
        default=0,
        type=int,
        help="Stride for tokenizer",
    )
    parser.add_argument("--use_new_2D_pos_emb", action="store_true")
    parser.add_argument(
        "--quant_step_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pos_emb_dim",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--tag_everything",
        action="store_true",
        help="If this is defined, all tokens will be tagged with class, unlike just the beginning of words",
    )
    parser.add_argument(
        "--bb_emb_dim",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--arrow_format_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--preprocessed_dataset_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report_all_metrics",
        action="store_true",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--use_classification_head",
        action="store_true",
        help="Whether to use the more complex classification head instead of just a single dense layer",
    )
    args = parser.parse_args()

    print(f"{datetime.now()} Started.")

    show_summary(args, __file__)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if "checkpoint" in args.model_name or "so_far_best" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            add_prefix_space=True if (args.model_name == "roberta-base") else False,
        )

    data_collator = MyMLDataCollatorForTokenClassification(
        tokenizer=tokenizer, max_length=512, padding="longest"
    )

    if args.use_BIO_format:
        # add complete background ()
        # unique_entities = ["O"]
        unique_entities = []
        # add class specific background tags
        # unique_entities.extend([f"O-{x.rstrip('_background')}" for x in classes[1:] if "background" in x])
        unique_entities.extend(
            [f"O-{x.rstrip('_background')}" for x in classes if "background" in x]
        )
        # add KILE and LIR class tags
        unique_entities.extend([f"B-{x}" for x in classes if "background" not in x])
        unique_entities.extend([f"I-{x}" for x in classes if "background" not in x])
        # add tags for LI
        unique_entities.extend(["B-LI", "I-LI", "E-LI"])
        # sort them out
        unique_entities.sort(key=lambda ent: ent if ent[0] != "O" else "")
    else:
        raise NotImplementedError("Not implemented yet.")

    id2label = dict(enumerate(unique_entities))
    label2id = {v: k for k, v in id2label.items()}

    val_docile_dataset = Dataset("val", args.docile_path, load_annotations=False, load_ocr=False)
    val_dataset = prepare_hf_dataset(
        val_docile_dataset,
        tokenizer,
        args.overlap_thr,
        args.arrow_format_path,
        args.preprocessed_dataset_path,
    )
    train_docile_dataset = Dataset(
        args.split, args.docile_path, load_annotations=False, load_ocr=False
    )
    train_dataset = prepare_hf_dataset(
        train_docile_dataset,
        tokenizer,
        args.overlap_thr,
        args.arrow_format_path,
        args.preprocessed_dataset_path,
    )

    if args.use_roberta:
        config = RobertaConfig.from_pretrained(args.model_name)
    else:
        raise Exception("Unknown type for NLP backbone selected.")

    print("\n\n\n")

    # instantiate model
    if args.use_roberta:
        config.use_2d_positional_embeddings = args.use_2d_positional_embeddings
        config.use_1d_positional_embeddings = args.use_1d_positional_embeddings
        config.use_new_2D_pos_emb = args.use_new_2D_pos_emb
        config.pos_emb_dim = args.pos_emb_dim
        config.quant_step_size = args.quant_step_size
        config.stride = args.stride
        config.bb_emb_dim = args.bb_emb_dim
        config.tag_everything = args.tag_everything
        config.num_labels = len(unique_entities)
        config.id2label = id2label
        config.label2id = label2id
        config.model_name = args.model_name
        config.use_bert = args.use_bert
        config.use_roberta = args.use_roberta
        config.use_BIO_format = args.use_BIO_format
        config.use_2d_concat = args.use_2d_concat
        config.use_classification_head = args.use_classification_head

        model = MyXLMRobertaMLForTokenClassification.from_pretrained(
            args.model_name, config=config
        )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.test_bs,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        report_to="tensorboard",
        save_total_limit=args.save_total_limit,
        seed=42,
        data_seed=42,
        metric_for_best_model="OVERALL_f1",
        greater_is_better=True,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print(f"INFO: tensorboard stored in {os.path.join(args.output_dir, 'runs')}")

    label_list = id2label

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        pred = torch.tensor(predictions)
        labs = torch.tensor(labels)

        num_labels = pred.shape[-1]

        labs_final = torch.where(labs == -100, 0, labs)
        pred_sig = torch.sigmoid(pred)
        pred_final = torch.where(pred_sig > 0.5, 1, 0)

        # break-down to classes
        result = torchmetrics.functional.stat_scores(
            pred_final.view(-1, num_labels),
            labs_final.view(-1, num_labels),
            task="multilabel",
            average=None,
            num_labels=num_labels,
        )

        results_breakdown = {}
        results_breakdown["KILE"] = torch.zeros_like(result[0])
        results_breakdown["LIR"] = torch.zeros_like(result[0])
        results_breakdown["OVERALL"] = torch.zeros_like(result[0])

        for i in range(num_labels):
            key = label_list[i]
            bio = key[0]
            key_without_BIO = key[2:]
            # 1. ignore background classes (i.e. those which start with O-)
            if bio != "O":
                # 2. group B- and I- results for the same class
                if key_without_BIO not in results_breakdown:
                    results_breakdown[key_without_BIO] = result[i]
                else:
                    results_breakdown[key_without_BIO] += result[i]
                results_breakdown["OVERALL"] += result[i]
                # 3. create summary precision, recall, f1, and accuracy also for KILE and LIR, separately
                if key_without_BIO != "LI" and not key_without_BIO.startswith("line_item_"):
                    # KILE
                    results_breakdown["KILE"] += result[i]
                if key_without_BIO != "LI" and key_without_BIO.startswith("line_item_"):
                    # LIR
                    results_breakdown["LIR"] += result[i]

        out = {}
        for key, (tp, fp, tn, fn, sup) in results_breakdown.items():
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            out[f"{key}_precision"] = precision if not precision.isnan() else torch.tensor(0.0)
            out[f"{key}_recall"] = recall if not recall.isnan() else torch.tensor(0.0)
            out[f"{key}_f1"] = f1 if not f1.isnan() else torch.tensor(0.0)
            out[f"{key}_accuracy"] = accuracy if not accuracy.isnan() else torch.tensor(0.0)
            out[f"{key}_support"] = sup

        ret = {}
        for key, value in out.items():
            if key.startswith("KILE") or key.startswith("LI") or key.startswith("OVERALL"):
                ret[key] = value
            elif args.report_all_metrics:
                ret[key] = value

        return ret

    # instantiate HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    # save the trained model
    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    print(f"Saving the best model to {best_model_path}")
    trainer.save_model(best_model_path)

    trainer.log_metrics("train", metrics)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # compute stats on train_dataset (skip for synthetic dataset, otherwise it can result in memory issues)
    if "synthetic" not in args.split:
        trainer.eval_dataset = train_dataset
        metrics = trainer.evaluate()
        trainer.log_metrics("train-eval", metrics)

    print(f"Tensorboard logs: {os.path.join(args.output_dir, 'runs')}")
    print(f"Best model saved to {best_model_path}")

    show_summary(args, __file__)

    print(f"{datetime.now()} Finished.")
