import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoConfig)
from my_layoutlmv3 import MyLayoutLMv3Config, MyLayoutLMv3ForTokenClassification
from data_collator import MyLayoutLMv3MLDataCollatorForTokenClassification
from transformers.models.layoutlmv3.processing_layoutlmv3 import LayoutLMv3Processor
from PIL import Image

from datasets import Dataset as ArrowDataset

from docile.dataset import Dataset
from docile.dataset.field import Field
from docile.dataset.bbox import BBox
from docile.evaluation.evaluate import evaluate_dataset

from helpers import print_docile_fields, show_summary

# KILE classes
KILE_CLASSES = [
    'account_num',
    'amount_due',
    'amount_paid',
    'amount_total_gross',
    'amount_total_net',
    'amount_total_tax',
    'bank_num',
    'bic',
    'currency_code_amount_due',
    'customer_billing_address',
    'customer_billing_name',
    'customer_delivery_address',
    'customer_delivery_name',
    'customer_id',
    'customer_order_id',
    'customer_other_address',
    'customer_other_name',
    'customer_registration_id',
    'customer_tax_id',
    'date_due',
    'date_issue',
    'document_id',
    'iban',
    'order_id',
    'payment_terms',
    'tax_detail_gross',
    'tax_detail_net',
    'tax_detail_rate',
    'tax_detail_tax',
    # 'variable_symbol',
    'payment_reference',
    'vendor_address',
    'vendor_email',
    'vendor_name',
    'vendor_order_id',
    'vendor_registration_id',
    'vendor_tax_id',
]

# LIR classes
LIR_CLASSES = [
    'line_item_amount_gross',
    'line_item_amount_net',
    'line_item_code',
    'line_item_currency',
    'line_item_date',
    'line_item_description',
    'line_item_discount_amount',
    'line_item_discount_rate',
    'line_item_hts_number',
    'line_item_order_id',
    'line_item_person_name',
    'line_item_position',
    'line_item_quantity',
    'line_item_tax',
    'line_item_tax_rate',
    'line_item_unit_price_gross',
    'line_item_unit_price_net',
    'line_item_units_of_measure',
    'line_item_weight',
]


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


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
    for ft in line_item:
        g = np.array(
            list(map(lambda y: np.abs(ft.bbox.to_tuple()[1] - y), clusters.values()))
        ).argmin()
        # ft.groups = [g]
        ft.line_item_id = [g, ft.line_item_id]
        # ft.line_item_id = [g]
    return line_item, clusters


def get_sorted_field_candidates(ocr_fields):
    fields = []
    ocr_fields, clusters = split_fields_by_text_lines(ocr_fields)
    # sort by estimated lines
    ocr_fields.sort(key=lambda x: x.line_item_id)
    # group by estimated lines
    groups = {}
    for field in ocr_fields:
        # gid = str(field.line_item_id)
        gid = str(field.line_item_id[0])
        if gid not in groups:
            groups[gid] = [field]
        else:
            groups[gid].append(field)
    for gid, fs in groups.items():
        # sort by x-axis (since we are dealing with a single line)
        fs.sort(key=lambda x: x.bbox.centroid[0])
        for f in fs:
            # lid_str = f"{f.line_item_id:04d}" if f.line_item_id else "-001"
            lid_str = f"{f.line_item_id[1]:04d}" if f.line_item_id[1] else "-001"
            f.line_item_id = [f"{lid_str}{int(gid.strip('[]')):>04d}"]
        fields.extend(fs)
    return fields, clusters


def merge_horizontally(f_tuple, sep=" "):
    f1, f2 = f_tuple
    assert f1.fieldtype == f2.fieldtype, "Trying to merge fields with non-matching fieldtype"
    if (f1 and f2) and (f1.bbox and f2.bbox):
        f1.bbox = f1.bbox.union(f2.bbox)
        # NOTE: (michal.uricar) assumption that f1 and f2 are provided in a correct order
        f1.text = f"{f1.text}{sep}{f2.text}"
    return f1


def merge_vertically(f_tuple, fields, sep="\n", THR=0.15):
    f1, f2 = f_tuple
    assert f1.fieldtype == f2.fieldtype, "Trying to merge fields with non-matching fieldtype"
    fieldtype = f1.fieldtype
    page_n = f1.page
    f3 = Field(fieldtype=fieldtype, bbox=BBox(0, 0, 0, 0), text="", page=page_n)
    if (f1 and f2) and (f1.bbox and f2.bbox):
        f3.bbox = f1.bbox.union(f2.bbox)
        # NOTE: (michal.uricar) assumption that f1 and f2 are provided in a correct order
        f3.text = f"{f1.text}{sep}{f2.text}"
        f3.groups = f1.groups
        f3.line_item_id = f1.line_item_id
    flag = True
    for f in fields:
        if (f != f1 and f != f2) and (f.fieldtype != f1.fieldtype):
            intersection = f3.bbox.intersection(f.bbox).area / f.bbox.area
            if intersection >= THR:
                flag = False
                break
    return f3 if flag else None


def merge_text_boxes(text_boxes):
    # group by fieldtype:
    groups = {}
    for field in text_boxes:
        gid = field.fieldtype
        if gid not in groups.keys():
            groups[gid] = [field]
        else:
            groups[gid].append(field)

    # 1. attempt simply merge all fields of the given detected type
    final_fields = []
    for ft, fs in groups.items():
        new_field = Field(bbox=fs[0].bbox, page=fs[0].page, line_item_id=fs[0].line_item_id, fieldtype=ft, score=0.0, text="")
        last_line = int(fs[0].groups[0][4:])
        for field in fs:
            curr_line = int(field.groups[0][4:])
            new_field.bbox = new_field.bbox.union(field.bbox)
            new_field.score += field.score
            if curr_line == last_line:
                new_field.text = f"{new_field.text} {field.text}"
            else:
                new_field.text = f"{new_field.text}\n{field.text}"
            last_line = curr_line
        # average final score
        new_field.score /= len(fs)
        # resolve line_item_id
        if ft in KILE_CLASSES:
            new_field.line_item_id = None
        if ft in LIR_CLASSES and new_field.line_item_id == None:
            new_field.line_item_id = 0
        final_fields.append(new_field)

    # 2. attempt also check the distance of text boxes, if too far, create new instance (so we should do some sort of clustering, try e.g. K-means ? )
    # TODO
    return final_fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--docile_path",
        type=Path,
        default=Path("/storage/pif_documents/dataset_exports/docile221221-0/")
    )
    parser.add_argument(
        "--overlap_thr",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
    )
    parser.add_argument(
        "--store_intermediate_results",
        action="store_true",
    )
    args, unknown_args = parser.parse_known_args()

    print(f"{datetime.now()} Started.")

    show_summary(args, __file__)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # DocILE DATASET
    dataset = Dataset(args.split, args.docile_path)

    docid_to_kile_predictions = {}
    docid_to_lir_predictions = {}

    intermediate_results = {}

    config = AutoConfig.from_pretrained(args.checkpoint)
    stride = 0
    # tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # model = MyXLMRobertaMLForTokenClassification.from_pretrained(args.checkpoint, config=config).to(device)
    model = MyLayoutLMv3ForTokenClassification.from_pretrained(args.checkpoint, config=config).to(device)

    model.eval()

    collator = MyLayoutLMv3MLDataCollatorForTokenClassification(processor)

    for document in tqdm(dataset):
        doc_id = document.docid
        # page_to_table_grids = document.annotation.content["metadata"]["page_to_table_grids"]
        gt_kile_fields = document.annotation.fields
        gt_li_fields = document.annotation.li_fields
        pred_kile_fields = []
        pred_kile_fields_final = []
        pred_li_fields = []
        pred_li_fields_final = []
        all_fields_final = []
        intermediate_fields = []
        for page in range(document.page_count):
            gt_kile_fields_page = [field for field in gt_kile_fields if field.page == page]
            gt_li_fields_page = [field for field in gt_li_fields if field.page == page]
            img = document.page_image(page)
            W, H = img.size
            img_preprocessed = np.array(img.resize((224, 224), Image.BICUBIC), dtype=np.uint8).transpose([2, 0, 1])

            for field in gt_kile_fields_page+gt_li_fields_page:
                field.bbox = field.bbox.to_absolute_coords(W, H)
            ocr = document.ocr.get_all_words(page, snapped=True)
            for ocr_field in ocr:
                ocr_field.bbox = ocr_field.bbox.to_absolute_coords(W, H)

            sorted_fields, clusters = get_sorted_field_candidates(ocr)

            text_tokens = [x.text for x in sorted_fields]
            bboxes = np.array([x.bbox.to_tuple() for x in sorted_fields])
            # groups = [x.groups for x in sorted_fields]
            groups = [x.line_item_id for x in sorted_fields]

            # bboxes.append([normalize_bbox(np.array([d[0][0], d[0][1], d[0][2], d[0][3]], dtype=np.int32), (W, H)) for d in pt_info])
            bboxes_preprocessed = [normalize_bbox(bb, (W, H)) for bb in bboxes]

            encoding = processor(
                img_preprocessed,
                text_tokens,
                boxes=bboxes_preprocessed,
                # is_split_into_words=True,
                add_special_tokens=True,
                truncation=True,
                stride=stride,
                padding=True,
                max_length=512,
                return_overflowing_tokens=True,  # important !!!
                return_length=True,
                verbose=True,
                return_tensors="pt",
            )

            length = encoding.pop("length")
            overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")

            for k in encoding.keys():
                if k != "pixel_values":
                    encoding[k] = encoding[k].to(device)
                else:
                    encoding[k] = torch.cat([x.unsqueeze(0) for x in encoding[k]]).to(device)

            outputs = model(**encoding)

            # multi-label prediction
            scores = torch.sigmoid(outputs.logits)
            predictions = torch.where(scores > 0.5, 1, 0)
            num_batch = outputs.logits.shape[0]

            # LI
            word_ids = []
            valid_ids = []
            tokens = []
            predictions_flattened = []
            # confs_flattened = []
            scores_flattened = []
            # gather info from batches
            for b_i in range(num_batch):
                if b_i > 0:
                    # start_i = model.config.stride + 1
                    star_i = stride + 1
                    # end_i = np.where(np.array(enc_tokens[b_i]) == processor.tokenizer.sep_token)[0][0]
                    end_i = np.where(np.array(encoding[b_i].tokens) == processor.tokenizer.sep_token)[0][0]
                else:
                    start_i = 1
                    end_i = -1
                # word_ids.extend(enc_word_ids[b_i][start_i:end_i])
                word_ids.extend(encoding.word_ids(b_i)[start_i:end_i])
                tokens.extend(
                    # tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"].tolist()[b_i])[start_i:end_i]
                    processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].tolist()[b_i])[start_i:end_i]
                )
                predictions_flattened.extend(predictions[b_i].tolist()[start_i:end_i])
                # confs_flattened.extend(confs[b_i].tolist()[start_i:end_i])
                scores_flattened.extend(scores[b_i].tolist()[start_i:end_i])
            word_ids = np.array(word_ids)
            valid_ids = np.where(word_ids != None)
            pred_classes_full = []
            for pred, score in zip(predictions_flattened, scores_flattened):
                pred_classes_full.append([(model.config.id2label[x], score[x]) for x in np.nonzero(pred)[0]])
            # post-process predicted classes (namely split into KILE, LIR and LI)
            pred_classes_groupped = {}
            kile = []
            lir = []
            li = []
            for pred_class in pred_classes_full:
                # print(f"pred_class = {pred_class}")
                pred_kile_classes = []
                pred_lir_classes = []
                pred_li_classes = []
                for pc in pred_class:
                    if pc[0][2:] in KILE_CLASSES or pc[0] == "O-KILE":
                        pred_kile_classes.append(pc)
                    if pc[0][2:] in LIR_CLASSES or pc[0] == "O-LIR":
                        pred_lir_classes.append(pc)
                    if pc[0] == "O-LI" or pc[0] == "B-LI" or pc[0] == "I-LI" or pc[0] == "E-LI":
                        pred_li_classes.append(pc)
                kile.append(pred_kile_classes)
                lir.append(pred_lir_classes)
                li.append(pred_li_classes)

            pred_classes_groupped["KILE"] = np.array(kile, dtype=object)
            pred_classes_groupped["LIR"] = np.array(lir, dtype=object)
            pred_classes_groupped["LI"] = np.array(li, dtype=object)

            # mark (by index to text_tokens) the beginnings and ends of the LI as predicted
            LI_beginnings = []
            LI_ends = []
            tmp_field_labels = []
            li_id = -1 if page == 0 else li_id
            for word_i, text_token in enumerate(text_tokens):
                idxs = np.where(word_ids == word_i)[0]
                if not text_token:
                    continue

                # KILE class
                pred_KILE = pred_classes_groupped["KILE"][idxs].tolist()
                KILE_preds_lengths = [len(x) for x in pred_KILE]
                N_kp = max(KILE_preds_lengths)
                if N_kp > 1:
                    # multiple predictions (thanks to multi-label formulation)
                    # treat each label separately
                    # discard prediction, if the other prediction is O-KILE ?
                    sub_token_classes = []
                    sub_token_scores = []
                    for all_pred_for_sub_token in pred_KILE:
                        sub_token_classes.append([None]*N_kp)
                        sub_token_scores.append([None]*N_kp)
                        # for sub_token_pred in all_pred_for_sub_token:
                        for tmp_i, sub_token_pred in enumerate(all_pred_for_sub_token):
                            sub_token_classes[-1][tmp_i] = sub_token_pred[0]
                            sub_token_scores[-1][tmp_i] = sub_token_pred[1]
                    stc = np.array(sub_token_classes)
                    sts = np.array(sub_token_scores)
                    n_tokens, n_preds = stc.shape
                    kile_pred = []
                    kile_score = []
                    for si in range(n_preds):
                        if np.all(stc[:, si] == stc[:, si][0]):
                            # consistent prediction for all sub-tokens
                            tmp = stc[:, si][0]
                            kile_pred.append(tmp[2:] if tmp[0] != "O" else "background")
                            kile_score.append(sts[:, si][0])
                        else:
                            # inconsistent prediction - just take the maximum scoring one for the whole word for now
                            sts[:, si] = np.where(sts[:, si]==None, 0, sts[:, si])
                            maxidx = sts[:, si].argmax()
                            tmp = stc[:, si][maxidx]
                            kile_pred.append(tmp[2:] if tmp[0] != "O" else "background")
                            kile_score.append(sts[:, si][maxidx])
                else:
                    # just 1 class predicted for each sub-token -> we can use the same approach as for single-label
                    pred_KILE = [item for sublist in pred_KILE for item in sublist]
                    pred_ents_KILE = np.array([x[0][2:] if x[0][0] != "O" else "background" for x in pred_KILE])    # remove the IOB tag
                    pred_ents_KILE_score = np.array([x[1] for x in pred_KILE])
                    if len(pred_ents_KILE):
                        if (pred_ents_KILE == pred_ents_KILE[0]).all():
                            # consistent prediction for all sub-tokens
                            kile_pred = pred_ents_KILE[0]
                            kile_score = pred_ents_KILE_score[0]
                        else:
                            # inconsistent prediction - just take the maximum scoring one for the whole word
                            # for now
                            kile_pred = pred_ents_KILE[pred_ents_KILE_score.argmax()]
                            kile_score = pred_ents_KILE_score[pred_ents_KILE_score.argmax()]
                    else:
                        kile_pred = "background"
                        kile_score = 0

                # LIR class
                pred_LIR = pred_classes_groupped["LIR"][idxs].tolist()
                pred_LIR = [item for sublist in pred_LIR for item in sublist]
                pred_ents_LIR = np.array([x[0][2:] if x[0][0] != "O" else "background" for x in pred_LIR])      # remove the IOB tag
                pred_ents_LIR_score = np.array([x[1] for x in pred_LIR])
                if len(pred_ents_LIR):
                    if (pred_ents_LIR == pred_ents_LIR[0]).all():
                        # consistent prediction for all sub-tokens
                        lir_pred = pred_ents_LIR[0]
                        lir_score = pred_ents_LIR_score[0]
                    else:
                        # inconsistent prediction - just take the maximum scoring one for the whole word
                        # for now
                        lir_pred = pred_ents_LIR[pred_ents_LIR_score.argmax()]
                        lir_score = pred_ents_LIR_score[pred_ents_LIR_score.argmax()]
                else:
                    lir_pred = "background"
                    lir_score = 0

                # LI class
                pred_LI = pred_classes_groupped["LI"][idxs].tolist()
                pred_LI = [item for sublist in pred_LI for item in sublist]
                pred_ents_LI = np.array([x[0] for x in pred_LI])
                pred_ents_LI_score = np.array([x[1] for x in pred_LI])
                if len(pred_ents_LI):
                    if (pred_ents_LI == pred_ents_LI[0]).all():
                        # consistent prediction for all sub-tokens
                        li_pred = pred_ents_LI[0]
                        li_score = pred_ents_LI_score[0]
                    else:
                        # inconsistent prediction - just take the maximum scoring one for the whole word
                        # for now
                        li_pred = pred_ents_LI[pred_ents_LI_score.argmax()]
                        li_score = pred_ents_LI_score[pred_ents_LI_score.argmax()]
                else:
                    li_pred = "O-LI"
                    li_score = 0

                group = groups[word_i]
                bbox = bboxes[word_i]

                # just for debugging
                if li_pred == "B-LI":
                    LI_beginnings.append((word_i, li_score))
                if li_pred == "E-LI":
                    LI_ends.append((word_i, li_score))
                # --------

                if li_pred == "B-LI":
                    li_id += 1

                if isinstance(kile_pred, str) and kile_pred == "background" and lir_pred == "background":
                    # TODO: add just one field
                    tmp_field_labels.append(
                        Field(
                            fieldtype="background",
                            bbox=BBox(*bbox),
                            text=text_token,
                            page=page,
                            groups=group,
                            line_item_id=li_id if li_id > 0 else None,
                            score=max(kile_score, lir_score),
                        )
                    )
                elif isinstance(kile_pred, str) and kile_pred == "background":
                    tmp_field_labels.append(
                        Field(
                            fieldtype=lir_pred,
                            bbox=BBox(*bbox),
                            text=text_token,
                            page=page,
                            groups=group,
                            line_item_id=li_id if li_id > 0 else None,
                            score=lir_score,
                        )
                    )
                elif lir_pred == "background":
                    if not isinstance(kile_pred, list):
                        kile_pred = [kile_pred]
                        kile_score = [kile_score]
                    for kp, ks in zip(kile_pred, kile_score):
                        tmp_field_labels.append(
                            Field(
                                fieldtype=kp,
                                bbox=BBox(*bbox),
                                text=text_token,
                                page=page,
                                groups=group,
                                line_item_id=li_id if li_id > 0 else None,
                                score=ks,
                            )
                        )
                else:
                    # TODO: add the field twice, once for kile and once for lir
                    if not isinstance(kile_pred, list):
                        kile_pred = [kile_pred]
                        kile_score = [kile_score]
                    for kp, ks in zip(kile_pred, kile_score):
                        tmp_field_labels.append(
                            Field(
                                fieldtype=kp,
                                bbox=BBox(*bbox),
                                text=text_token,
                                page=page,
                                groups=group,
                                line_item_id=li_id if li_id > 0 else None,
                                score=ks,
                            )
                        )
                    tmp_field_labels.append(
                        Field(
                            fieldtype=lir_pred,
                            bbox=BBox(*bbox),
                            text=text_token,
                            page=page,
                            groups=group,
                            line_item_id=li_id if li_id > 0 else None,
                            score=lir_score,
                        )
                    )

            #
            line_item_groups = {}
            for field in tmp_field_labels:
                gid = field.line_item_id
                if gid not in line_item_groups:
                    line_item_groups[gid] = [field]
                else:
                    line_item_groups[gid].append(field)

            # Merge text boxes to final predictions
            # out = []
            for _, fs in line_item_groups.items():
                # line_items = merge_text_boxes(fs)
                if args.store_intermediate_results:
                    for field in fs:
                        field2 = Field.from_dict(field.to_dict())
                        field2.bbox = field2.bbox.to_relative_coords(W, H)
                        intermediate_fields.append(field2)
                line_items = merge_text_boxes([x for x in fs if x.fieldtype != "background"])
                for field in line_items:
                    # skip background fields
                    if field.fieldtype != "background":
                        # transform back to relative coordinates
                        field.bbox = field.bbox.to_relative_coords(W, H)
                        all_fields_final.append(field)

        # add final predictions to docid_to_lir_predictions mapping
        docid_to_kile_predictions[doc_id] = [x for x in all_fields_final if x.fieldtype in KILE_CLASSES]
        docid_to_lir_predictions[doc_id] = [x for x in all_fields_final if x.fieldtype in LIR_CLASSES]
        if args.store_intermediate_results:
            intermediate_results[doc_id] = intermediate_fields

    # Store intermediate results
    if args.store_intermediate_results:
        predictions_to_store = {}
        for k, v in intermediate_results.items():
            predictions_to_store[k] = []
            for field in v:
                predictions_to_store[k].append(
                    {
                        "bbox": field.bbox.to_tuple(),
                        "page": field.page,
                        "score": field.score,
                        "text": field.text,
                        "fieldtype": field.fieldtype,
                        "line_item_id": field.line_item_id,
                        "groups": field.groups,
                    }
                )
        out_path = args.output_dir / f"{args.split}_intermediate_predictions.json"
        with open(out_path, "w") as json_file:
            json.dump(predictions_to_store, json_file)

    # Store predictions
    predictions_to_store = {}
    for k, v in docid_to_kile_predictions.items():
        predictions_to_store[k] = []
        for field in v:
            predictions_to_store[k].append(
                {
                    "bbox": field.bbox.to_tuple(),
                    "page": field.page,
                    "score": field.score,
                    "text": field.text,
                    "fieldtype": field.fieldtype,
                    "line_item_id": field.line_item_id,
                    "groups": field.groups,
                }
            )
    out_path = args.output_dir / f"{args.split}_predictions_KILE.json"
    with open(out_path, "w") as json_file:
        json.dump(predictions_to_store, json_file)

    print(f"Output stored to {out_path}")

    predictions_to_store = {}
    for k, v in docid_to_lir_predictions.items():
        predictions_to_store[k] = []
        for field in v:
            predictions_to_store[k].append(
                {
                    "bbox": field.bbox.to_tuple(),
                    "page": field.page,
                    "score": field.score,
                    "text": field.text,
                    "fieldtype": field.fieldtype,
                    "line_item_id": field.line_item_id,
                    "groups": field.groups,
                }
            )
    out_path = args.output_dir / f"{args.split}_predictions_LIR.json"
    with open(out_path, "w") as json_file:
        json.dump(predictions_to_store, json_file)

    print(f"Output stored to {out_path}")

    # Call DocILE evaluation
    print(f"RESULTS for {args.split}")

    # KILE
    evaluation_result_KILE = evaluate_dataset(dataset, docid_to_kile_predictions, {})
    print(evaluation_result_KILE.print_report())
    evaluation_result_KILE.to_file(args.output_dir / f"{args.split}_results_KILE.json")

    # LIR
    evaluation_result_LIR = evaluate_dataset(dataset, {}, docid_to_lir_predictions)
    print(evaluation_result_LIR.print_report())
    evaluation_result_LIR.to_file(args.output_dir / f"{args.split}_results_LIR.json")

    print(f"{datetime.now()} Finished.")
