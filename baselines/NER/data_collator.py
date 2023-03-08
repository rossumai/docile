from dataclasses import dataclass
from typing import Optional, Union

from transformers.data.data_collator import DataCollatorMixin
from transformers.models.layoutlmv3.processing_layoutlmv3 import LayoutLMv3Processor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class MyMLDataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100
    return_tensors: str = "pt"

    # Note that only pytorch is supported ATM
    def torch_call(self, features):
        import torch

        # custom features
        features_mod = []
        for feat in features:
            for i in range(len(feat["input_ids"])):
                mod_feat = {}
                mod_feat["input_ids"] = feat["input_ids"][i]
                if "token_type_ids" in feat:
                    mod_feat["token_type_ids"] = feat["token_type_ids"][i]
                mod_feat["attention_mask"] = feat["attention_mask"][i]
                mod_feat["labels"] = feat["labels"][i]
                if "bboxes" in feat:
                    mod_feat["bboxes"] = feat["bboxes"][i]
                features_mod.append(mod_feat)

        # TODO (michal.uricar): cut features_mod, so it has a fixed batch_size ?

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features_mod]
            if label_name in features_mod[0].keys()
            else None
        )
        N_labels = len(labels[0][0]) if labels else None
        bboxes = (
            [feature["bboxes"] for feature in features_mod]
            if "bboxes" in features_mod[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features_mod,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [[False] * N_labels] * (sequence_length - len(label))
                for label in labels
            ]
            if bboxes:
                batch["bboxes"] = [
                    list(bbox) + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in bboxes
                ]
        else:
            batch[label_name] = [
                [[False] * N_labels] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]
            if bboxes:
                batch["bboxes"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + list(bbox) for bbox in bboxes
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


@dataclass
class MyLayoutLMv3MLDataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    # tokenizer: PreTrainedTokenizerBase
    processor: LayoutLMv3Processor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100
    return_tensors: str = "pt"

    # Note that only pytorch is supported ATM
    def torch_call(self, features):
        import torch

        # custom features
        features_mod = []
        for feat in features:
            for i in range(len(feat["input_ids"])):
                mod_feat = {}
                mod_feat["input_ids"] = feat["input_ids"][i]
                if "token_type_ids" in feat:
                    mod_feat["token_type_ids"] = feat["token_type_ids"][i]
                mod_feat["attention_mask"] = feat["attention_mask"][i]
                mod_feat["labels"] = feat["labels"][i]
                if "bbox" in feat:
                    mod_feat["bbox"] = feat["bbox"][i]
                if "pixel_values" in feat:
                    mod_feat["pixel_values"] = feat["pixel_values"][i]
                features_mod.append(mod_feat)

        # NOTE (michal.uricar): cut features_mod, so it has a fixed batch_size ?

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features_mod]
            if label_name in features_mod[0].keys()
            else None
        )
        N_labels = len(labels[0][0]) if labels else None
        batch = self.processor.tokenizer.pad(
            features_mod,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.processor.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [[False] * N_labels] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [[False] * N_labels] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {
            k: torch.tensor(v, dtype=torch.int64 if k != "pixel_values" else torch.float32)
            for k, v in batch.items()
        }
        return batch
