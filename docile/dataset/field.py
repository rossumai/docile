import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from docile.dataset.bbox import BBox


@dataclasses.dataclass(frozen=True)
class Field:
    bbox: BBox
    page: int
    score: Optional[float] = None
    text: Optional[str] = None
    fieldtype: Optional[str] = None
    line_item_id: Optional[int] = None

    # The flag `use_only_for_ap` can be set for some predictions in which case these will be only
    # used for Average Precision (AP) computation but they will not be used for:
    # * f1, precision, recall, true positives, false positives, false negatives computation.
    # * Matching of line items. Line items matching is computed only once for each document (not
    #   iteratively as in AP matching) from all predictions that have use_only_for_ap=False.
    #
    # Notice that for AP it can never hurt to add additional predictions if their score is lower
    # than the score of all other predictions (across all documents) so this flag can be used for
    # predictions that would be otherwise discarded as noise.
    #
    # Warning: All predictions with this flag set to True are considered to have lower score than
    # all predictions with this flag set to False, even if provided `score` suggests otherwise.
    # This does not affect the usual use case which is to set this flag to True for all predictions
    # with score under some threshold.
    use_only_for_ap: bool = False

    @classmethod
    def from_dict(cls, dct: Mapping[str, Any]) -> "Field":
        dct_copy = dict(dct)
        bbox = BBox(*(dct_copy.pop("bbox")))

        # do not fail on extra keys (if participants save extra information), only warn.
        expected_keys = {f.name for f in dataclasses.fields(cls)}
        for k in list(dct_copy.keys()):
            if k not in expected_keys:
                warnings.warn(f"Ignoring unexpected key {k}", stacklevel=1)
                dct_copy.pop(k)

        return cls(bbox=bbox, **dct_copy)

    def to_dict(self) -> Dict[str, Any]:
        dct = dataclasses.asdict(self)
        dct["bbox"] = dataclasses.astuple(self.bbox)
        return dct

    @property
    def score_sort_key(self) -> Tuple[bool, float]:
        """
        Sort key used to sort predictions by score from highest to lowest.

        This function makes sure that all predictions with `use_only_for_ap=True` come after the
        remaining predictions.
        """
        score = self.score if self.score is not None else 0

        return (self.use_only_for_ap, -score)

    def __repr__(self) -> str:
        dataclass_fields_str = ", ".join(
            f"{dataclass_field.name}={getattr(self, dataclass_field.name)!r}"
            for dataclass_field in dataclasses.fields(self)
            if getattr(self, dataclass_field.name) != getattr(dataclass_field, "default", None)
        )
        return f"Field({dataclass_fields_str})"


def store_predictions(path: Path, docid_to_predictions: Mapping[str, Sequence[Field]]) -> None:
    path.write_text(
        json.dumps(
            {
                docid: [prediction.to_dict() for prediction in predictions]
                for docid, predictions in docid_to_predictions.items()
            },
            indent=2,
        )
    )


def load_predictions(path: Path) -> Dict[str, List[Field]]:
    docid_to_raw_predictions = json.loads(path.read_text())
    return {
        docid: [Field.from_dict(prediction) for prediction in predictions]
        for docid, predictions in docid_to_raw_predictions.items()
    }
