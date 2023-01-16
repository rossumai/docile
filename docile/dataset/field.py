import dataclasses
import warnings
from typing import Any, Mapping, Optional

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
                warnings.warn(f"Ignoring unexpected key {k}")
                dct_copy.pop(k)

        return cls(bbox=bbox, **dct_copy)

    @property
    def sorting_score(self) -> float:
        """
        Score used for sorting predictions.

        This function makes sure that all predictions with `use_only_for_ap=True` have lower score
        than the remaining predictions.
        """
        score = self.score if self.score is not None else 1
        if not 0 <= score <= 1:
            raise ValueError(f"Score of field {self} is not in range [0, 1]")

        # transform score to [0, 0.5) if use_only_for_ap == True, or to [0.5, 1] otherwise
        if self.use_only_for_ap:
            score /= 2.000001
        else:
            score = 0.5 + score / 2
        return score
