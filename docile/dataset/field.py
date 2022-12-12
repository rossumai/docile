import dataclasses
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

    @classmethod
    def from_annotation(cls, annotation_dict: Mapping[str, Any]) -> "Field":
        annotation_copy = dict(annotation_dict)
        bbox = BBox(*(annotation_copy.pop("bbox")))
        return cls(bbox=bbox, **annotation_copy)
