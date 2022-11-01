from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BBOX = Tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)  # type: ignore
class PCC:
    """Wrapper for a position with reference to a parent."""

    x: float
    y: float
    parent: "Field"


@dataclass()
class Field:
    bbox: BBOX
    score: Optional[float] = None
    text: Optional[str] = None
    page: Optional[int] = None
    fieldtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.text:
            self.pccs = self.calculate_pccs(self.bbox, self.text)
        else:
            self.pccs = None  # type: ignore

    @classmethod
    def from_annotation(cls, word_dict: Dict) -> "Field":
        return cls(**word_dict)

    @classmethod
    def from_ocr(cls, ocr_dict: Dict) -> "Field":
        lt, rb = ocr_dict["geometry"]
        return cls(text=ocr_dict["value"], bbox=(lt[0], lt[1], rb[0], rb[1]))

    def calculate_pccs(self, bbox: BBOX, text: str) -> List[PCC]:
        """Calculate Pseudo Character Boxes (PCCs) given bbox and text."""
        l, t, r, b = bbox
        C = (r - l) / len(text)
        return [PCC(x=l + (i + 1 / 2) * C, y=(t + b) / 2, parent=self) for i in range(len(text))]

    def __hash__(self) -> int:
        return hash(id(self))
