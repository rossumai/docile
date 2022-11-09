import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class BBox:
    left: float
    top: float
    right: float
    bottom: float

    def to_absolute_coords(self, width: float, height: float) -> "BBox":
        return BBox(
            self.left * width,
            self.top * height,
            self.right * width,
            self.bottom * height,
        )

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def intersects(self, other: "BBox") -> bool:
        if self.left > other.right or other.left > self.right:
            return False
        if self.top > other.bottom or other.top > self.bottom:
            return False
        return True
