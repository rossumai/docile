import dataclasses
from typing import Generic, Tuple, TypeVar

T = TypeVar("T", int, float)


@dataclasses.dataclass(frozen=True)
class BBox(Generic[T]):
    left: T
    top: T
    right: T
    bottom: T

    def to_absolute_coords(self, width: float, height: float) -> "BBox[int]":
        return BBox(
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )

    def to_relative_coords(self, width: float, height: float) -> "BBox[float]":
        return BBox(
            self.left / width,
            self.top / height,
            self.right / width,
            self.bottom / height,
        )

    def to_tuple(self) -> Tuple[T, T, T, T]:
        return self.left, self.top, self.right, self.bottom

    def intersects(self, other: "BBox") -> bool:
        if self.left > other.right or other.left > self.right:
            return False
        if self.top > other.bottom or other.top > self.bottom:
            return False
        return True
