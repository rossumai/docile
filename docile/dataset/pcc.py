import dataclasses
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List, Sequence, Set

from docile.dataset.bbox import BBox


@dataclasses.dataclass(slots=True, frozen=True)
class PCC:
    """Wrapper for a position in the document."""

    x: float
    y: float
    page: int


def calculate_pccs(bbox: BBox, text: str, page: int) -> List[PCC]:
    if text == "":
        raise ValueError("Cannot calculate PCCs from empty text")

    char_width = (bbox.right - bbox.left) / len(text)
    y_middle = (bbox.top + bbox.bottom) / 2
    return [
        PCC(x=bbox.left + (i + 1 / 2) * char_width, y=y_middle, page=page)
        for i in range(len(text))
    ]


class PCCSet:
    def __init__(self, pccs: Sequence[PCC]) -> None:
        page_to_pccs = defaultdict(list)
        for pcc in pccs:
            page_to_pccs[pcc.page].append(pcc)

        self._page_to_sorted_x_pccs = {
            page: sorted(page_pccs, key=lambda p: p.x) for page, page_pccs in page_to_pccs.items()
        }
        self._page_to_sorted_y_pccs = {
            page: sorted(page_pccs, key=lambda p: p.y) for page, page_pccs in page_to_pccs.items()
        }

    def get_covered_pccs(self, bbox: BBox, page: int) -> Set[PCC]:
        """Return all pccs on 'page' covered by 'bbox'."""

        # All pccs on the page are sorted by x and y coordinates. Then we find all pccs between
        # [bbox.left, bbox.right] and all pccs between [bbox.top, bbox.bottom] and return pccs in
        # the intersection of these two sets.
        sorted_x_pccs = self._page_to_sorted_x_pccs[page]
        sorted_y_pccs = self._page_to_sorted_y_pccs[page]

        i_l = bisect_left(sorted_x_pccs, bbox.left, key=lambda p: p.x)
        i_r = bisect_right(sorted_x_pccs, bbox.right, key=lambda p: p.x)
        x_subset = set(sorted_x_pccs[i_l:i_r])

        i_t = bisect_left(sorted_y_pccs, bbox.top, key=lambda p: p.y)
        i_b = bisect_right(sorted_y_pccs, bbox.bottom, key=lambda p: p.y)
        y_subset = set(sorted_y_pccs[i_t:i_b])

        return x_subset.intersection(y_subset)
