import dataclasses
import functools
import logging
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List, Sequence, Set

from docile.dataset import BBox, Document, Field

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PCC:
    """Class for a single Pseudo-Character Center (PCC), i.e., a position in the document."""

    x: float
    y: float
    page: int


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
        """Return all pccs on `page` covered by `bbox`."""

        # All pccs on the page are sorted by x and y coordinates. Then we find all pccs between
        # [bbox.left, bbox.right] and all pccs between [bbox.top, bbox.bottom] and return pccs in
        # the intersection of these two sets.
        sorted_x_pccs = self._page_to_sorted_x_pccs[page]
        sorted_x_pccs_x_only = [pcc.x for pcc in sorted_x_pccs]
        i_l = bisect_left(sorted_x_pccs_x_only, bbox.left)
        i_r = bisect_right(sorted_x_pccs_x_only, bbox.right)
        x_subset = set(sorted_x_pccs[i_l:i_r])

        sorted_y_pccs = self._page_to_sorted_y_pccs[page]
        sorted_y_pccs_y_only = [pcc.y for pcc in sorted_y_pccs]
        i_t = bisect_left(sorted_y_pccs_y_only, bbox.top)
        i_b = bisect_right(sorted_y_pccs_y_only, bbox.bottom)
        y_subset = set(sorted_y_pccs[i_t:i_b])

        return x_subset.intersection(y_subset)


def get_document_pccs(document: Document) -> PCCSet:
    """
    Get all Pseudo-Character Centers (PCCs) for the whole document.

    PCCs are computed from all OCR words that were snapped to the text.
    """
    pccs = []
    for word in _get_snapped_ocr_words(document):
        if word.text is None or word.text == "":
            logger.debug(f"Cannot generate PCCs for OCR word with empty text: {word}")
            continue
        pccs.extend(_calculate_pccs(word.bbox, word.text, word.page))

    return PCCSet(pccs)


def _get_snapped_ocr_words(document: Document) -> List[Field]:
    """Get OCR words snapped to the text."""
    words = []
    with document:
        for page in range(document.annotation.page_count):
            words.extend(
                document.ocr.get_all_words(
                    page=page,
                    snapped=True,
                    use_cached_snapping=True,
                    get_page_image=functools.partial(document.page_image, page),
                )
            )
    return words


def _calculate_pccs(bbox: BBox, text: str, page: int) -> List[PCC]:
    if text == "":
        raise ValueError("Cannot calculate PCCs from empty text")

    char_width = (bbox.right - bbox.left) / len(text)
    y_middle = (bbox.top + bbox.bottom) / 2
    return [
        PCC(x=bbox.left + (i + 1 / 2) * char_width, y=y_middle, page=page)
        for i in range(len(text))
    ]
