import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

from docile.dataset.bbox import BBox
from docile.dataset.cached_object import CachedObject, CachingConfig
from docile.dataset.field import Field
from docile.dataset.paths import PathMaybeInZip

logger = logging.getLogger(__name__)


class DocumentOCR(CachedObject[Dict]):
    _model = None

    def __init__(
        self,
        path: PathMaybeInZip,
        pdf_path: PathMaybeInZip,
        cache: CachingConfig = CachingConfig.DISK,
    ) -> None:
        super().__init__(path=path, cache=cache)
        self.pdf_path = pdf_path

    def from_disk(self) -> Dict:
        return json.loads(self.path.read_bytes())

    def to_disk(self, content: Any) -> None:
        self.path.full_path.parent.mkdir(parents=True, exist_ok=True)
        self.path.full_path.write_text(json.dumps(content))

    def predict(self) -> Dict:
        """Predict the OCR."""
        # Load dependencies inside so that they are not needed when the pre-computed OCR is used.
        from doctr.io import DocumentFile

        pdf_doc = DocumentFile.from_pdf(self.pdf_path.read_bytes())

        ocr_pred = self.get_model()(pdf_doc)
        return ocr_pred.export()

    def get_all_words(
        self,
        page: int,
        snapped: bool = False,
        use_cached_snapping: bool = True,
        get_page_image: Optional[Callable[[], Image.Image]] = None,
    ) -> List[Field]:
        """
        Get all OCR words on a given page.

        There are two possible settings, `snapped=False` returns the original word boxes as
        returned by the OCR predictions, while `snapped=True` performs additional heuristics to
        make the bounding boxes tight around text. This is used in evaluation to make sure correct
        predictions are not penalized if they differ from ground-truth only by the amount of
        background on any side of the bounding box.

        Parameters
        ----------
        page
            Page number (0-based) for which to get all OCR words.
        snapped
            If False, use original detections. If True, use bounding boxes snapped to the text.
        use_cached_snapping
            Only used if `snapped=True`. If True, the OCR cache (including the files on disk) is
            used to load/store the snapped bounding boxes.
        get_page_image
            Only used if `snapped=True`. Not needed if `use_cached_snapping=True` and the snapped
            bounding boxes are pre-computed. This should be a function that returns the image of
            the page. It is needed to perform the snapping. Tip: make sure the image is stored in
            memory. E.g., use `lambda page_img=page_img: page_img` or open the document first
            (`with document:`) to turn on memory caching and then use
            `functools.partial(document.page_image, page)`.
        """
        # Snapped bboxes are added to the dictionary, so copy the dict first if snapping is on.
        ocr_dict_original = self.content
        ocr_dict = copy.deepcopy(ocr_dict_original) if snapped else ocr_dict_original

        words = []

        ocr_page = ocr_dict["pages"][page]
        for block in ocr_page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    bbox = DocumentOCR._get_bbox_from_ocr_word(
                        word, snapped, use_cached_snapping, get_page_image
                    )
                    words.append(Field(text=word["value"], bbox=bbox, page=page))

        # If cached snapping is used and the snapping was not pre-computed, store it in the file.
        if snapped and use_cached_snapping and ocr_dict != ocr_dict_original:
            self.overwrite(ocr_dict)

        return words

    @classmethod
    def get_model(cls) -> Callable:
        if cls._model:
            return cls._model

        import torch
        from doctr.models import ocr_predictor

        logger.info("Initializing OCR predictor model.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("DocTR using device:", device)
        cls._model = ocr_predictor(pretrained=True).to(device=device)

        return cls._model

    @staticmethod
    def _get_bbox_from_ocr_word(
        word: Dict[str, Any],
        snapped: bool,
        use_cached_snapping: bool,
        get_page_image: Optional[Callable[[], Image.Image]] = None,
    ) -> BBox:
        """Get BBox for an OCR word and perform snapping if required."""
        bbox = DocumentOCR._get_bbox_from_ocr_geometry(word["geometry"])
        if not snapped:
            return bbox
        if use_cached_snapping and "snapped_geometry" in word:
            return DocumentOCR._get_bbox_from_ocr_geometry(word["snapped_geometry"])

        if get_page_image is None:
            raise ValueError(
                "Function to get page image not provided but it is needed to perform snapping"
            )
        snapped_bbox = _snap_bbox_to_text(bbox, get_page_image())

        if use_cached_snapping:
            word["snapped_geometry"] = DocumentOCR._get_ocr_geometry_from_bbox(snapped_bbox)

        return snapped_bbox

    @staticmethod
    def _get_bbox_from_ocr_geometry(geometry: List[List[float]]) -> BBox:
        """Convert the OCR geometry to the BBox structure."""
        left_top, right_bottom = geometry
        return BBox(
            left=left_top[0], top=left_top[1], right=right_bottom[0], bottom=right_bottom[1]
        )

    @staticmethod
    def _get_ocr_geometry_from_bbox(bbox: BBox) -> List[List[float]]:
        """Convert the bounding box into OCR geometry format."""
        return [[bbox.left, bbox.top], [bbox.right, bbox.bottom]]


def _snap_bbox_to_text(bbox: BBox, page_image: Image.Image) -> BBox:
    """
    Return a new bbox that is tight around the contained text.

    This is done by binarizing the cropped part of the image and removing rows/columns from
    top/bottom/left/right that are empty or probably do not contain the required text (this is done
    by heuristics explained in detail in `_foreground_text_bbox` function).
    """
    # Load dependencies inside so that they are not needed when the pre-computed OCR is used.
    import cv2

    scaled_bbox = bbox.to_absolute_coords(page_image.width, page_image.height)
    bbox_image = page_image.crop(scaled_bbox.to_tuple())
    bbox_image = ImageOps.grayscale(bbox_image)
    bbox_image_array = np.array(bbox_image)
    threshold, bbox_image_array = cv2.threshold(
        src=bbox_image_array, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    # We assume the more frequent color corresponds to the background.
    foreground_mask = bbox_image_array != np.median(bbox_image_array)

    # Snap to foreground or use the original bounding box if snapping was unsuccessful.
    snapped_bbox_crop = _foreground_text_bbox(foreground_mask)
    if snapped_bbox_crop is None:
        return bbox

    snapped_bbox_page = BBox(
        left=snapped_bbox_crop.left + scaled_bbox.left,
        top=snapped_bbox_crop.top + scaled_bbox.top,
        right=snapped_bbox_crop.right + scaled_bbox.left,
        bottom=snapped_bbox_crop.bottom + scaled_bbox.top,
    ).to_relative_coords(page_image.width, page_image.height)

    return snapped_bbox_page


def _foreground_text_bbox(
    foreground_mask: np.ndarray,
    margin_size: int = 5,
    min_char_width_margin: int = 6,
    min_line_height_margin: int = 10,
    min_char_width_inside: int = 2,
    min_line_height_inside: int = 5,
) -> Optional[BBox]:
    """
    Locate text inside of an array representing which pixels are in the foreground.

    The location is initialized as a bbox covering the whole array and then shranked in two phases:
    * In the first phase, more aggressive shrinking is done around the margins. In this phase the
      bbox is shranked by at most `margin_size` in each direction.
    * In the second phase, shrinking can proceed without limits but is less aggressive.

    If at any point the bbox would be shranked to an empty bbox, `None` is returned.

    Shrinking is controlled by two parameters for each phase (`min_char_width_*` and
    `min_line_height_*`). They define how many consecutive non-empty columns/rows are needed to
    stop further shrinking.

    Motivation:
    The implementation and parameters are fine-tuned to the pre-computed OCR provided with the
    DocILE dataset, which in practice is not tight on the text and often contains some visual
    elements around the margins of the detected text box (such as table borders or small portions
    of the surrounding text).

    Failure scenarios:
    The shrinking can be overly aggressive in the cases when very low or narrow character (such as
    'i', 'l', ':', ',', '.') is very close to the margins or when punctuation can be separated from
    the rest of the letters (such as in word 'mini'). This is not a problem for the purposes of
    DocILE evaluation where the Pseudo-Character Centers will be shifted only marginally in these
    cases.

    Parameters
    ----------
    foreground_mask
        Two dimensional numpy array containing booleans, representing which pixels are part of the
        foreground.
    margin_size
        Shrink the bounding box aggressively in the margins of the bounding box.
    min_char_width_margin
        Number of consecutive non-empty (i.e., containing at least one foreground pixel) columns
        needed to not shrink past the current column. This applies to the first and last
        `margin_size` columns.
    min_line_height_margin
        Number of consecutive non-empty rows needed to not shrink past the current row. This
        applies to the first and last `margin_size` rows.
    min_char_width_inside
        As `min_char_width_margin` but applies to columns that are further than `margin_size` from
        the edges.
    min_line_height_inside
        As `min_line_height_margin` but applies to rows that are further than `margin_size` from
        the edges.

    Returns
    -------
    BBox around the text located in the `foreground_mask` or `None` if the localization is
    unsuccessful.
    """

    width = foreground_mask.shape[1]
    height = foreground_mask.shape[0]
    left: Optional[int] = 0
    top: Optional[int] = 0
    right: Optional[int] = width
    bottom: Optional[int] = height

    # Notice that the second phase is done twice as shrinking the bbox can mark further
    # rows/columns as empty. This could be true even after the second iteration but in practice two
    # iterations are enough. In the first phase one iteration is sufficient because the
    # rows/columns in the margin are already ignored and the bbox cannot be shranked past them.
    for stop_at, min_char_width, min_line_height in [
        (margin_size, min_char_width_margin, min_line_height_margin),
        (None, min_char_width_inside, min_line_height_inside),
        (None, min_char_width_inside, min_line_height_inside),
    ]:
        # Find non-empty rows and columns
        if stop_at is not None:
            # In the first phase, when locating non-empty rows (resp. columns), consider columns
            # (resp. rows) within the margin as background as margins often contain noise. This is
            # not done in the second phase because if some side of the bbox did not move beyond the
            # margin in the first phase, text (not noise) is probably located within this margin.
            foreground_rows = foreground_mask[:, margin_size : (width - margin_size)].any(axis=1)
            foreground_columns = foreground_mask[margin_size : (height - margin_size), :].any(
                axis=0
            )
        else:
            # In the secnod phase, consider everything outside of (left, top, right, bottom) as
            # background (as if the image was already cropped).
            foreground_rows = foreground_mask[:, left:right].any(axis=1)
            foreground_rows[:top] = 0
            foreground_rows[bottom:] = 0

            foreground_columns = foreground_mask[top:bottom, :].any(axis=0)
            foreground_columns[:left] = 0
            foreground_columns[right:] = 0

        # Finally, update the (left, top, right, bottom) bbox such that there are the prescribed
        # number of non-empty consecutive rows/columns at each side of the bbox (but stop if
        # `stop_at` rows/columns were already thrown away).
        top = _find_nonzero_sequence(
            sequence=foreground_rows,
            stop_at=stop_at,
            min_consecutive_nonzero=min_line_height,
            from_start=True,
        )
        bottom = _find_nonzero_sequence(
            sequence=foreground_rows,
            stop_at=stop_at,
            min_consecutive_nonzero=min_line_height,
            from_start=False,
        )
        left = _find_nonzero_sequence(
            sequence=foreground_columns,
            stop_at=stop_at,
            min_consecutive_nonzero=min_char_width,
            from_start=True,
        )
        right = _find_nonzero_sequence(
            sequence=foreground_columns,
            stop_at=stop_at,
            min_consecutive_nonzero=min_char_width,
            from_start=False,
        )
        if any(coord is None for coord in [left, right, top, bottom]):
            return None

    return BBox(left=left, top=top, right=right, bottom=bottom)


def _find_nonzero_sequence(
    sequence: np.ndarray, stop_at: Optional[int], min_consecutive_nonzero: int, from_start: bool
) -> Optional[int]:
    """
    Find the first (or last) subsequence of consecutive non-zero values.

    Parameters
    ----------
    sequence
        One dimensional sequence of values.
    stop_at
        Only search among the first (resp. last) `stop_at` positions.
    min_consecutive_nonzero
        Search for a subsequence of non-zero items of this length.
    from_start
        Whether to search from the start or end of the sequence.

    Returns
    -------
    Returns the first index (resp. last if `from_start` is `False`) such that
    `min_consecutive_nonzero` items starting at (resp. ending before) this position are all
    nonzero. Return `stop_at` (resp. `len(sequence) - stop_at`) if the index was not among the
    first (resp. last) `stop_at` items. Return `None` if the subsequence of prescribed length does
    not exist and `stop_at` is `None`.
    """
    if not from_start:
        pos_from_right = _find_nonzero_sequence(
            np.flip(sequence), stop_at, min_consecutive_nonzero, from_start=True
        )
        if pos_from_right is None:
            return None
        return len(sequence) - pos_from_right

    for idx in sequence.nonzero()[0]:  # iterate over the nonzero indices
        if stop_at is not None and idx > stop_at:
            break
        if (
            idx + min_consecutive_nonzero <= len(sequence)
            and sequence[idx : (idx + min_consecutive_nonzero)].all()
        ):
            return idx

    # Return the maximum allowed value or `None` if `stop_at` is not set.
    return stop_at
