from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pdf2image import convert_from_path
from PIL.Image import Image

from docile.dataset.dataset_store import DatasetStore

MaxOptionalSize = Union[int, Tuple[Optional[int], Optional[int]]]


class Document:
    """Structure representing a single document, with or without annotations."""

    def __init__(self, docid: str, dataset_path: Path):
        self.docid = docid
        self.dataset_path = dataset_path

        self.annotations = DatasetStore(dataset_path).load_annotations(docid)
        self.page_count: int = self.annotations["metadata"]["page_count"]

        self._cache_page_images: Dict[MaxOptionalSize, List[Image]] = {}

    def page_image_with_fields(
        self, page: int, size: MaxOptionalSize = (None, None), cache: bool = True
    ) -> Image:
        # NOTE(simsa-st): Only header fields are displayed
        return self.page_image(page=page, size=size, cache=cache)

    def page_image(
        self, page: int, size: MaxOptionalSize = (None, None), cache: bool = True
    ) -> Image:
        """
        Image from a single page of the document.

        Parameters
        ----------
        page
            Page number to generate the image from.
        size
            Check https://pdf2image.readthedocs.io/en/latest/reference.html for documentation of
            this parameter.
        cache
            If True (default), cache the generated images for future calls. Notice that this helps
            even when generating images for multiple pages of the document only once (with
            cache=False all images are generated for each page).
        """

        return self.page_images(
            size=size,
            cache=cache,
        )[page]

    def page_images(self, size: MaxOptionalSize = (None, None), cache: bool = True) -> List[Image]:
        """Export images of individual pages."""
        if size in self._cache_page_images:
            return self._cache_page_images[size]
        images: List[Image] = convert_from_path(
            DatasetStore(self.dataset_path).pdf_path(self.docid),
            size=size,
        )
        if cache:
            self._cache_page_images[size] = images
        return images

    def clear_cache(self) -> None:
        self._cache_page_images.clear()
