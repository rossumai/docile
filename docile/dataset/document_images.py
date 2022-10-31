from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image

from docile.dataset.cached import Cached
from docile.dataset.paths import DataPaths
from docile.dataset.types import OptionalImageSize


class DocumentImages(Cached[List[Image.Image]]):
    def __init__(
        self, path: Path, pdf_path: Path, page_count: int, size: OptionalImageSize = (None, None)
    ):
        """
        Convert PDF Document to images for its pages.

        Parameters
        ----------
        path
            Path to directory where images of individual pages are stored.
        pdf_path
            Path to the input pdf document.
        page_count
            Number of pages of the document.
        size
            Check https://pdf2image.readthedocs.io/en/latest/reference.html for documentation of
            this parameter.
        """
        super().__init__(path=path, mem_cache=True, disk_cache=True)
        self.pdf_path = pdf_path
        self.page_count = page_count
        self.size = size

    def from_disk(self) -> List[Image.Image]:
        images = []
        for page_i in range(self.page_count):
            page_path = DataPaths.cache_page_image_path(self.path, page_i)
            with Image.open(str(page_path)) as page_img:
                page_img.load()
                images.append(page_img)
        return images

    def to_disk(self, content: List[Image.Image]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        for page_i in range(self.page_count):
            page_path = DataPaths.cache_page_image_path(self.path, page_i)
            content[page_i].save(str(page_path))

    def predict(self) -> List[Image.Image]:
        images = convert_from_path(self.pdf_path, size=self.size)
        if len(images) != self.page_count:
            raise RuntimeError(
                f"Generated unexpected number of images: {len(images)} (expected: {self.page_count}"
            )
        return images
