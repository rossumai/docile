import logging
from typing import List

from pdf2image import convert_from_bytes
from PIL import Image

from docile.dataset.cached_object import CachedObject, CachingConfig
from docile.dataset.paths import DataPaths, PathMaybeInZip
from docile.dataset.types import OptionalImageSize

logger = logging.getLogger(__name__)


class DocumentImages(CachedObject[List[Image.Image]]):
    def __init__(
        self,
        path: PathMaybeInZip,
        pdf_path: PathMaybeInZip,
        page_count: int,
        size: OptionalImageSize = (None, None),
        cache: CachingConfig = CachingConfig.OFF,
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
        cache
            Whether to cache images generated from pdfs to disk and/or to memory.
        """
        super().__init__(path=path, cache=cache)
        self.pdf_path = pdf_path
        self.page_count = page_count
        self.size = size

    def from_disk(self) -> List[Image.Image]:
        images = []
        for page_i in range(self.page_count):
            page_path = DataPaths.cache_page_image_path(self.path, page_i)
            with Image.open(str(page_path)) as page_img:
                try:
                    page_img.load()
                except Exception as e:
                    logger.error(
                        f"Error while loading image {page_path}, consider removing directory "
                        f"{self.path} from cache"
                    )
                    raise e
                images.append(page_img)
        return images

    def to_disk(self, content: List[Image.Image]) -> None:
        self.path.full_path.mkdir(parents=True, exist_ok=True)
        for page_i in range(self.page_count):
            page_path = DataPaths.cache_page_image_path(self.path, page_i)
            content[page_i].save(str(page_path.full_path))

    def predict(self) -> List[Image.Image]:
        images = convert_from_bytes(self.pdf_path.read_bytes(), size=self.size)
        if len(images) != self.page_count:
            raise RuntimeError(
                f"Generated unexpected number of images: {len(images)} (expected: {self.page_count}"
            )
        return images
