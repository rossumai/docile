from pathlib import Path
from typing import Union

from docile.dataset.types import OptionalImageSize


class DataPaths:
    def __init__(self, dataset_path: Union[Path, str]):
        self.dataset_path = Path(dataset_path)

    def dataset_index_path(self, split_name: str) -> Path:
        return (self.dataset_path / split_name).with_suffix(".json")

    def pdf_path(self, docid: str) -> Path:
        return self.dataset_path / "pdfs" / f"{docid}.pdf"

    def ocr_path(self, docid: str) -> Path:
        return self.dataset_path / "ocr" / f"{docid}.json"

    def annotation_path(self, docid: str) -> Path:
        return self.dataset_path / "annotations" / f"{docid}.json"

    def cache_images_path(self, docid: str, size: OptionalImageSize) -> Path:
        """Path to directory with cached images for the individual pages."""
        directory_name = docid
        size_tag = self._size_tag(size)
        if size_tag != "":
            directory_name += f"__{self._size_tag(size)}"
        return self.dataset_path / "cached_images" / directory_name

    @staticmethod
    def cache_page_image_path(cache_images_path: Path, page_i: int) -> Path:
        return cache_images_path / f"{page_i}.png"

    @staticmethod
    def _size_tag(size: OptionalImageSize) -> str:
        """Convert size param to string. This string is used as part of the cache path for images."""
        if size == (None, None):
            return ""
        if isinstance(size, int):
            return str(size)
        return f"{size[0]}x{size[1]}"
