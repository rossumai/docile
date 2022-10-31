from pathlib import Path


class DataPaths:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def dataset_index_path(self, split_name: str) -> Path:
        return (self.dataset_path / split_name).with_suffix(".json")

    def pdf_path(self, docid: str) -> Path:
        return self.dataset_path / "pdfs" / f"{docid}.pdf"

    def ocr_path(self, docid: str) -> Path:
        return self.dataset_path / "ocr" / f"{docid}.json"

    def annotation_path(self, docid: str) -> Path:
        return self.dataset_path / "annotations" / f"{docid}.json"

    def cache_images_path(self, docid: str, size_tag: str) -> Path:
        """Path to directory with cached images for the individual pages."""
        directory_name = docid
        if size_tag != "":
            directory_name += f"__{size_tag}"
        return self.dataset_path / "cached_images" / directory_name

    @staticmethod
    def cache_page_image_path(cache_images_path: Path, page_i: int) -> Path:
        return cache_images_path / f"{page_i}.png"
