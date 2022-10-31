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
