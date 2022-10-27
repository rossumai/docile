import json
from pathlib import Path
from typing import Any, Dict, List


class DatasetStore:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def load_dataset_index(self, dataset_name: str) -> List[str]:
        docids: List[str] = json.loads(self._dataset_index_path(dataset_name).read_text())
        return docids

    def load_annotations(self, docid: str) -> Dict[str, Any]:
        annotations: Dict[str, Any] = json.loads(self._annotation_path(docid).read_text())
        return annotations

    def pdf_path(self, docid: str) -> Path:
        return self.dataset_path / "pdfs" / f"{docid}.pdf"

    def _dataset_index_path(self, dataset_name: str) -> Path:
        return (self.dataset_path / dataset_name).with_suffix(".json")

    def _annotation_path(self, docid: str) -> Path:
        return self.dataset_path / "annotations" / f"{docid}.json"
