import json
from pathlib import Path
from typing import Any, Dict, List

from docile.dataset.cached_object import CachedObject
from docile.dataset.field import Field


class DocumentAnnotation(CachedObject[Dict]):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path, mem_cache=True, disk_cache=True)

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Field]:
        return [Field.from_annotation(a) for a in self.content["field_extractions"]]

    @property
    def li_fields(self) -> List[Field]:
        return [Field.from_annotation(a) for a in self.content["line_item_extractions"]]
