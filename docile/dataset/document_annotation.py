import json
from pathlib import Path
from typing import Any, Dict, List

from docile.dataset.cached_object import CachedObject


class DocumentAnnotation(CachedObject[Dict]):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path, mem_cache=True, disk_cache=True)

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Dict]:
        return self.content["field_extractions"]
