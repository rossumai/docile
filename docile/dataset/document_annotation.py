import json
from pathlib import Path
from typing import Any, Dict, List

from docile.dataset.cached import Cached


class DocumentAnnotation(Cached[Dict]):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path, mem_cache=True, disc_cache=True, overwrite=False)

    def from_file(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Dict]:
        return self.content["field_extractions"]
