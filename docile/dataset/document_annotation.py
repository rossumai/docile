import json
from pathlib import Path
from typing import Any, Dict, List

from docile.dataset.cached_object import CachedObject
from docile.dataset.field import Field


class DocumentAnnotation(CachedObject[Dict]):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path, mem_cache=True, disk_cache=True)
        self._fields = []
        self._li_fields = []

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Field]:
        if self._fields:
            return self._fields
        # TODO: fields without "bbox" should be removed during export # noqa
        self._fields = [
            Field.from_annotation(a) for a in self.content["field_extractions"] if "bbox" in a
        ]
        return self._fields

    @property
    def li_fields(self) -> List[Field]:
        if self._li_fields:
            return self._li_fields
        # TODO: fields without "bbox" should be removed during export # noqa
        self._li_fields = [
            Field.from_annotation(a) for a in self.content["field_extractions"] if "bbox" in a
        ]
        return self._li_fields
