import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from docile.dataset.bbox import BBox
from docile.dataset.cached_object import CachedObject
from docile.dataset.field import Field


class DocumentAnnotation(CachedObject[Dict]):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path, mem_cache=False, disk_cache=True)

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Field]:
        return [Field.from_dict(a) for a in self.content["field_extractions"]]

    @property
    def li_fields(self) -> List[Field]:
        return [Field.from_dict(a) for a in self.content["line_item_extractions"]]

    @property
    def cluster_id(self) -> int:
        """
        Id of the cluster the document belongs to.

        Cluster represents a group of documents with the same layout.
        """
        return self.content["metadata"]["cluster_id"]

    def get_table_bbox(self, page: int) -> Optional[BBox]:
        """
        Get bounding box of the whole table.

        Each page has at most one table. Notice that this is not equal to the union of bboxes of
        all line item fields of the page as the table can have some header/footer/gaps on sides
        etc.
        """
        page_str = str(page + 1)
        table_grids = self.content["metadata"]["page_to_table_grids"].get(page_str, None)
        if table_grids is None:
            return None
        assert len(table_grids) == 1  # the dataset contains at most one table for each page
        table_grid = table_grids[0]

        left = table_grid["columns"][0]["left_position"]
        right = left + table_grid["width"]
        top = table_grid["rows"][0]["top_position"]
        bottom = top + table_grid["height"]
        bbox_absolute_coords = BBox(left=left, top=top, right=right, bottom=bottom)

        page_shape = self.content["metadata"]["page_shapes"][page]
        return bbox_absolute_coords.to_relative_coords(*page_shape)
