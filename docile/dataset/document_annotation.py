import json
from typing import Any, Dict, List, Optional, Tuple

from docile.dataset.cached_object import CachedObject, CachingConfig
from docile.dataset.field import Field
from docile.dataset.paths import PathMaybeInZip
from docile.dataset.table_grid import TableGrid


class DocumentAnnotation(CachedObject[Dict]):
    def __init__(self, path: PathMaybeInZip, cache: CachingConfig = CachingConfig.DISK) -> None:
        super().__init__(path=path, cache=cache)

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_bytes())

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

    def page_size_at_200dpi(self, page: int) -> Tuple[int, int]:
        """Get (width, height) tuple of the pdf page when rendered at 200 DPI."""
        return self.content["metadata"]["page_sizes_at_200dpi"][page]

    def get_table_grid(self, page: int) -> Optional[TableGrid]:
        """
        Get table structure on a given page.

        While Line Items do not necessarily follow a table structure, most documents also contain
        annotation of the table structure -- bounding boxes of the table, rows (with row types) and
        columns (with column types, corresponding to line item fieldtypes). See TableGrid class for
        more info.

        Each page has at most one table. In some rare cases the table annotation might be missing
        even though the page has some line items annotated.

        Some documents have a second table present, for which the table grid is not available and
        from which line items were not extracted. Info about this is present in
        `table_grid.missing_second_table_on_page` attribute.
        """
        page_str = str(page)
        table_grid_dict = self.content["metadata"]["page_to_table_grid"].get(page_str, None)
        if table_grid_dict is None:
            return None
        return TableGrid.from_dict(table_grid_dict)
