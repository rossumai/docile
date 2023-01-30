import json
from typing import Any, Dict, List, Optional, Tuple

from docile.dataset.cached_object import CachedObject, CachingConfig
from docile.dataset.field import Field
from docile.dataset.paths import PathMaybeInZip
from docile.dataset.table_grid import TableGrid


class DocumentAnnotation(CachedObject[Dict]):
    """
    All annotations available for the document.

    Unlabeled documents still have some annotations available, namely: page_count, cluster_id,
    page_image_size_at_200dpi, source, original_filename. Notice that pre-computed OCR, which is
    also available for unlabeled documents, is provided separately through `document.ocr`
    (DocumentOCR class).

    This is also true for the test set with the exception of cluster_id which is not shared.

    Otherwise, all annotations are available for both annotated and synthetic documents, with the
    exception of `template_document_id` which is only defined for the synthetic documents.

    For synthetic documents, the values for document_type, source and original_filename are taken
    from the template document.
    """

    def __init__(self, path: PathMaybeInZip, cache: CachingConfig = CachingConfig.DISK) -> None:
        super().__init__(path=path, cache=cache)

    def from_disk(self) -> Dict[str, Any]:
        return json.loads(self.path.read_bytes())

    @property
    def page_count(self) -> int:
        return self.content["metadata"]["page_count"]

    @property
    def fields(self) -> List[Field]:
        """All KILE fields on the document."""
        return [Field.from_dict(a) for a in self.content["field_extractions"]]

    def page_fields(self, page: int) -> List[Field]:
        """KILE fields on the given page of the document."""
        return [f for f in self.fields if f.page == page]

    @property
    def li_fields(self) -> List[Field]:
        """All LI fields on the document."""
        return [Field.from_dict(a) for a in self.content["line_item_extractions"]]

    def page_li_fields(self, page: int) -> List[Field]:
        """LI fields on the given page of the document."""
        return [f for f in self.li_fields if f.page == page]

    @property
    def li_headers(self) -> List[Field]:
        """
        Fields corresponding to column headers in tables.

        Predicting these fields is not part of the LIR task.
        """
        return [Field.from_dict(a) for a in self.content["line_item_headers"]]

    def page_li_headers(self, page: int) -> List[Field]:
        """Fields corresponding to column headers in tables on the given page."""
        return [f for f in self.li_headers if f.page == page]

    @property
    def cluster_id(self) -> int:
        """
        Id of the cluster the document belongs to.

        Cluster represents a group of documents with the same layout.
        """
        return self.content["metadata"]["cluster_id"]

    def page_image_size_at_200dpi(self, page: int) -> Tuple[int, int]:
        """
        Page image size at 200 DPI.

        This is exactly equal to the generated image size when using `document.page_image(page)`
        with the default parameters.
        """
        return self.content["metadata"]["page_sizes_at_200dpi"][page]

    @property
    def document_type(self) -> str:
        return self.content["metadata"]["document_type"]

    @property
    def currency(self) -> str:
        """Document currency or 'other' if no currency is specified on the document."""
        return self.content["metadata"]["currency"]

    @property
    def language(self) -> str:
        return self.content["metadata"]["language"]

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

    @property
    def source(self) -> str:
        """Source of the document, either 'ucsf' or 'pif'."""
        return self.content["metadata"]["source"]

    @property
    def original_filename(self) -> str:
        """
        Id/filename of the document in the original source.

        Several documents can have the same original_filename if the original pdf was composed of
        several self-contained documents.
        """
        return self.content["metadata"]["original_filename"]

    @property
    def template_document_id(self) -> str:
        """
        Id of the annotated document that was used as a template for the document generation.

        Only available for synthetic documents.
        """
        return self.content["metadata"]["template_document_id"]
