import dataclasses
from typing import Any, Mapping, Sequence, Tuple

from docile.dataset.bbox import BBox


@dataclasses.dataclass(frozen=True)
class TableGrid:
    """Class describing a structure of a single table.

    Parameters
    ----------
    bbox: BBox
        Bounding box of the table. Notice that this is not equal to the union of bboxes of all line
        item fields of the page as the table can have some header/footer/gaps on sides etc.
    rows_bbox_with_type: Sequence[Tuple[BBox, str]]
        List of rows (in top-down order), each with a bbox (covering the whole table in the
        horizontal direction) and a row type. Possible row types are: `data`, `header`,
        `subsection-header`, `footer`, `gap`, `down-merge`, `up-merge` and `unknown`.
    columns_bbox_with_type: Sequence[Tuple[BBox, str]]
        List of columns (in left-right order), each with a bbox (covering the whole table in the
        vertical direction) and a column type. Column type is either one of the
        docile.dataset.LIR_FIELDTYPES or an empty string if the column does not contain data with
        one of the recognized field types. Sometimes not all columns are annotated, see the flag
        `missing_columns` below for details.
    missing_columns: bool
        Flag filled by annotators to indicate that the table contains more than 5 "other" columns.
        I.e., columns that do not contain data with one of the recognized field types. In this
        case, only 5 such columns are part of `columns_bbox_with_type`. Notice that this flag is
        global for the whole document, so it might be true for only one of the pages.
    missing_second_table_on_page: bool
        Flag filled by annotators to indicate that a second table exists on the page which was not
        annotated. Notice that this flag is global for the whole document, so it might be true for
        only one of the pages.
    table_border_type: str
        Filled by annotators to indicate whether the table has borders or not. Possible values are
        `no_borders`, `column_borders`, `row_borders`, `all_borders` and `unknown` (only for
        synthetic data). Notice that this was annotated globally for the whole document and not for
        the tables individually but in practice the table borders are the same on all pages.
    table_structure: str
        Filled by annotators to indicate whether the table has a basic or a more complicated
        structure. Possible values are `normal`, `merged_cells` and `structured_rows`. Notice that
        this was annotated globally for the whole document and not for the tables individually but
        in practice the table structure is the same on all pages.
    """

    bbox: BBox
    rows_bbox_with_type: Sequence[Tuple[BBox, str]]
    columns_bbox_with_type: Sequence[Tuple[BBox, str]]
    missing_columns: bool
    missing_second_table_on_page: bool
    table_border_type: str
    table_structure: str

    @classmethod
    def from_dict(cls, dct: Mapping[str, Any]) -> "TableGrid":
        dct_copy = dict(dct)
        table_bbox = BBox(*(dct_copy.pop("bbox")))
        rows = [
            (
                BBox(table_bbox.left, row["top"], table_bbox.right, row["bottom"]),
                row["row_type"],
            )
            for row in dct_copy.pop("rows")
        ]
        columns = [
            (
                BBox(column["left"], table_bbox.top, column["right"], table_bbox.bottom),
                column["column_type"],
            )
            for column in dct_copy.pop("columns")
        ]
        return cls(
            bbox=table_bbox, rows_bbox_with_type=rows, columns_bbox_with_type=columns, **dct_copy
        )
