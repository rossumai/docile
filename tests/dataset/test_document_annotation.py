from docile.dataset.bbox import BBox
from docile.dataset.dataset import Dataset


def test_document_annotation_get_table_grid(
    sample_dataset: Dataset, sample_dataset_docid: str
) -> None:
    doc = sample_dataset[sample_dataset_docid]
    grid = doc.annotation.get_table_grid(page=0)
    assert grid is not None
    assert grid.bbox == BBox(
        left=133 / 1240, top=579 / 1645, right=1132 / 1240, bottom=1423 / 1645
    )

    assert len(grid.rows_bbox_with_type) == 17
    assert all(
        row[0].left == grid.bbox.left and row[0].right == grid.bbox.right
        for row in grid.rows_bbox_with_type
    )
    assert {row[1] for row in grid.rows_bbox_with_type} == {
        "header",
        "gap",
        "data",
        "gap-with-text",
        "footer",
    }

    assert len(grid.columns_bbox_with_type) == 8
    assert all(
        column[0].top == grid.bbox.top and column[0].bottom == grid.bbox.bottom
        for column in grid.columns_bbox_with_type
    )
    assert [column[1] for column in grid.columns_bbox_with_type] == [
        "line_item_quantity",
        "line_item_code",
        "line_item_description",
        "line_item_quantity",
        "",
        "line_item_quantity",
        "line_item_unit_price_gross",
        "line_item_amount_gross",
    ]

    assert not grid.missing_columns
    assert not grid.missing_second_table_on_page

    assert grid.table_border_type == "column_borders"
    assert grid.table_structure == "normal"

    assert doc.annotation.get_table_grid(page=1) is None
