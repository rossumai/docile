from docile.dataset.bbox import BBox
from docile.dataset.dataset import Dataset


def test_document_fields_getters(sample_dataset: Dataset, sample_dataset_docid: str) -> None:
    doc = sample_dataset[sample_dataset_docid]
    assert len(doc.annotation.fields) == 11
    assert {f.fieldtype for f in doc.annotation.fields} == {
        "amount_due",
        "amount_total_gross",
        "customer_billing_address",
        "customer_billing_name",
        "customer_id",
        "date_due",
        "date_issue",
        "document_id",
        "payment_reference",
        "payment_terms",
        "vendor_name",
    }
    assert len(doc.annotation.page_fields(0)) == 11
    assert doc.annotation.page_fields(1) == []

    assert len(doc.annotation.li_fields) == 40
    assert {f.line_item_id for f in doc.annotation.li_fields} == set(range(1, 9))
    assert len(doc.annotation.page_li_fields(0)) == 40

    assert len(doc.annotation.li_headers) == 7
    assert {f.line_item_id for f in doc.annotation.li_headers} == {0}
    assert len(doc.annotation.page_li_headers(0)) == 7


def test_document_metadata_getters(sample_dataset: Dataset, sample_dataset_docid: str) -> None:
    doc = sample_dataset[sample_dataset_docid]
    assert doc.annotation.page_count == 1
    assert doc.annotation.cluster_id == 554
    assert doc.annotation.page_image_size_at_200dpi(0) == [1692, 2245]
    assert doc.annotation.document_type == "tax_invoice"
    assert doc.annotation.currency == "other"
    assert doc.annotation.language == "eng"
    assert doc.annotation.source == "ucsf"
    assert doc.annotation.original_filename == "nkvc0055"


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
