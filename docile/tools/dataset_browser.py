from typing import List, Tuple

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import clear_output, display

from docile.dataset import BBox, Dataset, Field


class DatasetBrowser:
    def __init__(
        self,
        dataset: Dataset,
        doc_i: int = 0,
        page_i: int = 0,
        kile_predictions: dict = None,
        lir_predictions: dict = None,
        display_grid: bool = False,
    ) -> None:
        """
        Dataset browser to interactively display document annotations and optionally predictions in a jupyter notebook/lab.

        Parameters
        ----------
        dataset
            A Dataset from docile.dataset.
        doc_i
            Index of document to show, as sorted in the Dataset (not document ID!).
        page_i
            Index of page to show.
        kile_predictions
            Dictionary with document IDs as keys and a lists of predicted KILE fields as values.
        lir_predictions
            Dictionary with document IDs as keys and a lists of predicted LIR fields as values.
        display_grid
            If True, show row and column annotations (imperfect, please refer to Supplementary Material for details).
        """
        self.dataset = dataset
        self.doc_i = doc_i
        self.page_i = page_i
        self.kile_predictions = kile_predictions if kile_predictions is not None else {}
        self.lir_predictions = lir_predictions if lir_predictions is not None else {}
        self.display_grid = display_grid

        self.button_prev_doc = widgets.Button(description="Previous document")
        self.button_next_doc = widgets.Button(description="Next document")
        self.button_prev_page = widgets.Button(description="Previous page")
        self.button_next_page = widgets.Button(description="Next page")
        self.output = widgets.Output()

        def next_doc_button_clicked(_b: widgets.Button) -> None:
            self.doc_i += 1
            self.page_i = 0
            self.update_output(self.doc_i, self.page_i)

        def prev_doc_button_clicked(_b: widgets.Button) -> None:
            self.doc_i -= 1
            self.page_i = 0
            self.update_output(self.doc_i, self.page_i)

        def next_page_button_clicked(_b: widgets.Button) -> None:
            self.page_i += 1
            self.update_output(self.doc_i, self.page_i)

        def prev_page_button_clicked(_b: widgets.Button) -> None:
            self.page_i -= 1
            self.update_output(self.doc_i, self.page_i)

        self.button_next_doc.on_click(next_doc_button_clicked)
        self.button_prev_doc.on_click(prev_doc_button_clicked)
        self.button_next_page.on_click(next_page_button_clicked)
        self.button_prev_page.on_click(prev_page_button_clicked)

        buttons = widgets.HBox(
            (
                self.button_prev_doc,
                self.button_next_doc,
                self.button_prev_page,
                self.button_next_page,
            )
        )
        widgets_layout = widgets.VBox((buttons, self.output))
        display(widgets_layout)
        with self.output:
            self.update_output(self.doc_i, self.page_i)

    def update_output(self, doc_i: int, page_i: int) -> None:
        self.doc_i = doc_i
        self.page_i = page_i
        self.button_prev_doc.disabled = self.doc_i == 0
        self.button_next_doc.disabled = self.doc_i == len(self.dataset) - 1
        self.button_prev_page.disabled = self.page_i == 0
        self.button_next_page.disabled = self.page_i == self.dataset[self.doc_i].page_count - 1

        with self.output:
            clear_output()
            print(  # noqa T201
                f"document {self.dataset[self.doc_i].docid} ({self.doc_i+1}/{len(self.dataset)}), "
                f"page {self.page_i+1}/{self.dataset[self.doc_i].page_count}"
            )
            self.fig = self.plot_page()

    def sorted_bboxes_and_descriptions(
        self, fields: list, merge_iou: float = 0.7, prefix: str = ""
    ) -> List[Tuple[BBox, str]]:
        # sort from largest to smallest for interactive browsing, so that smaller bboxes interact
        # on top of the larger
        fields = sorted(fields, key=lambda f: -f.bbox.area)

        already_merged_indices = set()
        bboxes_and_descriptions = []
        for i, field in enumerate(fields):
            if i in already_merged_indices:
                continue
            merged_indices = {i}
            for j in range(i + 1, len(fields)):
                if j in already_merged_indices:
                    continue
                bbox2 = fields[j].bbox
                iou = field.bbox.intersection(bbox2).area / field.bbox.union(bbox2).area
                if iou > merge_iou:
                    merged_indices.add(j)

            merged_description = "<br>".join(
                self._get_field_description(fields[k], prefix) for k in sorted(merged_indices)
            )
            for k in sorted(merged_indices):
                bboxes_and_descriptions.append((fields[k].bbox, merged_description))

            already_merged_indices.update(merged_indices)

        return bboxes_and_descriptions

    @staticmethod
    def _get_field_description(field: Field, prefix: str) -> str:
        li_suffix = f" @item {field.line_item_id}" if field.line_item_id is not None else ""
        multiline_text = field.text.replace("\n", "<br>") if field.text is not None else ""
        return f"[{prefix}{field.fieldtype}{li_suffix}]<br>{multiline_text}"

    def draw_fields(
        self, bboxes_and_descriptions: List[Tuple[BBox, str]], color: str = "RoyalBlue"
    ) -> None:
        # Add field bounding boxes
        for b, desc in bboxes_and_descriptions:
            x0 = b.left * self.scaled_width
            y0 = self.scaled_height - b.top * self.scaled_height
            x1 = b.right * self.scaled_width
            y1 = self.scaled_height - b.bottom * self.scaled_height

            self.fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line={"color": color})

            # Adding a trace with a fill, setting opacity to 0
            self.fig.add_trace(
                go.Scatter(
                    x=[x0, x0, x1, x1],
                    y=[y0, y1, y1, y0],
                    fill="toself",
                    mode="lines",
                    name="",
                    text=desc,
                    opacity=0,
                )
            )

    def plot_page(self, scale_factor: float = 0.5) -> go.Figure:
        img = self.dataset[self.doc_i].page_image(self.page_i)
        annotation = self.dataset[self.doc_i].annotation

        # Sort fields by size
        header_boxes_and_descriptions = self.sorted_bboxes_and_descriptions(
            [f for f in annotation.fields if f.page == self.page_i]
        )
        table_boxes_and_descriptions = self.sorted_bboxes_and_descriptions(
            [f for f in annotation.li_fields if f.page == self.page_i]
        )

        table_grid = annotation.get_table_grid(self.page_i)
        if table_grid is not None and self.display_grid:
            column_boxes_and_descriptions = [
                (bbox, f"Table column {col_type}")
                for bbox, col_type in table_grid.columns_bbox_with_type
            ]
            row_boxes_and_descriptions = [
                (bbox, f"Table row {row_type}")
                for bbox, row_type in table_grid.rows_bbox_with_type
            ]

        docid = self.dataset[self.doc_i].docid
        kile_preds = [f for f in self.kile_predictions.get(docid, []) if f.page == self.page_i]
        kile_predictions_boxes_and_descriptions = self.sorted_bboxes_and_descriptions(
            kile_preds, prefix="Predicted "
        )
        lir_preds = [f for f in self.lir_predictions.get(docid, []) if f.page == self.page_i]
        lir_predictions_boxes_and_descriptions = self.sorted_bboxes_and_descriptions(
            lir_preds, prefix="Predicted "
        )

        # Create figure
        self.fig = go.Figure()
        # Constants
        self.scaled_width = img.size[0] * scale_factor
        self.scaled_height = img.size[1] * scale_factor

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        self.fig.add_trace(
            go.Scatter(
                x=[0, self.scaled_width],
                y=[0, self.scaled_height],
                mode="markers",
                marker_opacity=0,
            )
        )

        # Configure axes
        self.fig.update_xaxes(visible=False, range=[0, self.scaled_width])

        self.fig.update_yaxes(
            visible=False,
            range=[0, self.scaled_height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )

        # Add image
        self.fig.add_layout_image(
            {
                "x": 0,
                "sizex": self.scaled_width,
                "y": self.scaled_height,
                "sizey": self.scaled_height,
                "xref": "x",
                "yref": "y",
                "opacity": 1.0,
                "layer": "below",
                "sizing": "stretch",
                "source": img,
            }
        )

        table_grid = annotation.get_table_grid(self.page_i)
        if table_grid is not None:
            self.draw_fields([(table_grid.bbox, "[table area]")], color="Yellow")
            if self.display_grid:
                self.draw_fields(row_boxes_and_descriptions, color="Yellow")
                self.draw_fields(column_boxes_and_descriptions, color="LightGreen")

        self.draw_fields(header_boxes_and_descriptions, color="RoyalBlue")
        self.draw_fields(table_boxes_and_descriptions, color="Green")

        self.draw_fields(kile_predictions_boxes_and_descriptions, color="Red")
        self.draw_fields(lir_predictions_boxes_and_descriptions, color="Orange")

        # Configure other layout
        self.fig.update_layout(
            width=self.scaled_width,
            height=self.scaled_height,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False,
        )
        self.fig.show(config={"doubleClick": "reset"})
        return self.fig
