import enum
import warnings
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import clear_output, display

from docile.dataset import BBox, Dataset, Field


class DisplayType(enum.Enum):
    ANNOTATION = 1
    ANNOTATION_MATCHED = 2
    ANNOTATION_UNMATCHED = 3
    PREDICTION = 4
    PREDICTION_MATCHED = 5
    PREDICTION_UNMATCHED = 6
    TABLE_AREA = 7
    TABLE_ROW = 8
    TABLE_COLUMN = 9

    def __str__(self) -> str:
        # old version of enum package without StrEnum
        d = {
            DisplayType.ANNOTATION: "Annotation",
            DisplayType.ANNOTATION_MATCHED: "Matched Annotation",
            DisplayType.ANNOTATION_UNMATCHED: "Unmatched Annotation",
            DisplayType.PREDICTION: "Prediction",
            DisplayType.PREDICTION_MATCHED: "Correct Prediction",
            DisplayType.PREDICTION_UNMATCHED: "False Prediction",
            DisplayType.TABLE_AREA: "Table Area",
            DisplayType.TABLE_ROW: "Table Row",
            DisplayType.TABLE_COLUMN: "Table Column",
        }
        return d[self]

    @property
    def prefix(self) -> str:
        type_to_prefix = {
            DisplayType.ANNOTATION: "Annotation ",
            DisplayType.ANNOTATION_MATCHED: "Matched annot. ",
            DisplayType.ANNOTATION_UNMATCHED: "Unmatched annot. ",
            DisplayType.PREDICTION: "Prediction ",
            DisplayType.PREDICTION_MATCHED: "Correct pred. ",
            DisplayType.PREDICTION_UNMATCHED: "False pred. ",
            DisplayType.TABLE_AREA: "Table area",
            DisplayType.TABLE_ROW: "Table row ",
            DisplayType.TABLE_COLUMN: "Table column ",
        }
        return type_to_prefix[self]

    @property
    def color(self) -> str:
        type_to_color = {
            DisplayType.ANNOTATION: "RoyalBlue",
            DisplayType.ANNOTATION_MATCHED: "RoyalBlue",
            DisplayType.ANNOTATION_UNMATCHED: "DarkRed",
            DisplayType.PREDICTION: "Orange",
            DisplayType.PREDICTION_MATCHED: "Green",
            DisplayType.PREDICTION_UNMATCHED: "RED",
            DisplayType.TABLE_AREA: "Yellow",
            DisplayType.TABLE_ROW: "Yellow",
            DisplayType.TABLE_COLUMN: "LightGreen",
        }
        return type_to_color[self]


@dataclass
class DisplayBox:
    box: BBox
    description: str
    display_type: DisplayType

    @property
    def color(self) -> str:
        return self.display_type.color

    @property
    def name(self) -> str:
        return str(self.display_type)


class DatasetBrowser:
    def __init__(
        self,
        dataset: Dataset,
        doc_i: int = 0,
        page_i: int = 0,
        kile_matching: Optional[Mapping] = None,
        lir_matching: Optional[Mapping] = None,
        kile_predictions: Optional[Mapping] = None,
        lir_predictions: Optional[Mapping] = None,
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
        kile_matching
            Dictionary with document IDs as keys and FieldMatching from KILE evaluation as values.
        lir_predictions
            Dictionary with document IDs as keys and FieldMatching from LIR evaluation as values.
        kile_predictions
            Dictionary with document IDs as keys and a lists of predicted KILE fields as values.
            Note: This input is ignored if kile_matching (predictions with matching from evaluation) is provided.
        lir_predictions
            Dictionary with document IDs as keys and a lists of predicted LIR fields as values.
            Note: This input is ignored if lir_matching (predictions with matching from evaluation) is provided.
        display_grid
            If True, show row and column annotations (imperfect, please refer to Supplementary Material for details).
        """
        if kile_matching is not None and kile_predictions is not None:
            warnings.warn(
                "Displaying predictions from provided kile_matching, kile_predictions are ignored.",
                stacklevel=1,
            )
        if lir_matching is not None and lir_predictions is not None:
            warnings.warn(
                "Displaying predictions from provided lir_matching, lir_predictions are ignored.",
                stacklevel=1,
            )

        self.dataset = dataset
        self.doc_i = doc_i
        self.docid = self.dataset[self.doc_i].docid
        self.page_i = page_i
        self.kile_predictions = kile_predictions
        self.lir_predictions = lir_predictions
        self.kile_matching = kile_matching
        self.lir_matching = lir_matching
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
        self.docid = self.dataset[self.doc_i].docid
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
            self.plot_page()

    def get_displayboxes_and_resolve_overlaps(
        self, fields_types: List[Tuple[Field, DisplayType]], merge_iou: float = 0.7
    ) -> List[DisplayBox]:
        # sort from largest to smallest for interactive browsing, so that smaller bboxes interact
        # on top of the larger
        fields_types = sorted(fields_types, key=lambda f: -f[0].bbox.area)

        descriptions = []
        for field, display_type in fields_types:
            descriptions.append(self._get_field_description(field, display_type.prefix))

        display_boxes = []
        for i, (field, display_type) in enumerate(fields_types):
            desc = [descriptions[i]]
            for j, (field2, _) in enumerate(fields_types):
                if i == j:
                    continue
                iou = (
                    field.bbox.intersection(field2.bbox).area / field.bbox.union(field2.bbox).area
                )
                if iou > merge_iou:
                    desc.append(descriptions[j])

            display_boxes.append(
                DisplayBox(field.bbox, description="<br>".join(desc), display_type=display_type)
            )
        return display_boxes

    @staticmethod
    def _get_field_description(field: Field, prefix: str) -> str:
        li_suffix = f" @item {field.line_item_id}" if field.line_item_id is not None else ""
        multiline_text = field.text.replace("\n", "<br>") if field.text is not None else ""
        return f"[{prefix}{field.fieldtype}{li_suffix}]<br>{multiline_text}"

    def draw_fields(self, display_boxes: List[DisplayBox]) -> None:
        displayed_types = set()
        # Add field bounding boxes
        for display_box in display_boxes:
            x0 = display_box.box.left * self.scaled_width
            y0 = self.scaled_height - display_box.box.top * self.scaled_height
            x1 = display_box.box.right * self.scaled_width
            y1 = self.scaled_height - display_box.box.bottom * self.scaled_height

            self.fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line={"color": display_box.color},
                name=display_box.name,
            )

            # Adding a trace with a fill, setting opacity to 0
            self.fig.add_trace(
                go.Scatter(
                    x=[x0, x0, x1, x1],
                    y=[y0, y1, y1, y0],
                    fill="toself",
                    mode="lines",
                    text=display_box.description,
                    name="",
                    opacity=0,
                    showlegend=False,
                )
            )
            displayed_types.add(display_box.display_type)

        for t in DisplayType:
            if t in displayed_types:
                self.fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=str(t),
                        marker={"size": 7, "color": t.color, "symbol": "square"},
                    )
                )

    def get_all_displayboxes(self) -> List[DisplayBox]:
        annotation = self.dataset[self.doc_i].annotation

        display_boxes = []

        try:
            table_grid = annotation.get_table_grid(self.page_i)
        except KeyError:
            table_grid = None
        if table_grid is not None:
            display_boxes.append(
                DisplayBox(table_grid.bbox, "[Table area]", DisplayType.TABLE_AREA)
            )
            if self.display_grid:
                display_boxes.extend(
                    [
                        DisplayBox(bbox, f"[Table column {col_type}]", DisplayType.TABLE_COLUMN)
                        for bbox, col_type in table_grid.columns_bbox_with_type
                    ]
                )
                display_boxes.extend(
                    [
                        DisplayBox(bbox, f"[Table row {row_type}]", DisplayType.TABLE_ROW)
                        for bbox, row_type in table_grid.rows_bbox_with_type
                    ]
                )

        fields_types = []

        # display KILE predictions with matching (if available) or without (if not available):
        if self.kile_matching is not None:
            if self.docid in self.kile_matching:
                fields_types.extend(
                    [
                        (f, DisplayType.PREDICTION_UNMATCHED)
                        for f in self.kile_matching[self.docid].false_positives
                        if f.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (f, DisplayType.ANNOTATION_UNMATCHED)
                        for f in self.kile_matching[self.docid].false_negatives
                        if f.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (m.pred, DisplayType.PREDICTION_MATCHED)
                        for m in self.kile_matching[self.docid].matches
                        if m.pred.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (m.gold, DisplayType.ANNOTATION_MATCHED)
                        for m in self.kile_matching[self.docid].matches
                        if m.gold.page == self.page_i
                    ]
                )
        else:
            try:
                fields_types.extend(
                    [(f, DisplayType.ANNOTATION) for f in annotation.page_fields(self.page_i)]
                )
            except KeyError:
                # annotations not available, this can happen for test set or unlabeled set
                pass
            if self.kile_predictions is not None:
                fields_types.extend(
                    [
                        (f, DisplayType.PREDICTION)
                        for f in self.kile_predictions.get(self.docid, [])
                        if f.page == self.page_i
                    ]
                )

        # display LIR predictions with matching (if available) or without (if not available):
        if self.lir_matching is not None:
            if self.docid in self.lir_matching:
                fields_types.extend(
                    [
                        (f, DisplayType.PREDICTION_UNMATCHED)
                        for f in self.lir_matching[self.docid].false_positives
                        if f.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (f, DisplayType.ANNOTATION_UNMATCHED)
                        for f in self.lir_matching[self.docid].false_negatives
                        if f.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (m.pred, DisplayType.PREDICTION_MATCHED)
                        for m in self.lir_matching[self.docid].matches
                        if m.pred.page == self.page_i
                    ]
                )
                fields_types.extend(
                    [
                        (
                            m.gold,
                            DisplayType.ANNOTATION_MATCHED,
                        )
                        for m in self.lir_matching[self.docid].matches
                        if m.gold.page == self.page_i
                    ]
                )
        else:
            try:
                fields_types.extend(
                    [(f, DisplayType.ANNOTATION) for f in annotation.page_li_fields(self.page_i)]
                )
            except KeyError:
                # annotations not available, this can happen for test set or unlabeled set
                pass
            if self.lir_predictions is not None:
                fields_types.extend(
                    [
                        (f, DisplayType.PREDICTION)
                        for f in self.lir_predictions.get(self.docid, [])
                        if f.page == self.page_i
                    ]
                )

        display_boxes.extend(self.get_displayboxes_and_resolve_overlaps(fields_types=fields_types))
        return display_boxes

    def plot_page(self, scale_factor: float = 0.5) -> None:
        img = self.dataset[self.doc_i].page_image(self.page_i)

        # Create figure
        self.fig = go.Figure()
        # Constants
        self.scaled_width = img.size[0] * scale_factor
        self.scaled_height = img.size[1] * scale_factor

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

        # prepare bboxes
        display_boxes = self.get_all_displayboxes()

        self.draw_fields(display_boxes)

        # Configure other layout
        self.fig.update_layout(
            width=self.scaled_width,
            height=self.scaled_height,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=True,
            legend={"yanchor": "top", "y": 0.9, "xanchor": "left", "x": 1},
        )
        self.fig.show(config={"doubleClick": "reset"})
