import os
import json
from typing import List, Tuple

import ipywidgets
import plotly.graph_objects as go
from IPython.display import clear_output, display

from docile.dataset import BBox, Dataset, Field
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDYPES
import numpy as np
from base64 import standard_b64encode
from io import BytesIO
from matplotlib import cm
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt

from docile.evaluation.evaluate import evaluate_dataset
from docile.evaluation import EvaluationResult
from pathlib import Path


def pilimage_to_b64(pilimage):
    mem_f = BytesIO()
    pilimage.save(mem_f, format="png")
    encoded = standard_b64encode(mem_f.getvalue())
    return encoded


def pilimage_to_svg(pilimage):
    return [
        '<image width="',
        str(pilimage.size[0]),
        '" height="',
        str(pilimage.size[1]),
        '" xlink:href="data:image/png;base64,',
        pilimage_to_b64(pilimage).decode(),
        '"/>',
    ]


# class Callback:
#     @property
#     def name(self):
#         return getattr(self, "_name", self.__class__.__name__)

fieldtypes = ["background"] + KILE_FIELDTYPES + LIR_FIELDYPES
fieldtype_to_id = {key: i for i, key in enumerate(fieldtypes)}


def bbox_str(bbox):
    if bbox:
        return f"{bbox[0]:>#04.1f}, {bbox[1]:>#04.1f}, {bbox[2]:>#04.1f}, {bbox[3]:>#04.1f}"
    else:
        return f"<NONE>"


def get_color(fieldtype, N=len(fieldtypes)):
    cmap = (plt.get_cmap("tab20", N).colors[:, 0:3]*255).astype(np.uint8)
    cmap[0] = 0
    return cmap[fieldtype_to_id[fieldtype]]


def get_style(fieldtype):
    # color = get_color(field.fieldtype if field.fieldtype is not None else "")
    color = get_color(fieldtype if fieldtype is not None else "")
    return f"stroke:rgb{tuple(color)};stroke-width:1;fill-opacity:0.25;fill:rgb{tuple(color)}"


def show_fields(fields, img):
    WIDTH, HEIGHT = img.size
    items = [(x.bbox.to_absolute_coords(WIDTH, HEIGHT).to_tuple(), x.fieldtype, x.text) for x in fields if x is not None]
    svg_bboxes = "\n".join(
        [
            f'<g><rect x="{item[0][0]}" y="{item[0][1]}" width="{item[0][2]-item[0][0]+1}" height="{item[0][3]-item[0][1]+1}" style="{get_style(item[1])}" /><text class="hiding" x="{item[0][0]}" y="{item[0][1]}">{item[2]} | {item[1].replace("line_item_", "") if item[1] else "background"} | ({bbox_str(item[0])})</text></g>'
            for item in items
        ]
    )
    return f"{svg_bboxes}"


def get_legend(WIDTH=1980, HEIGHT=100, PER_LINE=5):
    svg_legend = []
    for i, ft in enumerate(fieldtypes):
        color = get_color(ft)
        svg_legend.append(
            f"""
<g>
    <rect x="{(i % PER_LINE)*225}" y="{(i//PER_LINE)*25}" width="25" height="25" style="stroke:rgb{tuple(color)};stroke-width:1;fill-opacity:0.25;fill:rgb{tuple(color)}" />
    <text x="{((i % PER_LINE)*225)+27}" y="{((i//PER_LINE)*25)+15}" fill="rgb{tuple(color)}" >{ft if ft else "background"}</text>
</g>
"""
        )

    svg_legend = "\n".join(svg_legend)

    to_display=f"""
<svg width="{WIDTH}" viewbox="0 0 {WIDTH} {HEIGHT}" style="margin: 0 0 0 0">
{svg_legend}
</svg>
"""
# <style>
#     svg text.hiding {{display: block;}}
#     svg g:hover text.hiding {{display: block;}}
#     svg g text.hiding {{display: block;}}
# </style>
    return to_display


class MyDatasetBrowser:
    def __init__(
        self,
        dataset: Dataset,
        evaluation_results: EvaluationResult = None,
        kile_predictions: dict = None,
        lir_predictions: dict = None,
        intermediate_predictions: dict = None,
        # display_grid: bool = False,
        callbacks: List = None,
        render_size: tuple = (1920, 1080),
        random_seed: int = None,
        sort_by: str = "kile",
    ) -> None:

        self.evaluation_results = evaluation_results

        if evaluation_results is not None:
            sorted_documents = sorted(
                dataset.documents,
                key=lambda doc: evaluation_results.get_metrics(sort_by, docid=doc.docid)["AP"],
            )
            sorted_dataset = Dataset.from_documents("sorted", sorted_documents)
            self.dataset = sorted_dataset
        else:
            self.dataset = dataset

        self.document_idx = 0
        self.document_idxs = {doc.docid: idx for idx, doc in enumerate(self.dataset.documents)}
        self._document = None
        self._page = 0
        self.kile_predictions = kile_predictions if kile_predictions is not None else {}
        self.lir_predictions = lir_predictions if lir_predictions is not None else {}
        self.intermediate_predictions = intermediate_predictions if intermediate_predictions is not None else {}
        self.RENDER_W = render_size[0]
        self.RENDER_H = render_size[1]
        self.SVG_RENDER_WIDTH = self.RENDER_W
        self._rng = np.random.default_rng(seed=random_seed)
        self.context = {}

        # try:
        #     self.evaluation_result_KILE = evaluate_dataset(dataset, kile_predictions, {})
        # except Exception as ex:
        #     self.evaluation_result_KILE = None
        # try:
        #     self.evaluation_result_LIR = evaluate_dataset(dataset, {}, lir_predictions)
        # except Exception as ex:
        #     self.evaluation_result_LIR = None


        # ---- GUI ----
        # -- navigation --
        narrow_layout = ipywidgets.Layout(flex="0 0 auto", width="auto")
        wide_layout = ipywidgets.Layout(flex="1 1 auto", width="auto")

        previous_button = ipywidgets.Button(
            disabled=False, button_style="", icon="arrow-left", layout=narrow_layout
        )
        random_button = ipywidgets.Button(
            disabled=False, button_style="", icon="gift", layout=narrow_layout
        )
        next_button = ipywidgets.Button(
            disabled=False, button_style="", icon="arrow-right", layout=narrow_layout
        )
        self.redraw_button = ipywidgets.Button(
            disabled=False, button_style="", icon="retweet", layout=narrow_layout
        )
        self.document_search_text = ipywidgets.Text(
            value=str(self.document_idx),
            disabled=False,
            continuous_update=False,
            layout=wide_layout,
        )

        self.zoom_state = {"width": "auto", "height": "100%"}

        zoom_slider = ipywidgets.FloatSlider(
            value=1.0,
            min=1.0,
            max=8.0,
            step=0.25,
            description="Zoom:",
            disabled=False,
            continuous_update=True,
            readout=False,
            layout=wide_layout,
        )
        fit_width_button = ipywidgets.Button(
            description="\u2194",
            disabled=False,
            button_style="",
            tooltip="Fit width",
            layout=narrow_layout,
        )
        fit_height_button = ipywidgets.Button(
            description="\u2195",
            disabled=False,
            button_style="",
            tooltip="Fit height",
            layout=narrow_layout,
        )

        save_button = ipywidgets.Button(
            disabled=False, button_style="", icon="save", layout=narrow_layout
        )

        navigation_bar = ipywidgets.HBox(
            (
                previous_button,
                random_button,
                next_button,
                self.redraw_button,
                self.document_search_text,
                zoom_slider,
                fit_width_button,
                fit_height_button,
                save_button,
            ),
            layout=ipywidgets.Layout(
                display="flex", flex_flow="row nowrap", align_items="stretch", width="auto"
            ),
        )

        # -- status --
        self.status_text = ipywidgets.HTML()
        self.error_text = ipywidgets.HTML()

        status_bar = ipywidgets.VBox((self.status_text, self.error_text))

        # -- config --
        overlays_toggle = []
        self.callbacks = {}

        for callback in callbacks or []:
            # button = ipywidgets.ToggleButton(value=False, description=callback.name)
            button = ipywidgets.ToggleButton(value=False, description=callback)
            button.observe(lambda _: self.redraw_page(), "value")

            # self.callbacks[callback.name] = (callback, button)
            self.callbacks[callback] = (callback, button)
            overlays_toggle.append(button)

        self.overlays_toggle = ipywidgets.HBox(
            overlays_toggle,
            layout=ipywidgets.Layout(width="auto", flex_flow="row wrap", display="flex"),
        )

        # -- main view --
        height = 70
        self.document_tabs = ipywidgets.Tab(layout=ipywidgets.Layout(height=f"{height}vh"))

        # -- log text --
        self.log_text = ipywidgets.Output()

        # -- stats --
        self.statistics_text = ipywidgets.HTML()

        # -- whole layout --
        self.layout = ipywidgets.VBox(
            (
                navigation_bar, status_bar, self.overlays_toggle, self.document_tabs,
                self.statistics_text, self.log_text
            )
        )

        # attach callbacks
        previous_button.on_click(
            lambda _: self.change_document(document_idx=(self.document_idx - 1))
        )
        random_button.on_click(
            lambda _: self.change_document(
                document_idx=self._rng.integers(len(self.document_idxs))
            )
        )
        next_button.on_click(lambda _: self.change_document(document_idx=(self.document_idx + 1)))
        self.redraw_button.on_click(lambda _: self.redraw_document())

        self.document_search_text.observe(
            lambda change: self._choose_document(change["new"]), "value"
        )

        zoom_slider.observe(
            lambda change: self.change_zoom(
                width="auto", height="{}%".format(int(100 * change["new"]))
            ),
            "value",
        )
        fit_width_button.on_click(lambda _: self.change_zoom(width="100%", height="auto"))
        fit_height_button.on_click(lambda _: self.change_zoom(width="auto", height="100%"))
        save_button.on_click(lambda _: self.save_view())

        self.document_tabs.observe(lambda change: self.redraw_page(), "selected_index")

        # init
        display(self.layout)
        self.redraw_document()

    @property
    def document(self):
        self._document = self._change_document(self.dataset.documents[self.document_idx], self._document)
        return self._document

    @property
    def page(self):
        self._page = self._change_page(
            self.document, self._page, self.document_tabs.selected_index
        )
        return self._page

    @staticmethod
    def _change_document(new_document, document):
        if document is None:
            document = new_document
        elif document.docid != new_document.docid:
            document = new_document
        return document

    @staticmethod
    def _change_page(document, page, new_page_n):
        if page != new_page_n:
            page = new_page_n
        return page

    def change_zoom(self, width=None, height=None):
        if width is not None:
            self.zoom_state["width"] = width
        if height is not None:
            self.zoom_state["height"] = height

        for child in self.document_tabs.children:
            child.layout.width = self.zoom_state["width"]
            child.layout.height = self.zoom_state["height"]

    def _choose_document(self, query):
        try:
            self.change_document(document_idx=int(query))
        except ValueError:
            self.change_document(document_id=query)

    def change_document(self, document_idx=None, document_id=None):
        if (document_idx is not None) and (document_id is None):
            document_idx = np.clip(document_idx, 0, len(self.document_idxs) - 1).tolist()
        if document_idx != self.document_idx:
            self.clear_context()
            self.document_idx = document_idx
            with self.document_search_text.hold_trait_notifications():
                self.document_search_text.value = str(self.document_idx)
                self.redraw_document()

    def clear_context(self):
        """Called whenever the document is about to be changed."""
        self.context.pop("error_message", None)

    def redraw_document(self):
        self.redraw_button.disabled = True

        self.document_tabs.children = [
            ipywidgets.HTML() for _ in range(self.document.page_count)
        ]
        self.document_tabs.selected_index = 0

        for i, _ in enumerate(self.document_tabs.children):
            self.document_tabs.set_title(i, f"{i}")

        self.change_zoom()
        self.redraw_page()

        self.redraw_button.disabled = False

    def redraw_page(self):
        """
        Progress display related stuff.
        Also is a method to handle overlay-button clicks.
        """
        self.log_text.clear_output()
        self.log_text.__enter__()
        was_redrawing = self.redraw_button.disabled
        self.redraw_button.disabled = True

        self.status_text.value = f"Rendering {self.document.docid}"

        self.error_text.value = ""

        self.svg_content = self._render_svg_content()

        height = self.document.page_image(self.page).size[1]
        svg_data = f"""
<svg style="height: 100%; max-width: none; white-space: pre" viewbox="0 0 {self.RENDER_W} {height}" >
<style>
    svg text.hiding {{display: none;}}
    svg g:hover text.hiding {{display: block;}}
    svg g text.hiding {{display: none;}}
</style>
{self.svg_content}
</svg>
        """

        self.document_tabs.children[self.page].value = svg_data

        # close log
        self.log_text.__exit__(None, None, None)

        self.error_text.value = '<strong style="color: red;">{}</strong>'.format(
            self.context.get("error_message", "")
        )

        kile_results = self.evaluation_results.get_metrics("kile", docid=self.document.docid)
        lir_results = self.evaluation_results.get_metrics("lir", docid=self.document.docid)

        legend = get_legend(PER_LINE=8, HEIGHT=200)

        self.statistics_text.value = f"""
            <h3>Legend</h3>
            {legend}
            <h1>Error Stats for {self.document.docid}_{self.page}</h1>
            <h2>KILE task:<h2>
            <table style="width: 50%;">
                <thead>
                    <tr>
                        <th>AP</th>
                        <th>f1</th>
                        <th>precision</th>
                        <th>recall</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{kile_results["AP"]}</td>
                        <td>{kile_results["f1"]}</td>
                        <td>{kile_results["precision"]}</td>
                        <td>{kile_results["recall"]}</td>
                    </tr>
                </tbody>
            </table>
            <h2>LIR task:<h2>
            <table style="width: 50%;">
                <thead>
                    <tr>
                        <th>AP</th>
                        <th>f1</th>
                        <th>precision</th>
                        <th>recall</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{lir_results["AP"]}</td>
                        <td>{lir_results["f1"]}</td>
                        <td>{lir_results["precision"]}</td>
                        <td>{lir_results["recall"]}</td>
                    </tr>
                </tbody>
            </table>
"""

    def _render_svg_content(self):
        return self._render_page_svg(self.document, self.page, self.callbacks)

    def _render_page_svg(self, document, page, callbacks):
        # img = self.dataset[document].page_image(page)
        img = document.page_image(page)
        self.error_text.value = ""
        # construct html as a list of strings, then join it (speedup)
        svg_img = pilimage_to_svg(img)
        overlay_elements = []
        for callback, toggle_button in callbacks.values():
            # print(f"DBG_INFO: {callback}, {toggle_button}, {toggle_button.value}")
            if toggle_button.value:
                if callback == "Annotations_KILE":
                    gt_kile_fields = self.document.annotation.fields
                    gt_kile_fields_page = [field for field in gt_kile_fields if field.page == page]
                    overlay_elements.extend(show_fields(gt_kile_fields_page, img))
                if callback == "Annotations_LIR":
                    gt_li_fields = self.document.annotation.li_fields
                    gt_li_fields_page = [field for field in gt_li_fields if field.page == page]
                    overlay_elements.extend(show_fields(gt_li_fields_page, img))
                if callback == "Predictions_KILE":
                    predictions = self.kile_predictions[self.document.docid]
                    predictions_page = [field for field in predictions if field.page == page]
                    overlay_elements.extend(show_fields(predictions_page, img))
                if callback == "Predictions_LIR":
                    predictions = self.lir_predictions[self.document.docid]
                    predictions_page = [field for field in predictions if field.page == page]
                    overlay_elements.extend(show_fields(predictions_page, img))
                if callback == "Predictions_intermediate":
                    predictions = self.intermediate_predictions[self.document.docid]
                    predictions_page = [field for field in predictions if field.page == page]
                    overlay_elements.extend(show_fields(predictions_page, img))
        return "".join(svg_img + overlay_elements)

    def save_view(self, dir_name="."):
        was_disabled = self.redraw_button.disabled
        self.redraw_button.disabled = True

        html_template = (
            "<!DOCTYPE html><html><body>\n"
            '<svg style="height: {vh}px; max-width: none; white-space: pre" '
            'viewbox="0 0 {w} {h}">{content}</svg>\n'
            "</body></html>\n"
        )

        os.makedirs(dir_name, exist_ok=True)

        # save html
        path = os.path.join(dir_name, "{}_{}.html".format(self.document.id(), str(self.page)))
        with open(path, "w") as f:
            f.write(
                html_template.format(
                    vh=900,
                    w=self.SVG_RENDER_WIDTH,
                    h=self.RENDER_H,
                    content=self.svg_content.encode("utf-8"),
                )
            )

        self.redraw_button.disabled = was_disabled


def load_predictions(fn: Path):
    predictions = {}
    with open(fn, "r") as json_file:
        data = json.load(json_file)
    for k, v in data.items():
        predictions[k] = []
        for f in v:
            predictions[k].append(Field.from_dict(f))
    return predictions


if __name__ == "__main__":
    import json
    from pathlib import Path
    from docile.dataset import Dataset
    from docile.dataset import Field

    DATASET_PATH = Path("/storage/pif_documents/dataset_exports/docile221221-0/")
    dataset = Dataset("test", DATASET_PATH)

    # PREDICTION_PATH=Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_intermediate_predictions.json")

    # docid_to_predictions = {}
    # if PREDICTION_PATH.exists():
    #     docid_to_predictions_raw = json.loads((PREDICTION_PATH).read_text())
    #     docid_to_predictions = {
    #         docid: [Field.from_dict(f) for f in fields]
    #         for docid, fields in docid_to_predictions_raw.items()
    #     }
    #     total_predictions = sum(len(predictions) for predictions in docid_to_predictions.values())
    #     print(f"Loaded {total_predictions} predictions for {len(docid_to_predictions)} documents")
    # else:
    #     print("No predictions found.")

    EVALUATION_PATHS = [
        Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_results_KILE.json"),
        Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_results_LIR.json")
    ]

    evaluation_results = EvaluationResult.from_files(*EVALUATION_PATHS)

    intermediate_predictions = load_predictions(Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_intermediate_predictions.json"))
    kile_predictions = load_predictions(Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_predictions_KILE.json"))
    lir_predictions = load_predictions(Path("/storage/table_extraction/predictions/NER/fullpage_multilabel/docile221221-0/LayoutLMv3_wr025/v2/test_predictions_LIR.json"))

    # callbacks = ["Annotations", "Predictions"]
    # browser = MyDatasetBrowser(dataset, kile_predictions=docid_to_predictions, callbacks=callbacks)
    callbacks = ["Annotations_KILE", "Annotations_LIR", "Predictions_KILE", "Predictions_LIR", "Predictions_intermediate"]
    sbrowser = MyDatasetBrowser(dataset, evaluation_results=evaluation_results, kile_predictions=kile_predictions, lir_predictions=lir_predictions, intermediate_predictions=intermediate_predictions, callbacks=callbacks)

    pass