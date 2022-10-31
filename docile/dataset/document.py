from pathlib import Path

from PIL import Image, ImageDraw

from docile.dataset.document_annotation import DocumentAnnotation
from docile.dataset.document_ocr import DocumentOCR
from docile.dataset.document_pdf import DocumentPDF, MaxOptionalSize
from docile.dataset.paths import DataPaths


class Document:
    """Structure representing a single document, with or without annotations."""

    def __init__(self, docid: str, dataset_path: Path, image_size: MaxOptionalSize = (None, None)):
        self.docid = docid
        self.dataset_paths = DataPaths(dataset_path)

        annotation_path = self.dataset_paths.annotation_path(docid)
        self.annotation = DocumentAnnotation(annotation_path)

        pdf_path = self.dataset_paths.pdf_path(docid)
        self.pdf = DocumentPDF(pdf_path, size=image_size)

        ocr_path = self.dataset_paths.ocr_path(docid)
        self.ocr = DocumentOCR(ocr_path, pdf_path)

        self.page_count = self.annotation.page_count

    def page_image_with_fields(self, page: int) -> Image.Image:
        """Return page image with bboxes representing fields."""

        # NOTE(simsa-st): Only KILE fields are displayed as of now
        page_img = self.pdf.content[page]

        draw_img = page_img.copy()
        draw = ImageDraw.Draw(draw_img)

        for field in self.annotation.fields:
            if field["page"] != page:
                continue
            w, h = draw_img.size
            x1, y1, x2, y2 = field["bbox"]
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline="green")
        return draw_img
