from pathlib import Path

from PIL import Image, ImageDraw

from docile.dataset.document_annotation import DocumentAnnotation
from docile.dataset.document_images import DocumentImages
from docile.dataset.document_ocr import DocumentOCR
from docile.dataset.paths import DataPaths
from docile.dataset.types import OptionalImageSize


class Document:
    """Structure representing a single document, with or without annotations."""

    def __init__(self, docid: str, dataset_path: Path):
        self.docid = docid
        self.dataset_paths = DataPaths(dataset_path)

        annotation_path = self.dataset_paths.annotation_path(docid)
        self.annotation = DocumentAnnotation(annotation_path)

        ocr_path = self.dataset_paths.ocr_path(docid)
        pdf_path = self.dataset_paths.pdf_path(docid)
        self.ocr = DocumentOCR(ocr_path, pdf_path)

        self.images = {}

        self.page_count = self.annotation.page_count

    def page_image(self, page: int, image_size: OptionalImageSize = (None, None)) -> Image.Image:
        """
        Get image of the requested page.

        Parameters
        ----------
        page
            Number of the page (from 0 to page_count - 1)
        image_size
            Check https://pdf2image.readthedocs.io/en/latest/reference.html for documentation of
            this parameter.
        """
        if image_size not in self.images:
            self.images[image_size] = DocumentImages(
                path=self.dataset_paths.cache_images_path(self.docid, image_size),
                pdf_path=self.dataset_paths.pdf_path(self.docid),
                page_count=self.page_count,
                size=image_size,
            )

        return self.images[image_size].content[page]

    def page_image_with_fields(
        self, page: int, image_size: OptionalImageSize = (None, None)
    ) -> Image.Image:
        """Return page image with bboxes representing fields."""

        # NOTE(simsa-st): Only KILE fields are displayed as of now
        page_img = self.page_image(page, image_size)

        draw_img = page_img.copy()
        draw = ImageDraw.Draw(draw_img)

        for field in self.annotation.fields:
            if field.page != page:
                continue
            w, h = draw_img.size
            x1, y1, x2, y2 = field.bbox
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline="green")
        return draw_img
