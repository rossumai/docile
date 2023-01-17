from pathlib import Path
from types import TracebackType
from typing import Optional, Type, Union

from PIL import Image, ImageDraw

from docile.dataset.cached_object import CachingConfig
from docile.dataset.document_annotation import DocumentAnnotation
from docile.dataset.document_images import DocumentImages
from docile.dataset.document_ocr import DocumentOCR
from docile.dataset.paths import DataPaths
from docile.dataset.types import OptionalImageSize


class Document:
    """
    Structure representing a single document, with or without annotations.

    You can enter the document using the `with` statement to temporarily cache its annoations, ocr
    and generated images in memory.
    """

    def __init__(
        self,
        docid: str,
        dataset_path: Union[Path, str],
        load_annotations: bool = True,
        load_ocr: bool = True,
        cache_images: CachingConfig = CachingConfig.DISK,
    ):
        """
        Load document from the dataset path.

        You can temporarily cache document resources in memory (even when they were not loaded
        during initialization) by using it as a context manager:
        ```
        with Document("docid", "dataset_path", cache_images=CachingConfig.OFF) as document:
            for i in range(5):
                # Image is only generated once
                img = document.page_image(page=0)
        ```

        Parameters
        ----------
        docid
            Id of the document.
        dataset_path
            Path to the root directory containing the dataset, i.e., index files (`train`, `val`,
            ...) and folders with pdfs, annotations and ocr.
        load_annotations
            If true, annotations are loaded to memory.
        load_ocr
            If true, ocr is loaded to memory.
        cache_images
            Whether to cache images generated from the pdf to disk and/or to memory.
        """
        self.docid = docid
        self.dataset_paths = DataPaths(dataset_path)

        cache_annotation = (
            CachingConfig.DISK_AND_MEMORY if load_annotations else CachingConfig.DISK
        )
        annotation_path = self.dataset_paths.annotation_path(docid)
        self.annotation = DocumentAnnotation(annotation_path, cache=cache_annotation)
        if load_annotations:
            self.annotation.content

        cache_ocr = CachingConfig.DISK_AND_MEMORY if load_ocr else CachingConfig.DISK
        ocr_path = self.dataset_paths.ocr_path(docid)
        pdf_path = self.dataset_paths.pdf_path(docid)
        self.ocr = DocumentOCR(ocr_path, pdf_path, cache=cache_ocr)
        if load_ocr:
            self.ocr.content

        self.images = {}
        self.cache_images = cache_images

        # Page count is always cached, even when otherwise caching is turned off.
        self._page_count: Optional[int] = None

        self._open = False

    @property
    def page_count(self) -> int:
        if self._page_count is None:
            self._page_count = self.annotation.page_count
        return self._page_count

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
                cache=self.cache_images,
            )
            if self._open:
                self.images[image_size].__enter__()

        return self.images[image_size].content[page]

    def __enter__(self) -> "Document":
        self._open = True
        for ctx in (self.ocr, self.annotation, *self.images.values()):
            ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._open = False
        for ctx in (self.ocr, self.annotation, *self.images.values()):
            ctx.__exit__(exc_type, exc, traceback)

    def page_image_with_fields(
        self, page: int, image_size: OptionalImageSize = (None, None), show_ocr_words: bool = False
    ) -> Image.Image:
        """Return page image with bboxes representing fields."""

        page_img = self.page_image(page, image_size)

        draw_img = page_img.copy()
        draw = ImageDraw.Draw(draw_img)

        for fields, color in [
            (self.annotation.fields, "green"),
            (self.annotation.li_fields, "blue"),
        ] + ([(self.ocr.get_all_words(page), "red")] if show_ocr_words else []):
            for field in fields:
                if field.page != page:
                    continue
                scaled_bbox = field.bbox.to_absolute_coords(draw_img.width, draw_img.height)
                draw.rectangle(scaled_bbox.to_tuple(), outline=color)
        return draw_img

    def __str__(self) -> str:
        return f"Document({self.dataset_paths.dataset_path.name}/{self.docid})"

    def __repr__(self) -> str:
        return f"Document(docid={self.docid!r}, dataset_path={self.dataset_paths.dataset_path!r})"
