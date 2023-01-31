from pathlib import Path
from types import TracebackType
from typing import Optional, Tuple, Type, Union

from PIL import Image

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
        dataset_path: Union[Path, str, DataPaths],
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
            Path to the root directory with the unzipped dataset or a path to the ZIP file with the
            dataset.
        load_annotations
            If true, annotations are loaded to memory.
        load_ocr
            If true, ocr is loaded to memory.
        cache_images
            Whether to cache images generated from the pdf to disk and/or to memory.
        """
        self.docid = docid
        self.data_paths = DataPaths(dataset_path)

        if self.data_paths.is_in_zip() and cache_images.disk_cache:
            raise ValueError("Cannot use disk cache for images when reading dataset from ZIP file")

        cache_annotation = (
            CachingConfig.DISK_AND_MEMORY if load_annotations else CachingConfig.DISK
        )
        annotation_path = self.data_paths.annotation_path(docid)
        self.annotation = DocumentAnnotation(annotation_path, cache=cache_annotation)

        cache_ocr = CachingConfig.DISK_AND_MEMORY if load_ocr else CachingConfig.DISK
        ocr_path = self.data_paths.ocr_path(docid)
        pdf_path = self.data_paths.pdf_path(docid)
        self.ocr = DocumentOCR(ocr_path, pdf_path, cache=cache_ocr)

        self.load(load_annotations, load_ocr)

        self.images = {}
        self.cache_images = cache_images

        # Page count is always cached, even when otherwise caching is turned off.
        self._page_count: Optional[int] = None

        self._open = 0

    def load(self, annotations: bool = True, ocr: bool = True) -> "Document":
        """Load the annotations and/or OCR content to memory."""
        if annotations:
            self.annotation.load()
        if ocr:
            self.ocr.load()
        return self

    def release(self, annotations: bool = True, ocr: bool = True) -> "Document":
        """Free up document resources from memory."""
        if annotations:
            self.annotation.release()
        if ocr:
            self.ocr.release()
        return self

    @property
    def page_count(self) -> int:
        if self._page_count is None:
            self._page_count = self.annotation.page_count
        return self._page_count

    def page_image(self, page: int, image_size: OptionalImageSize = (None, None)) -> Image.Image:
        """
        Get image of the requested page.

        The image size with default parameters is equal to `self.page_image_size(page)`. It is an
        image rendered from the PDF at 200 DPI. To render images at lower DPI, you can use:
        ```
        image_size = document.page_image_size(page, dpi=72)
        image = document.page_image(page, image_size)
        ```

        Parameters
        ----------
        page
            Number of the page (from 0 to page_count - 1)
        image_size
            Size of the requested image as (width, height) tuple. If both dimensions are given,
            aspect ratio is not preserved. If one dimension is None, aspect ratio is preserved with
            the second dimension determining the image size. If both dimensions are None (default),
            aspect ratio is preserved and the image is rendered at 200 DPI. The parameter can be
            also a single integer in which case the result is a square image. Check
            https://pdf2image.readthedocs.io/en/latest/reference.html for more details.
        """
        if image_size not in self.images:
            self.images[image_size] = DocumentImages(
                path=self.data_paths.cache_images_path(self.docid, image_size),
                pdf_path=self.data_paths.pdf_path(self.docid),
                page_count=self.page_count,
                size=image_size,
                cache=self.cache_images,
            )
            if self._open:
                self.images[image_size].__enter__()

        return self.images[image_size].content[page]

    def page_image_size(self, page: int, dpi: int = 200) -> Tuple[int, int]:
        """
        Get (width, height) of the page when rendered with `self.page_image(page)` at `dpi`.

        In a very few cases in the unlabeled set, the rendering fails (due to the pdf pages being
        too big) and the rendered image has size (1,1). You can skip these documents or convert the
        pdfs to images in a different way.
        """
        width_200dpi, height_200dpi = self.annotation.page_image_size_at_200dpi(page)
        image_size = (
            max(1, round(dpi / 200 * width_200dpi)),
            max(1, round(dpi / 200 * height_200dpi)),
        )
        return image_size

    def __enter__(self) -> "Document":
        self._open += 1
        if self._open == 1:
            for ctx in (self.ocr, self.annotation, *self.images.values()):
                ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._open -= 1
        if self._open == 0:
            for ctx in (self.ocr, self.annotation, *self.images.values()):
                ctx.__exit__(exc_type, exc, traceback)

    def __str__(self) -> str:
        return f"Document({self.data_paths.name}:{self.docid})"

    def __repr__(self) -> str:
        return (
            f"Document(docid={self.docid!r}, "
            f"dataset_path={self.data_paths.dataset_path.root_path!r})"
        )
