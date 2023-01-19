from pathlib import Path
from types import TracebackType
from typing import Optional, Type, Union

from PIL import Image

from docile.dataset.cached_object import CachingConfig
from docile.dataset.document_annotation import DocumentAnnotation
from docile.dataset.document_images import DocumentImages
from docile.dataset.document_ocr import DocumentOCR
from docile.dataset.paths import DataPaths


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
        if load_annotations:
            self.annotation.content

        cache_ocr = CachingConfig.DISK_AND_MEMORY if load_ocr else CachingConfig.DISK
        ocr_path = self.data_paths.ocr_path(docid)
        pdf_path = self.data_paths.pdf_path(docid)
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

    def page_image(self, page: int, dpi: int = 200) -> Image.Image:
        """
        Get image of the requested page.

        You can get the image size for the default DPI=200 without generating the image by using
        `document.annotation.page_size_at_200dpi(page)`.

        Parameters
        ----------
        page
            Number of the page (from 0 to page_count - 1)
        dpi
            Quality at which the image is generated from the pdf.
        """
        if dpi not in self.images:
            self.images[dpi] = DocumentImages(
                path=self.data_paths.cache_images_path(self.docid, dpi),
                pdf_path=self.data_paths.pdf_path(self.docid),
                page_count=self.page_count,
                dpi=dpi,
                cache=self.cache_images,
            )
            if self._open:
                self.images[dpi].__enter__()

        return self.images[dpi].content[page]

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

    def __str__(self) -> str:
        return f"Document({self.data_paths.name}:{self.docid})"

    def __repr__(self) -> str:
        return (
            f"Document(docid={self.docid!r}, "
            f"dataset_path={self.data_paths.dataset_path.root_path!r})"
        )
