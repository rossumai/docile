from pathlib import Path
from typing import List, Optional, Tuple, Union

from pdf2image import convert_from_path
from PIL import Image

from docile.dataset.cached import Cached

MaxOptionalSize = Union[int, Tuple[Optional[int], Optional[int]]]


class DocumentPDF(Cached[List[Image.Image]]):
    def __init__(self, path: Path, size: MaxOptionalSize = (None, None), overwrite: bool = False):
        super().__init__(
            path=path,
            mem_cache=True,
            disc_cache=False,
            overwrite=overwrite,
        )
        self.size = size

    def predict(self) -> List[Image.Image]:
        """
        Get images from the PDF Document.

        Parameters
        ----------
        size
                Check https://pdf2image.readthedocs.io/en/latest/reference.html for documentation of
                this parameter.
        """
        return convert_from_path(self.path, size=self.size)
