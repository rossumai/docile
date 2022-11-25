import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from docile.dataset.cached_object import CachedObject
from docile.dataset.field import Field
from docile.dataset.pcc import PCC, calculate_pccs

logger = logging.getLogger(__name__)


class DocumentOCR(CachedObject[Dict]):
    _model = None

    def __init__(self, path: Path, pdf_path: Path) -> None:
        super().__init__(path=path, mem_cache=False, disk_cache=True)
        self.pdf_path = pdf_path

    @classmethod
    def get_model(cls) -> Callable:
        from doctr.models import ocr_predictor

        if cls._model:
            return cls._model

        logger.info("Initializing OCR predictor model.")
        cls._model = ocr_predictor(pretrained=True)
        return cls._model

    def from_disk(self) -> Dict:
        return json.loads(self.path.read_text())

    def to_disk(self, content: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(content))

    def predict(self) -> Dict:
        """Predict the OCR. Load dependencies inside."""
        from doctr.io import DocumentFile

        pdf_doc = DocumentFile.from_pdf(self.pdf_path)

        ocr_pred = self.get_model()(pdf_doc)
        return ocr_pred.export()

    def get_all_words(self, page: int) -> List[Field]:
        def yield_next_word(_page: int) -> Generator:
            ocr_page = self.content["pages"][_page]
            for block in ocr_page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        yield Field.from_ocr(word, page)

        return list(yield_next_word(page))

    def get_all_pccs(self, page: Optional[int] = None) -> List[PCC]:
        """Get all Pseudo Character Boxes (PCCs) for the whole document or on a given page."""
        if page is None:
            pages = list(range(len(self.content["pages"])))
        else:
            pages = [page]

        pccs = []
        for page in pages:
            for word in self.get_all_words(page):
                if word.text is None or word.text == "":
                    logger.debug(f"Cannot generate PCCs for OCR word with empty text: {word}")
                    continue
                pccs.extend(calculate_pccs(word.bbox, word.text, page))

        return pccs
