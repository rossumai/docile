import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List

from docile.document.cached import Cached

logger = logging.getLogger(__name__)


class DocumentOCR(Cached[Dict]):
    _model = None

    def __init__(self, path: Path, pdf_path: Path, overwrite: bool = False) -> None:
        super().__init__(path=path, mem_cache=True, disc_cache=True, overwrite=overwrite)
        self.pdf_path = pdf_path

    @classmethod
    def get_model(cls) -> Callable:
        from doctr.models import ocr_predictor

        if cls._model:
            return cls._model

        logger.info("Initializing OCR predictor model.")
        cls._model = ocr_predictor(pretrained=True)
        return cls._model

    @classmethod
    def from_file(cls, path: Path) -> Dict:
        return json.loads(path.read_text())

    def to_file(self, content: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(content))

    def predict(self) -> Dict:
        """Predict the OCR. Load dependencies inside."""
        from doctr.io import DocumentFile

        pdf_doc = DocumentFile.from_pdf(self.pdf_path)

        ocr_pred = self.get_model()(pdf_doc)
        return ocr_pred.export()

    def get_all_words(self, page: int) -> List[Dict]:
        def yield_next_word(_page: int) -> Generator:
            ocr_page = self.content["pages"][_page]
            for block in ocr_page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        yield word

        return list(yield_next_word(page))
