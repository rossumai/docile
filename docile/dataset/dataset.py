import json
from pathlib import Path
from typing import Iterable, Union

from tqdm import tqdm

from docile.document import Document
from docile.paths import DataPaths


class Dataset:
    """Structure representing a dataset, i.e., a collection of documents."""

    def __init__(self, docids: Iterable[str], dataset_path: Path):
        self.docids = list(docids)
        self.dataset_path = dataset_path
        self.docs = {
            docid: Document(docid, dataset_path)
            for docid in tqdm(self.docids, desc="Loading documents")
        }
        self.doc2index = {d: i for (i, d) in enumerate(docids)}
        self.index2doc = {i: d for (i, d) in enumerate(docids)}

    @classmethod
    def from_file(cls, split_name: str, dataset_path: Path) -> "Dataset":
        path = DataPaths(dataset_path).dataset_index_path(split_name)
        docids = json.loads(path.read_text())
        return cls(docids, dataset_path)

    def __getitem__(self, id_or_pos: Union[str, int]) -> Document:
        if isinstance(id_or_pos, str):
            return self.docs[id_or_pos]
        elif isinstance(id_or_pos, int):
            return self[self.docids[id_or_pos]]
        raise KeyError(f"Unknown document ID or index {id_or_pos}.")

    def __iter__(self) -> Iterable[Document]:
        for docid in self.docids:
            yield self.docs[docid]

    def __len__(self) -> int:
        return len(self.docids)
