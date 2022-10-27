from pathlib import Path
from typing import Iterable, Optional, Union

from tqdm import tqdm

from docile.dataset.dataset_store import DatasetStore
from docile.dataset.document import Document


class Dataset:
    """Structure representing a dataset, i.e., a collection of documents."""

    def __init__(
        self,
        docids: Iterable[str],
        dataset_path: Path,
    ):
        self.docids = list(docids)
        self.dataset_path = dataset_path
        self.docs = {}

        for docid in tqdm(self.docids, desc="Loading documents"):
            self.docs[docid] = Document(docid, dataset_path)

        self.doc2index = {d: i for (i, d) in enumerate(docids)}
        self.index2doc = {i: d for (i, d) in enumerate(docids)}

    @classmethod
    def from_file(cls, index_name: str, dataset_path: Path) -> "Dataset":
        docids = DatasetStore(dataset_path).load_dataset_index(index_name)

        return cls(docids, dataset_path)

    def __getitem__(self, id_or_pos: Union[str, int]) -> Optional[Document]:
        if isinstance(id_or_pos, str):
            return self.docs.get(id_or_pos)
        elif isinstance(id_or_pos, int):
            return self[self.docids[id_or_pos]]

    def __iter__(self) -> Iterable[Document]:
        for docid in self.docids:
            yield self.docs[docid]

    def __len__(self) -> int:
        return len(self.docids)
