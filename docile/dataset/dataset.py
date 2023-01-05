import json
from pathlib import Path
from typing import Iterable, Union, overload

from tqdm import tqdm

from docile.dataset.document import Document
from docile.dataset.paths import DataPaths


class Dataset:
    """Structure representing a dataset, i.e., a collection of documents."""

    def __init__(self, docids: Iterable[str], dataset_path: Path, split_name: str):
        self.docids = list(docids)
        self.dataset_path = dataset_path
        self.split_name = split_name
        self.docs = {
            docid: Document(docid, dataset_path)
            for docid in tqdm(self.docids, desc=f"Loading documents for {self.name}")
        }
        self.doc2index = {d: i for (i, d) in enumerate(docids)}
        self.index2doc = {i: d for (i, d) in enumerate(docids)}

    @classmethod
    def from_file(cls, split_name: str, dataset_path: Path) -> "Dataset":
        path = DataPaths(dataset_path).dataset_index_path(split_name)
        docids = json.loads(path.read_text())
        return cls(docids, dataset_path, split_name)

    @property
    def name(self) -> str:
        return f"{self.dataset_path.name}/{self.split_name}"

    @overload
    def __getitem__(self, id_or_pos_or_slice: Union[str, int]) -> Document:
        pass

    @overload
    def __getitem__(self, id_or_pos_or_slice: slice) -> "Dataset":
        pass

    def __getitem__(
        self, id_or_pos_or_slice: Union[str, int, slice]
    ) -> Union[Document, "Dataset"]:
        """
        Get a single document or a sliced dataset.

        The function has three possible behaviours based on the parameter type:
        * If the parameter is string, return the document with this docid
        * If the parameter is int, return the document with this index
        * If the parameter is slice, return a new Dataset representing the corresponding subset of
          documents.
        """
        if isinstance(id_or_pos_or_slice, slice):
            str_start = "" if id_or_pos_or_slice.start is None else str(id_or_pos_or_slice.start)
            str_stop = "" if id_or_pos_or_slice.stop is None else str(id_or_pos_or_slice.stop)
            if id_or_pos_or_slice.step is None:
                str_slice = f"{str_start}:{str_stop}"
            else:
                str_slice = f"{str_start}:{str_stop}:{id_or_pos_or_slice.step}"
            return self.__class__(
                docids=self.docids[id_or_pos_or_slice],
                dataset_path=self.dataset_path,
                split_name=f"{self.split_name}[{str_slice}]",
            )
        if isinstance(id_or_pos_or_slice, str):
            return self.docs[id_or_pos_or_slice]
        elif isinstance(id_or_pos_or_slice, int):
            return self[self.docids[id_or_pos_or_slice]]
        raise KeyError(f"Unknown document ID or index {id_or_pos_or_slice}.")

    def __iter__(self) -> Iterable[Document]:
        for docid in self.docids:
            yield self.docs[docid]

    def __len__(self) -> int:
        return len(self.docids)
