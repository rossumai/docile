import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, overload

from tqdm import tqdm

from docile.dataset.document import Document
from docile.dataset.paths import DataPaths

logger = logging.getLogger(__name__)


class Dataset:
    """Structure representing a dataset, i.e., a collection of documents."""

    def __init__(
        self, split_name: str, dataset_path: Path, docids: Optional[Sequence[str]] = None
    ):
        """
        Load dataset from index file or from a custom list of document ids.

        Parameters
        ----------
        split_name
            Name of the dataset split. If there is an index file stored in the dataset folder (such
            as for `train`, `val`, `test` or `trainval`), it will be used to load the document ids.
        dataset_path
            Path to the root directory containing the dataset, i.e., index files (`train`, `val`,
            ...) and folders with pdfs, annotations and ocr.
        docids
            Custom list of document ids that are part of the dataset split.
        """
        index_path = DataPaths(dataset_path).dataset_index_path(split_name)
        docids_from_file = json.loads(index_path.read_text()) if index_path.exists() else None

        if docids is None and docids_from_file is None:
            raise ValueError(
                f"Index file at path {index_path} does not exist and no docids were passed."
            )

        if docids is not None and docids_from_file is not None and docids != docids_from_file:
            raise ValueError(
                f"Passed docids do not match the content of the index file at path {index_path}."
            )

        self.docids = docids if docids is not None else docids_from_file
        self.split_name = split_name
        self.dataset_path = dataset_path

        self.docid_to_index = {d: i for i, d in enumerate(self.docids)}
        self.docs = [
            Document(docid, dataset_path)
            for docid in tqdm(self.docids, desc=f"Loading documents for {self.name}")
        ]

    @property
    def name(self) -> str:
        return f"{self.dataset_path.name}/{self.split_name}"

    def store_index(self) -> None:
        """Store dataset index to disk."""
        index_path = DataPaths(self.dataset_path).dataset_index_path(self.split_name)
        if index_path.exists():
            raise RuntimeError(
                f"Index file for {self} already exists at path {index_path}. Delete it first if "
                "you want to overwrite it."
            )

        index_path.write_text(json.dumps(self.docids, indent=2))
        logger.info(f"Stored index for {self} to file {index_path}")

    def get_split(self, split_name: str, docids: Optional[Sequence[str]] = None) -> "Dataset":
        """Get a different split for the current dataset."""
        return self.__class__(split_name=split_name, dataset_path=self.dataset_path, docids=docids)

    def get_cluster(self, cluster_id: int) -> "Dataset":
        docids = [docid for docid in self.docids if docid.annotation.cluster_id == cluster_id]
        return self.get_split(
            split_name=f"{self.split_name}[cluster_id={cluster_id}]",
            docids=docids,
        )

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
            return self.get_split(
                split_name=f"{self.split_name}[{str_slice}]",
                docids=self.docids[id_or_pos_or_slice],
            )
        if isinstance(id_or_pos_or_slice, str):
            return self.docs[self.docid_to_index[id_or_pos_or_slice]]
        elif isinstance(id_or_pos_or_slice, int):
            return self.docs[id_or_pos_or_slice]
        raise KeyError(f"Unknown document ID or index {id_or_pos_or_slice}.")

    def __iter__(self) -> Iterable[Document]:
        return iter(self.docs)

    def __len__(self) -> int:
        return len(self.docids)

    def __str__(self) -> str:
        return f"Dataset({self.name})"

    def __repr__(self) -> str:
        return f"Dataset(split_name={self.split_name}, dataset_path={self.dataset_path})"
