import json
import logging
import os
from pathlib import Path
from random import Random
from typing import Iterator, List, Optional, Sequence, Union, overload

from tqdm import tqdm

from docile.dataset.cached_object import CachingConfig
from docile.dataset.document import Document
from docile.dataset.paths import DataPaths

logger = logging.getLogger(__name__)


class Dataset:
    """Structure representing a dataset, i.e., a collection of documents."""

    def __init__(
        self,
        split_name: str,
        dataset_path: Union[Path, str, DataPaths],
        load_annotations: bool = True,
        load_ocr: bool = True,
        cache_images: CachingConfig = CachingConfig.DISK,
        docids: Optional[Sequence[str]] = None,
    ):
        """
        Load dataset from index file or from a custom list of document ids.

        By default, annotations and OCR are loaded from disk into memory. This is useful for
        smaller datasets -- for 10000 pages it takes 1-2 minutes to load these resources and ~3 GB
        of memory. Note: The 'train' split has 6759 pages.

        When annotations and OCR are not loaded into memory, you can still temporarily cache
        document resources in memory by using the document as a context manager:
        ```
        with dataset[5] as document:
            # Now document.annotation, document.ocr and images generated with document.page_image()
            # are cached in memory.
        ```

        Parameters
        ----------
        split_name
            Name of the dataset split. If there is an index file stored in the dataset folder (such
            as for `train`, `val`, `test` or `trainval`), it will be used to load the document ids.
        dataset_path
            Path to the root directory with the unzipped dataset or a path to the ZIP file with the
            dataset.
        load_annotations
            If true, annotations for all documents are loaded immediately to memory.
        load_ocr
            If true, ocr for all documents are loaded immediately to memory.
        cache_images
            Whether to cache images generated from pdfs to disk and/or to memory. Use
            CachingConfig.OFF if you do not have enough disk space to store all images (e.g., for
            the unlabeled dataset).
        docids
            Custom list of document ids that are part of the dataset split.
        """
        self.split_name = split_name
        self.data_paths = DataPaths(dataset_path)

        docids_from_file = self._load_docids_from_index(split_name)
        if docids is None and docids_from_file is None:
            raise ValueError(
                f"Index file for split {split_name} does not exist and no docids were passed."
            )
        if docids is not None and docids_from_file is not None and docids != docids_from_file:
            raise ValueError(
                f"Passed docids do not match the content of the index file for split {split_name}."
            )
        docids = docids if docids is not None else docids_from_file
        assert docids is not None  # this is guaranteed thanks to the checks above

        documents = [
            Document(
                docid=docid,
                dataset_path=self.data_paths,
                load_annotations=False,
                load_ocr=False,
                cache_images=cache_images,
            )
            for docid in tqdm(
                docids,
                desc=f"Initializing documents for {self.name}",
                disable=len(docids) <= 10000,
            )
        ]
        self._set_documents(documents)
        self.load(load_annotations, load_ocr)

    def load(self, annotations: bool = True, ocr: bool = True) -> "Dataset":
        """
        Load document resources to memory.

        It can be useful to delay loading of document resources, e.g., when working with just a
        sample of a big dataset.
        ```
        dataset_sample = (
            Dataset("unlabeled", DATASET_PATH, load_annotations=False, load_ocr=False)
            .sample(10)
            .load()
        )
        ```

        Parameters
        ----------
        annotations
            If true, load annotations for all documents to memory.
        ocr
            If true, load ocr for all documents to memory.

        Returns
        -------
        Dataset (self) with loaded document resources.
        """
        if not annotations and not ocr:
            return self
        if ocr and len(self) > 10000:
            logger.warning(
                f"Loading OCR for {len(self)} documents will have a big memory footprint."
            )
        for doc in tqdm(self.documents, desc=f"Loading documents for {self.name}"):
            doc.load(annotations, ocr)
        return self

    def release(self, annotations: bool = True, ocr: bool = True) -> "Dataset":
        """Free up document resources from memory."""
        for doc in self.documents:
            doc.release(annotations, ocr)
        return self

    @property
    def name(self) -> str:
        return f"{self.data_paths.name}:{self.split_name}"

    @property
    def docids(self) -> List[str]:
        return [doc.docid for doc in self.documents]

    def get_cluster(self, cluster_id: int) -> "Dataset":
        return self.from_documents(
            split_name=f"{self.split_name}[cluster_id={cluster_id}]",
            documents=[doc for doc in self.documents if doc.annotation.cluster_id == cluster_id],
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
            return self.from_documents(
                split_name=f"{self.split_name}[{str_slice}]",
                documents=self.documents[id_or_pos_or_slice],
            )
        if isinstance(id_or_pos_or_slice, str):
            return self.documents[self.docid_to_index[id_or_pos_or_slice]]
        elif isinstance(id_or_pos_or_slice, int):
            return self.documents[id_or_pos_or_slice]
        raise KeyError(f"Unknown document ID or index {id_or_pos_or_slice}.")

    def sample(self, sample_size: int, seed: Optional[int] = None) -> "Dataset":
        """
        Return a dataset with a random subsample of the current documents.

        Parameters
        ----------
        sample_size
            Number of documents in the sample
        seed
            Random seed to be used for the subsample. If None, it will be chosen randomly.
        """
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
        rng = Random(seed)
        sample_documents = rng.sample(self.documents, sample_size)
        split_name = f"{self.split_name}[sample({sample_size},seed={seed})]"
        return self.from_documents(split_name, sample_documents)

    def __iter__(self) -> Iterator[Document]:
        """Iterate over documents in the dataset, temporarily turning on memory caching."""
        for document in self.documents:
            with document:
                yield document

    def __len__(self) -> int:
        return len(self.documents)

    def total_page_count(self) -> int:
        return sum(doc.page_count for doc in self.documents)

    def __str__(self) -> str:
        return f"Dataset({self.name})"

    def __repr__(self) -> str:
        return (
            f"Dataset(split_name={self.split_name!r}, "
            f"dataset_path={self.data_paths.dataset_path.root_path!r})"
        )

    @classmethod
    def from_documents(
        cls,
        split_name: str,
        documents: Sequence[Document],
    ) -> "Dataset":
        """
        Create a dataset directly from documents, rather than from docids.

        This is useful when the documents were already loaded once, e.g., when creating a dataset
        with just a sample of the current documents.
        """
        if len(documents) == 0:
            raise ValueError("Cannot create a dataset with no documents")

        data_paths = documents[0].data_paths
        dataset = cls(
            split_name=split_name,
            dataset_path=data_paths,
            # Do not load annotations and OCR since it might be already loaded once in `documents`.
            load_annotations=False,
            load_ocr=False,
            cache_images=CachingConfig.OFF,
            docids=[doc.docid for doc in documents],
        )
        dataset._set_documents(documents)
        return dataset

    def store_index(self) -> None:
        """Store dataset index to disk."""
        index_path = self.data_paths.dataset_index_path(self.split_name)
        if index_path.exists():
            raise RuntimeError(
                f"Index file for {self} already exists at path {index_path}. Delete it first if "
                "you want to overwrite it."
            )

        index_path.write_text(json.dumps(self.docids, indent=2))
        logger.info(f"Stored index for {self} to file {index_path}")

    def _load_docids_from_index(self, split_name: str) -> Optional[Sequence[str]]:
        """
        Load docids from the index file on disk.

        Returns
        -------
        Docids loaded from the index file or None if the file does not exist.
        """
        index_path = self.data_paths.dataset_index_path(split_name)
        if index_path.exists():
            return json.loads(index_path.read_bytes())
        return None

    def _set_documents(self, documents: Sequence[Document]) -> None:
        """Set dataset documents to the provided documents."""
        self.documents = documents
        self.docid_to_index = {doc.docid: i for i, doc in enumerate(self.documents)}
