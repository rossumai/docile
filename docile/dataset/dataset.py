import json
import logging
import os
from pathlib import Path
from random import Random
from typing import Iterable, List, Optional, Sequence, Union, overload

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
        dataset_path: Union[Path, str],
        docids: Optional[Sequence[str]] = None,
        load_annotations: bool = True,
        load_ocr: bool = True,
        cache_images: CachingConfig = CachingConfig.DISK,
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
            Path to the root directory containing the dataset, i.e., index files (`train`, `val`,
            ...) and folders with pdfs, annotations and ocr.
        docids
            Custom list of document ids that are part of the dataset split.
        load_annotations
            If true, annotations for all documents are loaded immediately to memory.
        load_ocr
            If true, ocr for all documents are loaded immediately to memory.
        cache_images
            Whether to cache images generated from pdfs to disk and/or to memory. Use
            CachingConfig.OFF if you do not have enough disk space to store all images (e.g., for
            the unlabeled dataset).
        """
        self.split_name = split_name
        self.dataset_path = Path(dataset_path)

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

        preload = load_annotations or load_ocr
        documents = [
            Document(
                docid=docid,
                dataset_path=dataset_path,
                load_annotations=load_annotations,
                load_ocr=load_ocr,
                cache_images=cache_images,
            )
            for docid in tqdm(
                docids, desc=f"Loading documents for {self.name}", disable=not preload
            )
        ]
        self._set_documents(documents)

        self.cache_images = cache_images

    @property
    def name(self) -> str:
        return f"{self.dataset_path.name}/{self.split_name}"

    @property
    def docids(self) -> List[str]:
        return [doc.docid for doc in self.documents]

    def load_split(self, split_name: str, load_annotations_and_ocr: bool = True) -> "Dataset":
        """
        Load a different split for the current dataset, using the same config for image caching.

        If you want to only load annotations and not OCR, use the main constructor instead.

        Parameters
        ----------
        split_name
            Name of the dataset split to load.
        load_annotations_and_ocr
            Preload annotations and OCR to memory.
        """
        return self.__class__(
            split_name=split_name,
            dataset_path=self.dataset_path,
            load_annotations=load_annotations_and_ocr,
            load_ocr=load_annotations_and_ocr,
            cache_images=self.cache_images,
        )

    def get_cluster(self, cluster_id: int) -> "Dataset":
        return self.from_documents(
            split_name=f"{self.split_name}[cluster_id={cluster_id}]",
            documents=[doc for doc in self.documents if doc.annotation.cluster_id == cluster_id],
            cache_images=self.cache_images,
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
                cache_images=self.cache_images,
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
        return self.from_documents(split_name, sample_documents, self.cache_images)

    def __iter__(self) -> Iterable[Document]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __str__(self) -> str:
        return f"Dataset({self.name})"

    def __repr__(self) -> str:
        return f'Dataset(split_name="{self.split_name}", dataset_path="{self.dataset_path}")'

    @classmethod
    def from_documents(
        cls,
        split_name: str,
        documents: Sequence[Document],
        cache_images: CachingConfig = CachingConfig.DISK,
    ) -> "Dataset":
        """
        Create a dataset directly from documents, rather than from docids.

        This is useful when the documents were already loaded once, e.g., when creating a dataset
        with just a sample of the current documents.
        """
        if len(documents) == 0:
            raise ValueError("Cannot create a dataset with no documents")

        dataset_path = documents[0].dataset_paths.dataset_path
        dataset = cls(
            split_name=split_name,
            dataset_path=dataset_path,
            docids=[doc.docid for doc in documents],
            # Do not load annotations and OCR since it might be already loaded once in `documents`.
            load_annotations=False,
            load_ocr=False,
            cache_images=cache_images,
        )
        dataset._set_documents(documents)
        return dataset

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

    def _load_docids_from_index(self, split_name: str) -> Optional[Sequence[str]]:
        """
        Load docids from the index file on disk.

        Returns
        -------
        Docids loaded from the index file or None if the file does not exist.
        """
        index_path = DataPaths(self.dataset_path).dataset_index_path(split_name)
        if index_path.exists():
            return json.loads(index_path.read_text())
        return None

    def _set_documents(self, documents: Sequence[Document]) -> None:
        """Set dataset documents to the provided documents."""
        self.documents = documents
        self.docid_to_index = {doc.docid: i for i, doc in enumerate(self.documents)}
