from pathlib import Path, PurePosixPath
from typing import Optional, Union
from zipfile import ZipFile

from docile.dataset.types import OptionalImageSize


class PathMaybeInZip:
    """Path that can point to the file system or to a path inside of a ZIP file."""

    def __init__(
        self,
        root_path: Path,
        relative_path: Union[PurePosixPath, str] = "",
        open_zip_file: Optional[ZipFile] = None,
    ):
        self.root_path = root_path
        self.relative_path = PurePosixPath(relative_path)

        if self.is_in_zip() and open_zip_file is None:
            self._zip_file: Optional[ZipFile] = ZipFile(self.root_path, "r")
            self._zip_file_owner = True
        else:
            self._zip_file = open_zip_file
            self._zip_file_owner = False

    def __del__(self) -> None:
        if self._zip_file_owner:
            self.zip_file.close()

    @property
    def zip_file(self) -> ZipFile:
        assert self._zip_file is not None
        return self._zip_file

    def exists(self) -> bool:
        if self.is_in_zip():
            return str(self.relative_path) in self.zip_file.namelist()
        return self.full_path.exists()

    def read_bytes(self) -> bytes:
        if self.is_in_zip():
            return self.zip_file.read(str(self.relative_path))
        return self.full_path.read_bytes()

    def write_text(self, text: str) -> int:
        if self.is_in_zip():
            raise RuntimeError(
                f"Trying to write to ZIP file {self.root_path} (path {self.relative_path}) which "
                "is not allowed, you need to unzip the dataset first.",
            )
        return self.full_path.write_text(text)

    def with_suffix(self, suffix: str) -> "PathMaybeInZip":
        return PathMaybeInZip(
            self.root_path, self.relative_path.with_suffix(suffix), self._zip_file
        )

    def __truediv__(self, key: str) -> "PathMaybeInZip":
        return PathMaybeInZip(self.root_path, self.relative_path / key, self._zip_file)

    def is_in_zip(self) -> bool:
        return self.root_path.suffix == ".zip"

    @property
    def full_path(self) -> Path:
        assert not self.is_in_zip(), "Path object can only be created if path is not in ZIP"
        return self.root_path / self.relative_path

    def __str__(self) -> str:
        if self.is_in_zip():
            if self.relative_path == PurePosixPath():
                return str(self.root_path)
            return f"{self.root_path!s}[{self.relative_path!s}]"
        return str(self.full_path)

    def __repr__(self) -> str:
        return (
            f"PathMaybeInZip(root_path={self.root_path!r}, relative_path={self.relative_path!r})"
        )


class DataPaths:
    def __init__(self, dataset_path: Union[Path, str, "DataPaths"]):
        if not isinstance(dataset_path, DataPaths):
            self.dataset_path = PathMaybeInZip(Path(dataset_path))
        else:
            self.dataset_path = dataset_path.dataset_path

    @property
    def name(self) -> str:
        return self.dataset_path.root_path.name

    def is_in_zip(self) -> bool:
        return self.dataset_path.is_in_zip()

    def dataset_index_path(self, split_name: str) -> PathMaybeInZip:
        return (self.dataset_path / split_name).with_suffix(".json")

    def pdf_path(self, docid: str) -> PathMaybeInZip:
        return self.dataset_path / "pdfs" / f"{docid}.pdf"

    def ocr_path(self, docid: str) -> PathMaybeInZip:
        return self.dataset_path / "ocr" / f"{docid}.json"

    def annotation_path(self, docid: str) -> PathMaybeInZip:
        return self.dataset_path / "annotations" / f"{docid}.json"

    def cache_images_path(self, docid: str, size: OptionalImageSize) -> PathMaybeInZip:
        """Path to directory with cached images for the individual pages."""
        directory_name = docid
        size_tag = self._size_tag(size)
        if size_tag != "":
            directory_name += f"__{self._size_tag(size)}"
        return self.dataset_path / "cached_images" / directory_name

    @staticmethod
    def cache_page_image_path(cache_images_path: PathMaybeInZip, page_i: int) -> PathMaybeInZip:
        return cache_images_path / f"{page_i}.png"

    @staticmethod
    def _size_tag(size: OptionalImageSize) -> str:
        """Convert size param to string. This string is used as part of the cache path for images."""
        if size == (None, None):
            return ""
        if isinstance(size, int):
            return str(size)
        return f"{size[0]}x{size[1]}"
