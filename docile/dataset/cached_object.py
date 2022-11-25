from pathlib import Path
from typing import Generic, Optional, TypeVar

CT = TypeVar("CT")


class CachedObject(Generic[CT]):
    def __init__(
        self,
        path: Path,
        mem_cache: bool = True,
        disk_cache: bool = True,
    ):
        self._content: Optional[CT] = None

        self.path = path
        self.mem_cache = mem_cache
        self.disk_cache = disk_cache

    def from_disk(self) -> CT:  # noqa
        raise NotImplementedError

    def to_disk(self, content: CT) -> None:  # noqa
        raise NotImplementedError

    def predict(self) -> CT:
        raise NotImplementedError

    def predict_and_overwrite(self) -> CT:
        content = self.predict()

        if self.mem_cache:
            self._content = content
        if self.disk_cache:
            self.to_disk(content)

        return content

    @property
    def content(self) -> CT:
        """Try to load the content from cache."""
        if self.mem_cache and self._content:
            return self._content
        if self.disk_cache and self.path.exists():
            content = self.from_disk()
            if self.mem_cache:
                self._content = content
            return content

        # Nothing had been found, we need to predict
        try:
            return self.predict_and_overwrite()
        except NotImplementedError:
            raise ValueError(
                f"Object {self.path} not found in memory, on disk and cannot be created"
            )
