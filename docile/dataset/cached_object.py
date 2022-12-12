from pathlib import Path
from types import TracebackType
from typing import Generic, Optional, Type, TypeVar

CT = TypeVar("CT")


class CachedObject(Generic[CT]):
    """
    Base class for objects that can be cached to disk and memory.

    To use disk caching, you have to implement `from_disk` and `to_disk` method. You can also
    implement `predict` which will be used if the object does not exist in cache (resp. on disk).

    You can temporarily turn on memory caching by entering the object as a context manager.
    """

    def __init__(
        self,
        path: Path,
        mem_cache: bool = False,
        disk_cache: bool = True,
    ):
        self._content: Optional[CT] = None

        self.path = path
        self.mem_cache_permanent = mem_cache
        self.mem_cache = mem_cache
        self.disk_cache = disk_cache

    def __enter__(self) -> "CachedObject":
        self.mem_cache = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],  # noqa: U100
        exc: Optional[BaseException],  # noqa: U100
        traceback: Optional[TracebackType],  # noqa: U100
    ) -> None:
        if not self.mem_cache_permanent:
            self.mem_cache = False
            self._content = None

    def from_disk(self) -> CT:
        raise NotImplementedError

    def to_disk(self, content: CT) -> None:  # noqa: U100
        raise NotImplementedError

    def predict(self) -> CT:
        raise NotImplementedError

    def overwrite(self, content: CT) -> None:
        if self.mem_cache:
            self._content = content
        if self.disk_cache:
            self.to_disk(content)

    def predict_and_overwrite(self) -> CT:
        content = self.predict()
        self.overwrite(content)
        return content

    @property
    def content(self) -> CT:
        """Try to load the content from cache."""
        if self.mem_cache and self._content is not None:
            return self._content
        if self.disk_cache and self.path.exists():
            content = self.from_disk()
            if self.mem_cache:
                self._content = content
            return content

        # Object not found in cache, we need to predict it.
        try:
            return self.predict_and_overwrite()
        except NotImplementedError:
            raise ValueError(
                f"Object {self.path} not found in memory, on disk and cannot be created"
            )
