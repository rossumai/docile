from enum import Enum, auto
from types import TracebackType
from typing import Generic, Optional, Type, TypeVar

from docile.dataset.paths import PathMaybeInZip

CT = TypeVar("CT")


class CachingConfig(Enum):
    OFF = auto()
    DISK = auto()
    MEMORY = auto()
    DISK_AND_MEMORY = auto()

    @property
    def disk_cache(self) -> bool:
        return self in [self.DISK, self.DISK_AND_MEMORY]

    @property
    def memory_cache(self) -> bool:
        return self in [self.MEMORY, self.DISK_AND_MEMORY]


class CachedObject(Generic[CT]):
    """
    Base class for objects that can be cached to disk and memory.

    To use disk caching, you have to implement `from_disk` and `to_disk` method. You can also
    implement `predict` which will be used if the object does not exist in cache (resp. on disk).

    You can temporarily turn on memory caching by entering the object as a context manager.
    """

    def __init__(self, path: PathMaybeInZip, cache: CachingConfig):
        # initialize in-memory cache
        self._content: Optional[CT] = None

        self.path = path
        self.memory_cache_permanent = cache.memory_cache
        self.memory_cache = cache.memory_cache
        self.disk_cache = cache.disk_cache

    def load(self) -> None:
        self.memory_cache_permanent = True
        self.memory_cache = True
        if self._content is None:
            self.content

    def release(self) -> None:
        self.memory_cache_permanent = False
        self.memory_cache = False
        self._content = None

    def __enter__(self) -> "CachedObject":
        self.memory_cache = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if not self.memory_cache_permanent:
            self.memory_cache = False
            self._content = None

    def from_disk(self) -> CT:
        raise NotImplementedError

    def to_disk(self, content: CT) -> None:  # noqa: U100
        raise NotImplementedError

    def predict(self) -> CT:
        raise NotImplementedError

    def overwrite(self, content: CT) -> None:
        if self.memory_cache:
            self._content = content
        if self.disk_cache:
            if self.path.is_in_zip():
                raise RuntimeError(f"Cannot write to disk cache since path {self.path} is in ZIP")
            self.to_disk(content)

    def predict_and_overwrite(self) -> CT:
        content = self.predict()
        self.overwrite(content)
        return content

    @property
    def content(self) -> CT:
        """Try to load the content from cache."""
        if self.memory_cache and self._content is not None:
            return self._content
        if self.disk_cache and self.path.exists():
            content = self.from_disk()
            if self.memory_cache:
                self._content = content
            return content

        # Object not found in cache, we need to predict it.
        try:
            return self.predict_and_overwrite()
        except NotImplementedError:
            raise ValueError(
                f"Object {self.path} not found in memory, on disk and cannot be created"
            )
