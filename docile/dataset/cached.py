from pathlib import Path
from typing import Generic, Optional, TypeVar

CT = TypeVar("CT")


class Cached(Generic[CT]):
    def __init__(
        self,
        path: Path,
        overwrite: bool = False,
        mem_cache: bool = True,
        disc_cache: bool = True,
    ):
        self._content: Optional[CT] = None

        self.path = path
        self.overwrite = overwrite
        self.mem_cache = mem_cache
        self.disc_cache = disc_cache

    def from_file(self, path: Path) -> CT:  # noqa
        raise NotImplementedError

    def to_file(self, content: CT) -> None:  # noqa
        raise NotImplementedError

    def predict(self) -> CT:
        raise NotImplementedError

    @property
    def content(self) -> CT:
        """Try to load the content from cache."""
        if (self.mem_cache and not self.overwrite) and self._content:
            return self._content
        if (self.disc_cache and not self.overwrite) and self.path.exists():
            return self.from_file(self.path)

        # Nothing had been found, we need to predict
        content = self.predict()

        if self.mem_cache:
            self._content = content
        if self.disc_cache:
            self.to_file(content)

        return content
