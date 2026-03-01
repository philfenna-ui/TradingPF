from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Any

from core.models import ModuleScore


class BaseModule(ABC):
    """Thread-safe module contract used by all engines."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = RLock()

    @abstractmethod
    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        raise NotImplementedError

