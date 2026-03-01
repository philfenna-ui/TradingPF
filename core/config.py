from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.exceptions import ConfigurationError


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw.get("runtime", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw.get("scoring", {})

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw.get("execution", {})

    @property
    def retraining(self) -> dict[str, Any]:
        return self.raw.get("retraining", {})


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Config not found: {cfg_path}")
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("Config root must be an object.")
    return AppConfig(raw=loaded)

