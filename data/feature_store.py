from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class FeatureStore:
    """
    Lightweight file-backed feature store.
    Uses CSV/JSON artifacts for portability without extra runtime dependencies.
    """

    def __init__(self, root: str = "data_store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_frame(self, namespace: str, name: str, frame: pd.DataFrame) -> Path:
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        path = ns / f"{name}.csv"
        frame.to_csv(path, index=True)
        return path

    def load_frame(self, namespace: str, name: str) -> pd.DataFrame:
        path = self.root / namespace / f"{name}.csv"
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def save_json(self, namespace: str, name: str, payload: dict[str, Any]) -> Path:
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        path = ns / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def load_json(self, namespace: str, name: str) -> dict[str, Any]:
        path = self.root / namespace / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def save_pickle(self, namespace: str, name: str, payload: Any) -> Path:
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        path = ns / f"{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        return path

    def load_pickle(self, namespace: str, name: str) -> Any:
        path = self.root / namespace / f"{name}.pkl"
        with path.open("rb") as f:
            return pickle.load(f)

    def exists(self, namespace: str, name: str, ext: str = "pkl") -> bool:
        path = self.root / namespace / f"{name}.{ext}"
        return path.exists()
