from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RunState:
    last_refresh_ts: str = ""
    last_prices: dict[str, float] = field(default_factory=dict)
    last_scores: dict[str, float] = field(default_factory=dict)
    last_prob_1d: dict[str, float] = field(default_factory=dict)
    model_version: int = 0
    last_retrain_ts: str = ""
    last_feature_importance: dict[str, float] = field(default_factory=dict)
    last_performance: dict[str, float] = field(default_factory=dict)


class RuntimeStateStore:
    def __init__(self, path: str = "logs/run_state.json") -> None:
        self.path = Path(path)

    def load(self) -> RunState:
        if not self.path.exists():
            return RunState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return RunState(**raw)

    def save(self, state: RunState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def detect_static_output(
    prev: RunState,
    prices: dict[str, float],
    scores: dict[str, float],
    prob_1d: dict[str, float],
) -> tuple[bool, str]:
    if not prev.last_prices or not prev.last_scores or not prev.last_prob_1d:
        return False, ""
    same_prices = all(abs(prices.get(k, -999) - v) < 1e-9 for k, v in prev.last_prices.items())
    same_scores = all(abs(scores.get(k, -999) - v) < 1e-9 for k, v in prev.last_scores.items())
    same_probs = all(abs(prob_1d.get(k, -999) - v) < 1e-9 for k, v in prev.last_prob_1d.items())
    if same_prices and same_scores and same_probs:
        return True, "Model output static — investigate data feed or retraining."
    return False, ""
