from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from core.logging_utils import append_jsonl


@dataclass(slots=True)
class RetrainDecision:
    should_retrain: bool
    reason: str
    timestamp: str


class RetrainingManager:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def evaluate(self, samples_since_last_train: int, days_since_last_train: int) -> RetrainDecision:
        enabled = bool(self.cfg.get("enabled", True))
        cadence = int(self.cfg.get("cadence_days", 7))
        min_samples = int(self.cfg.get("min_new_samples", 500))
        should = enabled and (days_since_last_train >= cadence or samples_since_last_train >= min_samples)
        reason = "cadence_or_sample_threshold_met" if should else "threshold_not_met"
        decision = RetrainDecision(
            should_retrain=should,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        append_jsonl("logs/retraining_events.jsonl", asdict(decision))
        return decision
