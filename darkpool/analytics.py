from __future__ import annotations

from typing import Any

import numpy as np

from core.base import BaseModule
from core.models import ModuleScore


class DarkPoolAnalyticsModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="dark_pool")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            dp = bundle["dark_pool"]
            ratio = float(dp["dark_pool_volume_ratio"])
            blocks = float(dp["block_trade_count"])
            repeated = float(dp["repeated_prints_score"])
            accumulation = float(np.clip(10 * (0.6 * ratio + 0.25 * min(blocks / 10, 1) + 0.15 * repeated), 0, 10))
            distribution = float(np.clip(10 * (0.4 * (1 - ratio) + 0.4 * min(blocks / 10, 1) + 0.2 * (1 - repeated)), 0, 10))
            breakout_adj = float(np.clip((accumulation - distribution) / 10, -1, 1))
            score = float(np.clip((accumulation - distribution + 10) / 2, 0, 10))
            confidence = float(np.clip(0.4 + 0.3 * ratio + 0.05 * min(blocks, 10), 0.35, 0.88))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=confidence,
                metadata={
                    "accumulation_score": accumulation,
                    "distribution_score": distribution,
                    "breakout_probability_adjustment": breakout_adj,
                },
            )

