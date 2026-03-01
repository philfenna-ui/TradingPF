from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class LiquidityMicrostructureModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="liquidity_pattern")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            intra: pd.DataFrame = bundle["intraday"]
            r = intra["close"].pct_change().dropna()
            range_comp = float(r.tail(80).std() < r.tail(300).std() * 0.85)
            vol_spike = float(intra["volume"].iloc[-1] / max(intra["volume"].tail(100).mean(), 1))
            breakout = float(intra["close"].iloc[-1] > intra["close"].tail(120).max() * 0.998)
            false_breakout = float((breakout > 0) and (r.tail(10).sum() < 0))
            vwap = (intra["close"] * intra["volume"]).sum() / max(intra["volume"].sum(), 1)
            vwap_state = float((intra["close"].iloc[-1] - vwap) / vwap)
            sweep = float(np.percentile(np.abs(r.tail(100)), 90) > np.percentile(np.abs(r), 70))
            cont_prob = float(np.clip(0.35 + 0.2 * breakout + 0.15 * range_comp + 0.1 * sweep - 0.2 * false_breakout, 0.05, 0.95))
            score = float(np.clip(10 * cont_prob, 0, 10))
            conf = float(np.clip(0.45 + 0.1 * min(vol_spike, 3) + 0.1 * abs(vwap_state), 0.35, 0.9))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=conf,
                metadata={
                    "pattern_classification": "breakout_continuation" if breakout and not false_breakout else "range_or_reversal",
                    "confidence_pct": 100 * conf,
                    "risk_reward_projection": float(1.2 + 1.8 * cont_prob),
                    "volatility_expansion_probability": float(np.clip(cont_prob + 0.1 * sweep, 0, 1)),
                    "vwap_reclaim_loss": vwap_state,
                },
            )

