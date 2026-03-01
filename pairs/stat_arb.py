from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class PairsStatArbModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="pairs")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            d: pd.DataFrame = bundle["daily"]
            px = d["close"]
            hedge_ratio = 1.0
            spread = np.log(px + 1e-6) - hedge_ratio * np.log(px.rolling(3).mean().bfill() + 1e-6)
            z = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-9)
            z_last = float(z.iloc[-1])
            vol_conf = float(d["volume"].iloc[-1] / max(d["volume"].tail(20).mean(), 1))
            contraction = float(px.pct_change().dropna().tail(20).std() < px.pct_change().dropna().tail(120).std())
            entry = abs(z_last) > 2 and contraction > 0 and vol_conf > 0.8
            score = float(np.clip(10 * min(abs(z_last) / 3, 1) * (0.7 + 0.3 * contraction), 0, 10))
            conf = float(np.clip(0.4 + 0.2 * min(vol_conf, 2) + 0.2 * contraction, 0.35, 0.88))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=conf,
                metadata={
                    "pair": f"{ticker}/synthetic_beta",
                    "zscore": z_last,
                    "entry_signal": bool(entry),
                    "exit_threshold_zscore": 0.5,
                    "expected_reversion_days": int(max(2, 12 - min(abs(z_last) * 2, 8))),
                    "sharpe_projection": float(0.6 + 0.8 * min(abs(z_last) / 3, 1)),
                },
            )
