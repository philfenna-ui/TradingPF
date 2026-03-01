from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class VolatilityArbitrageModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="volatility_adjustment")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            daily: pd.DataFrame = bundle["daily"]
            chain: pd.DataFrame = bundle["options_chain"]
            rv = float(daily["close"].pct_change().dropna().tail(20).std() * np.sqrt(252))
            iv = float(chain["iv"].mean())
            ivrv = iv - rv
            iv_rank = float((chain["iv"].mean() - chain["iv"].quantile(0.1)) / (chain["iv"].quantile(0.9) - chain["iv"].quantile(0.1) + 1e-6))
            term_shift = float(chain["iv"].std())
            long_vol = float(np.clip(0.5 + 2.0 * ivrv + 0.8 * term_shift, 0, 1))
            short_vol = float(np.clip(0.5 - 2.0 * ivrv + 0.5 * (1 - min(term_shift, 1)), 0, 1))
            score = float(np.clip((long_vol - short_vol + 1) * 5, 0, 10))
            conf = float(np.clip(0.45 + 0.3 * abs(ivrv), 0.35, 0.9))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=conf,
                metadata={
                    "implied_vol": iv,
                    "realized_vol": rv,
                    "iv_rank": float(np.clip(iv_rank, 0, 1)),
                    "long_vol_opportunity": long_vol,
                    "short_vol_opportunity": short_vol,
                    "expected_payoff_projection": float(1.2 + 2.0 * abs(ivrv)),
                    "risk_scenario": "vol_crush" if iv > rv * 1.3 else "vol_expansion",
                },
            )

