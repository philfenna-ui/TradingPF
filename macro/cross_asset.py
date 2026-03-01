from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class CrossAssetMacroModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="macro_alignment")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            daily: pd.DataFrame = bundle["daily"]
            macro = bundle["macro"]

            returns = daily["close"].pct_change().dropna().tail(30)
            corr = float(returns.autocorr(lag=1))
            trend_30 = float(daily["close"].pct_change(30).iloc[-1])
            curve_spread = float(macro["yield_10y"] - macro["yield_2y"])
            liquidity = float(macro["liquidity_index"])
            vix = float(macro["vix"])
            dxy = float(macro["dxy"])

            risk_on = 0.3 * (trend_30 > 0) + 0.25 * (vix < 22) + 0.25 * (liquidity > 0.52) + 0.2 * (curve_spread > -0.4)
            capital_flow = 0.5 * trend_30 + 0.3 * (liquidity - 0.5) - 0.2 * (dxy - 100) / 12
            score = float(np.clip(10 * risk_on + 2.5 * capital_flow + 0.5 * max(corr, 0), 0, 10))
            confidence = float(np.clip(0.45 + 0.2 * abs(capital_flow), 0.35, 0.9))

            return ModuleScore(
                module=self.name,
                value=score,
                confidence=confidence,
                metadata={
                    "rolling_30d_corr_proxy": corr,
                    "risk_on_off_score": float(risk_on),
                    "capital_flow_direction_index": float(capital_flow),
                    "liquidity_expansion_index": liquidity,
                    "tactical_strategic_split": [0.35, 0.65] if score >= 6 else [0.2, 0.8],
                    "hedge_recommendation": "long_vol" if vix < 15 else "short_duration_bonds",
                },
            )

