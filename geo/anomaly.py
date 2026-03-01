from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.base import BaseModule
from core.models import ModuleScore


class RegimeAndAnomalyModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="geo_regime_anomaly")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            daily: pd.DataFrame = bundle["daily"]
            macro = bundle["macro"]
            rets = daily["close"].pct_change().dropna()
            vol20 = float(rets.tail(20).std())
            ma50 = float(daily["close"].rolling(50).mean().iloc[-1])
            ma200 = float(daily["close"].rolling(200).mean().iloc[-1])
            trend = float(daily["close"].iloc[-1] / max(ma200, 1e-6) - 1)

            if trend > 0.05 and vol20 < 0.02:
                regime = "bull"
            elif trend < -0.05 and vol20 > 0.018:
                regime = "bear"
            elif vol20 > 0.025:
                regime = "high_volatility"
            else:
                regime = "sideways"
            if macro["vix"] > 35:
                regime = "crisis"

            features = np.column_stack(
                [
                    rets.tail(120).values,
                    rets.tail(120).rolling(5).std().fillna(0).values,
                ]
            )
            model = IsolationForest(random_state=11, contamination=0.05)
            model.fit(features)
            an_scores = -model.decision_function(features)
            severity = float(np.clip(np.percentile(an_scores, 90) * 8, 0, 10))
            systemic_risk = float(np.clip(0.2 + 0.6 * (severity / 10) + 0.2 * (macro["vix"] / 40), 0, 1))
            derisk = float(np.clip(systemic_risk * 0.5, 0.0, 0.6))
            score = float(np.clip(10 - severity, 0, 10))
            conf = float(np.clip(0.5 + 0.25 * (1 - severity / 10), 0.35, 0.9))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=conf,
                metadata={
                    "regime": regime,
                    "anomaly_severity": severity,
                    "systemic_risk_probability": systemic_risk,
                    "suggested_derisk_pct": derisk,
                    "hedge_overlay": "long_treasury_plus_long_vol" if systemic_risk > 0.65 else "balanced_hedge",
                    "ma50": ma50,
                    "ma200": ma200,
                },
            )

