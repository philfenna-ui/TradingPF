from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.base import BaseModule
from core.models import ModuleScore


class MLProbabilityModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="ml_probability")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            daily: pd.DataFrame = bundle["daily"]
            rets = daily["close"].pct_change().dropna()
            feat = pd.DataFrame(
                {
                    "ret_1d": rets,
                    "vol_10d": rets.rolling(10).std(),
                    "mom_20d": daily["close"].pct_change(20).reindex(rets.index),
                }
            ).dropna()
            if len(feat) < 60:
                return ModuleScore(module=self.name, value=5.0, confidence=0.3, metadata={"reason": "insufficient_data"})
            model = IsolationForest(random_state=42, contamination=0.06)
            model.fit(feat.values)
            scores = -model.decision_function(feat.values)
            anomaly = float(np.clip(np.percentile(scores, 85), 0, 1.5))
            prob_up = float(np.clip(0.5 + 0.8 * feat["mom_20d"].iloc[-1] - 0.2 * feat["vol_10d"].iloc[-1], 0.05, 0.95))
            score = float(np.clip(prob_up * 10 * (1 - 0.25 * anomaly), 0, 10))
            conf = float(np.clip(0.55 + 0.3 * (1 - anomaly), 0.35, 0.9))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=conf,
                metadata={"prob_up": prob_up, "anomaly_factor": anomaly},
            )

