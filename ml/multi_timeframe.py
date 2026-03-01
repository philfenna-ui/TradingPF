from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


HORIZON_DAYS = {"1D": 1, "2D": 2, "1W": 5, "1M": 21, "6M": 126}


@dataclass(slots=True)
class TimeframeModelOutput:
    probabilities: dict[str, float]
    confidence: dict[str, float]
    target_ranges: dict[str, tuple[float, float]]
    historical_accuracy: dict[str, float]
    rolling_win_rate: dict[str, float]
    feature_importance: dict[str, float]


class MultiTimeframeModeler:
    def _features(self, daily: pd.DataFrame) -> pd.DataFrame:
        px = daily["close"]
        rets = px.pct_change()
        feat = pd.DataFrame(
            {
                "ret_1d": rets,
                "ret_5d": px.pct_change(5),
                "ret_21d": px.pct_change(21),
                "vol_10d": rets.rolling(10).std(),
                "vol_21d": rets.rolling(21).std(),
                "vwap_dev": (px - daily["vwap"]) / np.maximum(daily["vwap"], 1e-9),
                "atr_norm": daily["atr_proxy"] / np.maximum(px, 1e-9),
            }
        )
        return feat.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _calibrate_prob(
        model: LogisticRegression,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> np.ndarray:
        if len(np.unique(y_train)) < 2:
            return np.full(x_test.shape[0], 0.5, dtype=float)
        try:
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated.fit(x_train, y_train)
            return calibrated.predict_proba(x_test)[:, 1]
        except Exception:
            model.fit(x_train, y_train)
            raw = model.predict_proba(x_test)[:, 1]
            # Isotonic post-calibration fallback
            iso = IsotonicRegression(out_of_bounds="clip")
            idx = np.arange(len(raw))
            iso.fit(idx, raw)
            return iso.predict(idx)

    def fit_predict(self, daily: pd.DataFrame, institutional_score: float) -> TimeframeModelOutput:
        feat = self._features(daily)
        if len(feat) < 180:
            px = float(daily["close"].iloc[-1])
            vol = float(daily["close"].pct_change().dropna().tail(30).std())
            fallback_probs = {}
            fallback_conf = {}
            fallback_tgts = {}
            for k, d in HORIZON_DAYS.items():
                p = float(np.clip(0.45 + (institutional_score - 50) / 300, 0.25, 0.7))
                sigma = max(vol, 0.01) * np.sqrt(d)
                fallback_probs[k] = p
                fallback_conf[k] = float(np.clip(0.35 + 0.1 * (1 - sigma), 0.25, 0.6))
                fallback_tgts[k] = (float(px * (1 - 1.2 * sigma)), float(px * (1 + 1.2 * sigma)))
            return TimeframeModelOutput(
                probabilities=fallback_probs,
                confidence=fallback_conf,
                target_ranges=fallback_tgts,
                historical_accuracy={k: 0.5 for k in HORIZON_DAYS},
                rolling_win_rate={k: 0.5 for k in HORIZON_DAYS},
                feature_importance={c: 0.0 for c in feat.columns},
            )

        px = daily["close"].reindex(feat.index)
        rets = px.pct_change().fillna(0.0)

        probs: dict[str, float] = {}
        confs: dict[str, float] = {}
        targets: dict[str, tuple[float, float]] = {}
        acc: dict[str, float] = {}
        win: dict[str, float] = {}

        model = LogisticRegression(max_iter=600)
        x_all = feat.values
        coeff_accum = np.zeros(x_all.shape[1], dtype=float)
        coeff_count = 0

        last_price = float(px.iloc[-1])
        base_vol = float(rets.tail(30).std())

        for h, days in HORIZON_DAYS.items():
            future_ret = px.shift(-days) / px - 1
            y = (future_ret > 0).astype(int).reindex(feat.index).dropna()
            x = feat.reindex(y.index).values
            if len(y) < 80 or len(np.unique(y)) < 2:
                p = float(np.clip(0.45 + (institutional_score - 50) / 300, 0.25, 0.7))
                probs[h] = p
                acc[h] = 0.5
                win[h] = 0.5
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y.values, test_size=0.25, shuffle=False)
                y_prob = self._calibrate_prob(model, x_train, y_train, x_test)
                y_pred = (y_prob >= 0.5).astype(int)
                acc[h] = float(accuracy_score(y_test, y_pred))
                probs[h] = float(y_prob[-1])
                win[h] = float(np.mean(y_test[-min(30, len(y_test)):])) if len(y_test) else 0.5
                try:
                    model.fit(x_train, y_train)
                    coeff_accum += np.abs(model.coef_[0])
                    coeff_count += 1
                except Exception:
                    pass

            if h in {"1D", "2D", "1W"}:
                # realistic short-horizon cap unless extreme setup
                extreme = institutional_score > 88 and base_vol < 0.012
                cap = 0.9 if extreme else 0.74
                probs[h] = float(min(probs[h], cap))
            probs[h] = float(np.clip(probs[h], 0.08, 0.92))

            sigma = max(base_vol, 0.008) * np.sqrt(days)
            drift = (probs[h] - 0.5) * 0.55
            tgt_low = last_price * (1 + drift - 1.35 * sigma)
            tgt_high = last_price * (1 + drift + 1.35 * sigma)
            targets[h] = (float(tgt_low), float(tgt_high))
            confs[h] = float(np.clip(0.35 + 0.45 * acc[h] - 0.2 * sigma, 0.25, 0.9))

        feat_imp = {c: float(v) for c, v in zip(feat.columns, (coeff_accum / max(coeff_count, 1)))}
        return TimeframeModelOutput(
            probabilities=probs,
            confidence=confs,
            target_ranges=targets,
            historical_accuracy=acc,
            rolling_win_rate=win,
            feature_importance=feat_imp,
        )

