from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base import BaseModule
from core.models import ModuleScore


class TechnicalAnalysisModule(BaseModule):
    def __init__(self) -> None:
        super().__init__(name="technical")

    def evaluate(self, ticker: str, bundle: dict[str, Any]) -> ModuleScore:
        with self._lock:
            df: pd.DataFrame = bundle["daily"].copy()
            df["ema9"] = df["close"].ewm(span=9).mean()
            df["ema21"] = df["close"].ewm(span=21).mean()
            df["ema50"] = df["close"].ewm(span=50).mean()
            df["ema200"] = df["close"].ewm(span=200).mean()

            delta = df["close"].diff()
            gains = delta.clip(lower=0).rolling(14).mean()
            losses = (-delta.clip(upper=0)).rolling(14).mean() + 1e-9
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))

            fast = df["close"].ewm(span=12).mean()
            slow = df["close"].ewm(span=26).mean()
            macd = fast - slow
            signal = macd.ewm(span=9).mean()

            bb_mid = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std().fillna(0)
            bb_up = bb_mid + 2 * bb_std
            bb_dn = bb_mid - 2 * bb_std

            atr = df["atr_proxy"].rolling(14).mean().iloc[-1]
            vwap_dev = (df["close"].iloc[-1] - df["vwap"].iloc[-1]) / df["vwap"].iloc[-1]
            vol_spike = df["volume"].iloc[-1] / max(df["volume"].tail(20).mean(), 1.0)
            breakout = float(df["close"].iloc[-1] > df["close"].tail(20).max() * 0.995)

            alignment = 0.0
            alignment += 2.0 if df["ema9"].iloc[-1] > df["ema21"].iloc[-1] else 0.0
            alignment += 2.0 if df["ema21"].iloc[-1] > df["ema50"].iloc[-1] else 0.0
            alignment += 2.0 if df["ema50"].iloc[-1] > df["ema200"].iloc[-1] else 0.0
            alignment += 1.5 if 45 <= rsi.iloc[-1] <= 70 else 0.0
            alignment += 1.0 if macd.iloc[-1] > signal.iloc[-1] else 0.0
            alignment += 1.5 if breakout > 0 else 0.0
            score = float(np.clip(alignment, 0.0, 10.0))
            confidence = float(np.clip(0.45 + 0.08 * min(vol_spike, 3.0), 0.35, 0.92))
            return ModuleScore(
                module=self.name,
                value=score,
                confidence=confidence,
                metadata={
                    "ema_stack_bullish": bool(df["ema9"].iloc[-1] > df["ema21"].iloc[-1] > df["ema50"].iloc[-1]),
                    "rsi14": float(rsi.iloc[-1]),
                    "macd": float(macd.iloc[-1]),
                    "bb_upper": float(bb_up.iloc[-1]),
                    "bb_lower": float(bb_dn.iloc[-1]),
                    "atr": float(atr),
                    "volume_spike": float(vol_spike),
                    "vwap_deviation": float(vwap_dev),
                    "breakout": breakout,
                },
            )

