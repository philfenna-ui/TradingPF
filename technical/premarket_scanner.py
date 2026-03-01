from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PreMarketTacticalScanner:
    def scan(self, ticker: str, bundle: dict[str, Any]) -> dict[str, Any]:
        daily: pd.DataFrame = bundle["daily"]
        intra: pd.DataFrame = bundle["intraday"]
        news = bundle.get("news", [])

        prev_close = float(daily["close"].iloc[-2])
        latest = float(intra["close"].iloc[-1])
        gap = (latest / prev_close - 1) * 100
        rel_volume = float(intra["volume"].tail(10).mean() / max(intra["volume"].tail(150).mean(), 1))
        vwap = float((intra["close"] * intra["volume"]).sum() / max(intra["volume"].sum(), 1))
        vwap_dev = float((latest - vwap) / vwap)
        atr = float(daily["atr_proxy"].tail(14).mean())
        atr_exp = float(atr / max(float(daily["atr_proxy"].tail(120).mean()), 1e-6))
        catalyst = any("guidance" in n["headline"].lower() or "earnings" in n["headline"].lower() for n in news)

        directional = "Bullish" if gap > 0 and vwap_dev > 0 else "Bearish"
        probability = float(np.clip(0.35 + 0.1 * abs(gap) + 0.15 * min(rel_volume, 3) + 0.1 * catalyst, 0.1, 0.95))
        risk = "LOW" if abs(gap) < 2 and rel_volume < 2 else "MEDIUM" if abs(gap) < 4 else "HIGH"
        target_mult = 1.8 if directional == "Bullish" else -1.8
        return {
            "ticker": ticker,
            "directional_bias": directional,
            "entry_zone": [latest * 0.998, latest * 1.002],
            "invalidation_level": latest - 1.2 * atr if directional == "Bullish" else latest + 1.2 * atr,
            "target_projection": latest + target_mult * atr,
            "probability_score": probability,
            "risk_classification": risk,
            "scan_flags": {
                "gap_pct": gap,
                "relative_volume": rel_volume,
                "earnings_catalyst": catalyst,
                "vwap_deviation": vwap_dev,
                "atr_expansion": atr_exp,
            },
        }

