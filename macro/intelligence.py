from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.schemas import MarketBundle


@dataclass(slots=True)
class CrossAssetIntel:
    correlation_30d: dict[str, dict[str, float]]
    risk_on_off_score: float
    capital_flow_direction_index: float
    liquidity_expansion_index: float
    sector_rotation_momentum: dict[str, float]
    lead_lag_proxy: dict[str, float]
    tactical_vs_strategic_split: list[float]
    hedge_recommendations: list[str]


class CrossAssetIntelligenceEngine:
    def compute(self, bundles: dict[str, MarketBundle]) -> CrossAssetIntel:
        returns = {
            t: b.daily["close"].pct_change().dropna().tail(30)
            for t, b in bundles.items()
        }
        frame = pd.DataFrame(returns).dropna(axis=0, how="any")
        corr = frame.corr().fillna(0.0)
        mean_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean()) if corr.shape[0] > 1 else 0.0

        macro = next(iter(bundles.values())).macro
        risk_on = (
            0.3 * float(macro["vix"] < 20)
            + 0.25 * float(macro["liquidity_index"] > 0.52)
            + 0.25 * float(macro["yield_10y"] - macro["yield_2y"] > -0.35)
            + 0.2 * float(frame.mean().mean() > 0)
        )
        capital_flow = float(frame.mean().mean() * 100 + 0.4 * (macro["liquidity_index"] - 0.5) - 0.1 * (macro["dxy"] - 100) / 10)
        liquidity = float(macro["liquidity_index"])

        sector_mom: dict[str, list[float]] = {}
        for bundle in bundles.values():
            sec = bundle.sector
            mom = float(bundle.daily["close"].pct_change(20).iloc[-1])
            sector_mom.setdefault(sec, []).append(mom)
        sector_rotation = {k: float(np.mean(v)) for k, v in sector_mom.items()}

        lead_lag = {}
        for col in frame.columns:
            x = frame[col]
            y = frame.mean(axis=1)
            lead_lag[col] = float(x.shift(1).corr(y))

        hedges = ["long_volatility_overlay"] if macro["vix"] < 17 else ["short_duration_treasuries"]
        if mean_corr > 0.65:
            hedges.append("pair_trade_dispersion_overlay")

        split = [0.4, 0.6] if risk_on > 0.55 else [0.25, 0.75]
        return CrossAssetIntel(
            correlation_30d=corr.to_dict(),
            risk_on_off_score=float(np.clip(risk_on, 0, 1)),
            capital_flow_direction_index=capital_flow,
            liquidity_expansion_index=liquidity,
            sector_rotation_momentum=sector_rotation,
            lead_lag_proxy=lead_lag,
            tactical_vs_strategic_split=split,
            hedge_recommendations=hedges,
        )

