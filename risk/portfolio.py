from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.models import AllocationDecision


@dataclass(slots=True)
class PortfolioConstructionResult:
    weights: dict[str, float]
    expected_volatility: float
    concentration_hhi: float
    notes: list[str]


class RiskFirstPortfolioConstructor:
    def __init__(self, max_single_weight: float = 0.2) -> None:
        self.max_single_weight = max_single_weight

    def construct(self, allocation: AllocationDecision, covariance: np.ndarray, tickers: list[str]) -> PortfolioConstructionResult:
        if len(tickers) == 0:
            return PortfolioConstructionResult(weights={}, expected_volatility=0.0, concentration_hhi=0.0, notes=["No tickers passed filter."])
        raw = np.array([allocation.weights.get(t, 0.0) for t in tickers], dtype=float)
        if raw.sum() <= 0:
            raw = np.ones(len(tickers), dtype=float) / len(tickers)
        raw = raw / raw.sum()
        capped = np.minimum(raw, self.max_single_weight)
        if capped.sum() <= 0:
            capped = np.ones_like(capped) / len(capped)
        weights = capped / capped.sum()
        exp_vol = float(np.sqrt(weights.T @ covariance @ weights) * np.sqrt(252))
        hhi = float(np.sum(weights**2))
        notes = []
        if hhi > 0.22:
            notes.append("Concentration elevated; consider increasing diversification.")
        return PortfolioConstructionResult(
            weights={t: float(w) for t, w in zip(tickers, weights)},
            expected_volatility=exp_vol,
            concentration_hhi=hhi,
            notes=notes,
        )
