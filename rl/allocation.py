from __future__ import annotations

from typing import Any

import numpy as np

from core.models import AllocationDecision


class RLAllocationEngine:
    """
    Research-mode policy approximation for allocation.
    Uses regime/features to map into weights with walk-forward placeholders.
    """

    def decide(
        self,
        state: dict[str, Any],
        tickers: list[str],
    ) -> AllocationDecision:
        regime = state.get("regime", "sideways")
        base = {
            "bull": 1.0,
            "bear": 0.65,
            "sideways": 0.8,
            "high_volatility": 0.6,
            "crisis": 0.4,
        }.get(regime, 0.75)

        darkpool = float(state.get("dark_pool_score", 5.0)) / 10
        options = float(state.get("options_flow_score", 5.0)) / 10
        catalyst = float(state.get("catalyst_score", 5.0)) / 10
        macro = float(state.get("cross_asset_flow_index", 0.0))

        conviction = np.clip(0.25 + 0.25 * darkpool + 0.2 * options + 0.2 * catalyst + 0.1 * max(macro, -1), 0.2, 1.0)
        gross = float(np.clip(base * conviction, 0.15, 1.0))

        raw = np.linspace(1.0, 0.4, num=len(tickers))
        raw = raw / raw.sum()
        weights = {t: float(w * gross) for t, w in zip(tickers, raw)}
        efficiency = float(np.clip(3 + 12 * conviction, 2, 18))
        confidence = float(np.clip(0.45 + 0.4 * conviction, 0.35, 0.9))
        return AllocationDecision(
            weights=weights,
            confidence_adjusted_exposure=gross,
            efficiency_improvement_pct=efficiency,
            confidence=confidence,
            metadata={"regime": regime, "walk_forward_required": True},
        )

