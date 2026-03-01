from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.models import AllocationDecision, RiskReport, TradeRecommendation


class DashboardReporter:
    def export_snapshot(
        self,
        recommendations: list[TradeRecommendation],
        risk_report: RiskReport,
        allocation: AllocationDecision,
        backtest_metrics: dict[str, Any],
        cross_asset_intelligence: dict[str, Any] | None = None,
        portfolio_construction: dict[str, Any] | None = None,
        output_file: str = "logs/dashboard_snapshot.json",
    ) -> Path:
        payload = {
            "market_regime": risk_report.regime,
            "geopolitical_risk_index": 100 * risk_report.tail_risk_probability,
            "top_tactical_trades": [asdict(r) for r in recommendations],
            "allocation_weights": allocation.weights,
            "confidence_adjusted_exposure": allocation.confidence_adjusted_exposure,
            "risk_metrics": asdict(risk_report),
            "cross_asset_intelligence": cross_asset_intelligence or {},
            "portfolio_construction": portfolio_construction or {},
            "performance_tracking": backtest_metrics,
            "volatility_mispricing_alerts": [
                {"ticker": r.ticker, "signal": r.module_scores["volatility_adjustment"].metadata["risk_scenario"]}
                for r in recommendations
                if "volatility_adjustment" in r.module_scores
            ],
            "dark_pool_leaders": [
                {"ticker": r.ticker, "accumulation_score": r.module_scores["dark_pool"].metadata["accumulation_score"]}
                for r in recommendations
                if "dark_pool" in r.module_scores
            ],
            "pairs_trades": [
                {"ticker": r.ticker, "pair": r.module_scores["pairs"].metadata["pair"]}
                for r in recommendations
                if "pairs" in r.module_scores
            ],
        }
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out
