from __future__ import annotations

from typing import Any

import numpy as np

from core.models import RiskReport


class RiskCommandCenter:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @staticmethod
    def _var(returns: np.ndarray, level: float) -> float:
        return float(-np.quantile(returns, 1 - level))

    @staticmethod
    def _es(returns: np.ndarray, level: float) -> float:
        threshold = np.quantile(returns, 1 - level)
        tail = returns[returns <= threshold]
        return float(-tail.mean()) if len(tail) else 0.0

    def evaluate(self, portfolio_returns: np.ndarray, exposures: dict[str, dict[str, float]]) -> RiskReport:
        var95 = self._var(portfolio_returns, 0.95)
        var99 = self._var(portfolio_returns, 0.99)
        es95 = self._es(portfolio_returns, 0.95)
        es99 = self._es(portfolio_returns, 0.99)
        max_dd = float(abs(np.min(np.cumsum(portfolio_returns))))
        tail_prob = float(np.clip((var99 + es99) * 8, 0, 1))

        stress = {
            "equity_shock_-5pct": float(0.05 * sum(exposures["asset_class"].values())),
            "rate_shock_+100bps": float(0.03 * exposures["asset_class"].get("rates", 0.0)),
            "vol_spike_+40pct": float(0.04 * exposures["asset_class"].get("equities", 0.0)),
        }

        notes: list[str] = []
        allowed = True
        if max_dd > self.cfg.get("max_daily_drawdown_pct", 0.025):
            allowed = False
            notes.append("Max daily drawdown threshold breached.")
        if tail_prob > self.cfg.get("tail_risk_trigger", 0.75):
            notes.append("Tail risk elevated: de-risk trigger active.")

        regime = "crisis" if tail_prob > 0.8 else "high_volatility" if var95 > 0.025 else "bull"
        return RiskReport(
            portfolio_var_95=var95,
            portfolio_var_99=var99,
            expected_shortfall_95=es95,
            expected_shortfall_99=es99,
            tail_risk_probability=tail_prob,
            sector_exposure=exposures["sector"],
            asset_class_exposure=exposures["asset_class"],
            stress_test_loss_pct=stress,
            regime=regime,
            allowed=allowed,
            notes=notes,
        )

