from __future__ import annotations

from dataclasses import dataclass

from core.models import RiskReport, TradeRecommendation


@dataclass(slots=True)
class ExecutionDecision:
    allowed: bool
    side: str
    reason: str
    size_multiplier: float


class ExecutionPolicyEngine:
    def decide(self, rec: TradeRecommendation, risk: RiskReport) -> ExecutionDecision:
        if not risk.allowed:
            return ExecutionDecision(False, "HOLD", "risk_gate_blocked", 0.0)
        horizon = rec.horizons[0]
        if rec.institutional_score < 55 or horizon.probability < 0.52:
            return ExecutionDecision(False, "HOLD", "score_or_probability_below_threshold", 0.0)
        side = "BUY" if rec.horizons[0].upside_pct >= abs(rec.horizons[0].downside_pct) else "SELL"
        size = 1.0
        if risk.regime in {"high_volatility", "crisis"}:
            size = 0.5
        return ExecutionDecision(True, side, "policy_passed", size)

