from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ModuleScore:
    module: str
    value: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HorizonSignal:
    horizon: str
    probability: float
    target_low: float
    target_high: float
    upside_pct: float
    downside_pct: float
    risk_classification: str


@dataclass(slots=True)
class TradeRecommendation:
    ticker: str
    name: str
    category: str
    current_price: float
    institutional_score: float
    buy_category: str
    buy_color: str
    mock_data_used: bool
    mock_fields: list[str]
    stale_data_used: bool
    stale_age_hours: float
    action: str
    module_scores: dict[str, ModuleScore]
    horizons: list[HorizonSignal]
    narrative: str
    plain_reason: str
    plain_reason_points: list[str]
    risk_metrics: dict[str, float]
    explainability: dict[str, dict[str, float | str]] = field(default_factory=dict)
    sparkline: list[float] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class RiskReport:
    portfolio_var_95: float
    portfolio_var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    tail_risk_probability: float
    sector_exposure: dict[str, float]
    asset_class_exposure: dict[str, float]
    stress_test_loss_pct: dict[str, float]
    regime: str
    allowed: bool
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AllocationDecision:
    weights: dict[str, float]
    confidence_adjusted_exposure: float
    efficiency_improvement_pct: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
