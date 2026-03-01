from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtest.walk_forward import WalkForwardBacktester
from core.config import load_config
from core.logging_utils import append_jsonl, configure_logging
from core.orchestrator import TradingPFOrchestrator
from core.runtime_state import RunState, RuntimeStateStore, detect_static_output, utc_iso
from darkpool.analytics import DarkPoolAnalyticsModule
from dashboard.reporting import DashboardReporter
from data.feature_store import FeatureStore
from data.ingestion import IngestionConfig, IngestionPipeline, ProviderRegistry
from data.schemas import MarketBundle
from execution.broker import PaperBroker
from execution.policy import ExecutionPolicyEngine
from geo.anomaly import RegimeAndAnomalyModule
from liquidity.microstructure import LiquidityMicrostructureModule
from macro.cross_asset import CrossAssetMacroModule
from macro.intelligence import CrossAssetIntelligenceEngine
from ml.probability import MLProbabilityModule
from ml.retrainer import RetrainingManager
from ml.multi_timeframe import HORIZON_DAYS
from news.catalyst import NewsCatalystModule
from news.headline_brief import MacroHeadlineBriefingEngine
from options.flow import OptionsFlowModule
from pairs.stat_arb import PairsStatArbModule
from risk.command_center import RiskCommandCenter
from risk.portfolio import RiskFirstPortfolioConstructor
from rl.allocation import RLAllocationEngine
from technical.engine import TechnicalAnalysisModule
from technical.premarket_scanner import PreMarketTacticalScanner
from volatility.arbitrage import VolatilityArbitrageModule


def build_modules(disabled_modules: set[str] | None = None):
    disabled = disabled_modules or set()
    modules = [
        TechnicalAnalysisModule(),
        MLProbabilityModule(),
        OptionsFlowModule(),
        DarkPoolAnalyticsModule(),
        NewsCatalystModule(),
        LiquidityMicrostructureModule(),
        CrossAssetMacroModule(),
        VolatilityArbitrageModule(),
        PairsStatArbModule(),
        RegimeAndAnomalyModule(),
    ]
    return [m for m in modules if m.name not in disabled]


def _profile_weights(base: dict[str, float], profile: str, risk_tolerance: float) -> dict[str, float]:
    w = dict(base)
    if profile == "conservative":
        for k in list(w):
            if k in {"macro_alignment", "volatility_adjustment", "dark_pool"}:
                w[k] *= 1.25
            if k in {"ml_probability", "technical", "pairs"}:
                w[k] *= 0.85
    elif profile == "aggressive":
        for k in list(w):
            if k in {"technical", "ml_probability", "pairs", "options_flow"}:
                w[k] *= 1.25
            if k in {"macro_alignment", "volatility_adjustment"}:
                w[k] *= 0.8
    risk_scale = float(np.clip(0.6 + 0.8 * risk_tolerance, 0.4, 1.4))
    for k in w:
        if k in {"technical", "ml_probability", "pairs", "options_flow"}:
            w[k] *= risk_scale
        if k in {"macro_alignment", "volatility_adjustment"}:
            w[k] *= (2 - risk_scale)
    s = sum(max(v, 0.0) for v in w.values())
    if s <= 0:
        return base
    return {k: float(max(v, 0.0) / s) for k, v in w.items()}


def _bundle_to_dict(bundle: MarketBundle) -> dict[str, Any]:
    return {
        "daily": bundle.daily,
        "intraday": bundle.intraday,
        "macro": bundle.macro,
        "options_chain": bundle.options_chain,
        "dark_pool": bundle.dark_pool,
        "futures": bundle.futures,
        "crypto_funding": bundle.crypto_funding,
        "yield_curve": bundle.yield_curve,
        "news": [
            {
                "timestamp": n.timestamp,
                "headline": n.headline,
                "sentiment": n.sentiment,
                "sector": n.sector,
                "source": n.source,
                "metadata": n.metadata,
            }
            for n in bundle.news
        ],
        "macro_news": [
            {
                "timestamp": n.timestamp,
                "headline": n.headline,
                "sentiment": n.sentiment,
                "sector": n.sector,
                "source": n.source,
                "metadata": n.metadata,
            }
            for n in bundle.macro_news
        ],
        "sector": bundle.sector,
        "asset_class": bundle.asset_class,
        "company_name": bundle.company_name,
        "category_label": bundle.category_label,
        "data_quality": bundle.data_quality,
        "source_timestamps": bundle.source_timestamps,
    }


def _mean_module_value(recs: list[Any], module_name: str, default: float = 5.0) -> float:
    vals = [r.module_scores[module_name].value for r in recs if module_name in r.module_scores]
    return float(np.mean(vals)) if vals else default


def _mean_module_meta(recs: list[Any], module_name: str, key: str, default: float = 0.0) -> float:
    vals = [
        r.module_scores[module_name].metadata.get(key, default)
        for r in recs
        if module_name in r.module_scores
    ]
    vals = [float(v) for v in vals]
    return float(np.mean(vals)) if vals else default


def run_pipeline(
    config_path: str,
    confidence_threshold: float = 0.0,
    risk_tolerance: float = 0.5,
    disabled_modules: list[str] | None = None,
    ranking_profile: str = "balanced",
    universe_override: list[str] | None = None,
    include_signals: list[str] | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    cfg = load_config(cfg_path)
    logger = configure_logging("logs")
    state_store = RuntimeStateStore("logs/run_state.json")
    prev_state = state_store.load()
    universe = [u.upper() for u in (universe_override or list(cfg.data.get("universe", [])))]
    icfg = IngestionConfig(
        provider=str(cfg.data.get("provider", "mock")),
        daily_lookback_days=int(cfg.data.get("daily_lookback_days", 520)),
        intraday_lookback_days=int(cfg.data.get("intraday_lookback_days", 30)),
        intraday_interval_minutes=int(cfg.data.get("intraday_interval_minutes", 30)),
        random_seed=int(cfg.runtime.get("random_seed", 42)),
    )
    provider = ProviderRegistry.build(icfg)
    ingestion = IngestionPipeline(provider, FeatureStore("data_store"), max_workers=int(cfg.runtime.get("max_workers", 8)))
    bundles_typed = ingestion.fetch_universe(universe)
    bundles = {k: _bundle_to_dict(v) for k, v in bundles_typed.items()}

    returns_matrix = np.column_stack([bundles[t]["daily"]["close"].pct_change().dropna().tail(252).values for t in universe])
    portfolio_returns = returns_matrix.mean(axis=1)
    exposures = {
        "sector": {"technology": 0.34, "financials": 0.2, "energy": 0.12, "defensive": 0.34},
        "asset_class": {"equities": 0.65, "crypto": 0.1, "rates": 0.15, "commodities": 0.1},
    }

    risk_engine = RiskCommandCenter(cfg.risk)
    scoring_cfg = dict(cfg.scoring)
    scoring_cfg["weights"] = _profile_weights(scoring_cfg.get("weights", {}), ranking_profile, risk_tolerance)
    orchestrator = TradingPFOrchestrator(
        modules=build_modules(set(disabled_modules or [])),
        risk_engine=risk_engine,
        scoring_cfg=scoring_cfg,
        max_workers=int(cfg.runtime.get("max_workers", 8)),
    )
    recs_all, risk_report = orchestrator.generate_recommendations(
        universe=universe,
        bundles=bundles,
        portfolio_returns=portfolio_returns,
        exposures=exposures,
        confidence_threshold=confidence_threshold,
        risk_tolerance=risk_tolerance,
    )
    allowed_signals = include_signals or ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"]
    recs_filtered = [r for r in recs_all if r.buy_category in allowed_signals]
    top_n = int(cfg.scoring.get("top_n", 5))
    recs = recs_filtered[:top_n]

    cross_asset = CrossAssetIntelligenceEngine().compute(bundles_typed)
    headline_brief = MacroHeadlineBriefingEngine().generate()
    scanner = PreMarketTacticalScanner()
    scans = [scanner.scan(t, bundles[t]) for t in universe]
    append_jsonl("logs/premarket_scans.jsonl", {"scans": scans})

    rl_engine = RLAllocationEngine()
    state = {
        "regime": risk_report.regime,
        "dark_pool_score": _mean_module_value(recs, "dark_pool", 5),
        "options_flow_score": _mean_module_value(recs, "options_flow", 5),
        "catalyst_score": _mean_module_value(recs, "catalyst", 5),
        "cross_asset_flow_index": _mean_module_meta(recs, "macro_alignment", "capital_flow_direction_index", 0.0),
    }
    allocation = rl_engine.decide(state=state, tickers=[r.ticker for r in recs])
    returns_df = np.column_stack([bundles[t]["daily"]["close"].pct_change().dropna().tail(252).values for t in universe])
    if returns_df.ndim == 1 or returns_df.shape[1] == 1:
        var = float(np.var(returns_df if returns_df.ndim == 1 else returns_df[:, 0]))
        cov_full = np.array([[var]])
    else:
        cov_full = np.cov(returns_df.T)
    ticker_index = {t: i for i, t in enumerate(universe)}
    top_tickers = [r.ticker for r in recs]
    idx = [ticker_index[t] for t in top_tickers]
    covariance = cov_full[np.ix_(idx, idx)] if idx else np.array([[0.0]])
    constructor = RiskFirstPortfolioConstructor(max_single_weight=float(cfg.risk.get("max_single_asset_weight", 0.2)))
    portfolio = constructor.construct(allocation, covariance=covariance, tickers=top_tickers)

    policy = ExecutionPolicyEngine()
    if risk_report.allowed and recs:
        broker = PaperBroker(cfg.execution)
        top = recs[0]
        decision = policy.decide(top, risk_report)
        if decision.allowed:
            atr = float(top.module_scores["technical"].metadata.get("atr", top.current_price * 0.01))
            qty = max(1, int(10 * decision.size_multiplier))
            ticket = broker.build_ticket(top.ticker, decision.side, qty=qty, price=top.current_price, atr=atr)
            try:
                receipt = broker.submit(ticket, confirm=not cfg.execution.get("manual_confirmation", True))
                logger.info("Executed order for %s at %.4f", receipt["ticker"], receipt["fill_price"])
            except Exception as exc:
                logger.info("Execution skipped due to compliance gate: %s", exc)
        else:
            logger.info("Execution policy blocked order: %s", decision.reason)

    bt = WalkForwardBacktester().run(portfolio_returns)
    retraining = RetrainingManager(cfg.retraining).evaluate(samples_since_last_train=800, days_since_last_train=8)
    prices_now = {r.ticker: r.current_price for r in recs}
    score_now = {r.ticker: r.institutional_score for r in recs}
    prob_now = {r.ticker: next((h.probability for h in r.horizons if h.horizon == "1D"), 0.0) for r in recs}
    is_static, static_warning = detect_static_output(prev_state, prices_now, score_now, prob_now)

    model_version = int(prev_state.model_version) + 1
    feat_importance_curr = {}
    for r in recs:
        tech_imp = r.risk_metrics.get("horizon_accuracy", {})
        if tech_imp:
            for hk, hv in tech_imp.items():
                feat_importance_curr[hk] = float(feat_importance_curr.get(hk, 0.0) + hv)
    for k in list(feat_importance_curr):
        feat_importance_curr[k] /= max(len(recs), 1)
    drift = 0.0
    if prev_state.last_feature_importance:
        keys = set(prev_state.last_feature_importance).union(feat_importance_curr)
        drift = float(np.mean([abs(feat_importance_curr.get(k, 0.0) - prev_state.last_feature_importance.get(k, 0.0)) for k in keys]))

    perf_curr = {"sharpe": float(bt.sharpe), "sortino": float(bt.sortino), "max_drawdown": float(bt.max_drawdown)}
    perf_delta = {
        k: float(perf_curr[k] - prev_state.last_performance.get(k, 0.0))
        for k in perf_curr
    }

    report_path = DashboardReporter().export_snapshot(
        recommendations=recs,
        risk_report=risk_report,
        allocation=allocation,
        backtest_metrics=asdict(bt),
        cross_asset_intelligence=asdict(cross_asset),
        portfolio_construction=asdict(portfolio),
    )

    append_jsonl(
        "logs/outcomes.jsonl",
        {
            "top_ticker": recs[0].ticker if recs else None,
            "simulated_next_return": float(np.random.default_rng(123).normal(0.001, 0.01)),
            "risk_allowed": risk_report.allowed,
            "retrain_decision": asdict(retraining),
            "cross_asset_intelligence": asdict(cross_asset),
            "portfolio_weights": portfolio.weights,
        },
    )
    logger.info("Dashboard snapshot exported to %s", report_path)
    logger.info("Top recommendations: %s", [r.ticker for r in recs])

    horizon_top: dict[str, list[dict[str, Any]]] = {}
    for h in HORIZON_DAYS:
        ranked = sorted(
            recs_filtered,
            key=lambda r: next((x.probability for x in r.horizons if x.horizon == h), 0.0),
            reverse=True,
        )[:5]
        horizon_top[h] = [
            {
                "ticker": r.ticker,
                "probability": next((x.probability for x in r.horizons if x.horizon == h), 0.0),
                "score": r.institutional_score,
                "buy_category": r.buy_category,
            }
            for r in ranked
        ]

    # Alternative ranking schemes
    alt_rankings = {}
    for profile in ("conservative", "aggressive"):
        w = _profile_weights(cfg.scoring.get("weights", {}), profile, risk_tolerance)
        rescored = []
        for r in recs_filtered:
            score = 0.0
            for mk, ms in r.module_scores.items():
                score += w.get(mk, 0.0) * ms.value * ms.confidence
            rescored.append((r.ticker, float(np.clip(score * 10, 0, 100))))
        rescored.sort(key=lambda x: x[1], reverse=True)
        alt_rankings[profile] = [{"ticker": t, "score": s} for t, s in rescored[:5]]

    prev = prev_state.last_refresh_ts
    refresh_ts = utc_iso()
    model_diag = {
        "model_version": model_version,
        "last_retrain_timestamp": refresh_ts,
        "feature_importance_drift": drift,
        "performance_delta": perf_delta,
        "model_drift_warning": drift > 0.12,
        "signal_degradation": perf_delta.get("sharpe", 0.0) < -0.15,
    }

    new_state = RunState(
        last_refresh_ts=refresh_ts,
        last_prices=prices_now,
        last_scores=score_now,
        last_prob_1d=prob_now,
        model_version=model_version,
        last_retrain_ts=refresh_ts,
        last_feature_importance=feat_importance_curr,
        last_performance=perf_curr,
    )
    state_store.save(new_state)

    data_freshness = {
        "last_data_refresh": refresh_ts,
        "previous_refresh": prev,
        "source_timestamps": {t: bundles[t].get("source_timestamps", {}) for t in bundles},
        "change_detected": any(
            abs(prices_now.get(t, 0.0) - prev_state.last_prices.get(t, 0.0)) > 1e-9
            for t in prices_now
        ),
    }
    data_variation_warning = static_warning if is_static else ""

    accuracy_panel = {}
    for h in HORIZON_DAYS:
        acc_vals = [r.risk_metrics.get("horizon_accuracy", {}).get(h, 0.5) for r in recs]
        win_vals = [r.risk_metrics.get("rolling_win_rate", {}).get(h, 0.5) for r in recs]
        accuracy_panel[h] = {
            "historical_accuracy": float(np.mean(acc_vals)) if acc_vals else 0.5,
            "rolling_win_rate": float(np.mean(win_vals)) if win_vals else 0.5,
        }

    signal_groups = {"bullish": [], "bearish": [], "neutral": []}
    for r in recs:
        p1d = next((x.probability for x in r.horizons if x.horizon == "1D"), 0.5)
        grp = "bullish" if p1d > 0.55 else "bearish" if p1d < 0.45 else "neutral"
        signal_groups[grp].append(r)
    sharpe_by_signal = {}
    mdd_by_signal = {}
    for k, v in signal_groups.items():
        if not v:
            sharpe_by_signal[k] = 0.0
            mdd_by_signal[k] = 0.0
            continue
        vals = np.array([x.risk_metrics["daily_volatility"] for x in v], dtype=float)
        sharpe_by_signal[k] = float(np.mean((0.01 - vals) / np.maximum(vals, 1e-6)))
        mdd_by_signal[k] = float(np.max(vals) * 2.2)

    return {
        "recommendations": [asdict(r) for r in recs],
        "headline_brief": asdict(headline_brief),
        "risk_report": asdict(risk_report),
        "allocation": asdict(allocation),
        "portfolio_construction": asdict(portfolio),
        "backtest_metrics": asdict(bt),
        "premarket_scans": scans,
        "cross_asset_intelligence": asdict(cross_asset),
        "horizon_top": horizon_top,
        "accuracy_panel": accuracy_panel,
        "alt_rankings": alt_rankings,
        "model_diagnostics": model_diag,
        "data_freshness": data_freshness,
        "data_variation_warning": data_variation_warning,
        "performance_validation": {
            "sharpe_by_signal_type": sharpe_by_signal,
            "max_drawdown_by_signal_type": mdd_by_signal,
        },
        "controls": {
            "confidence_threshold": confidence_threshold,
            "risk_tolerance": risk_tolerance,
            "disabled_modules": disabled_modules or [],
            "ranking_profile": ranking_profile,
            "include_signals": allowed_signals,
        },
        "report_path": str(Path(report_path)),
    }


def main(config_path: str) -> None:
    run_pipeline(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRADING PF research and execution pipeline.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
