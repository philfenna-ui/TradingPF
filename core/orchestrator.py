from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

import numpy as np

from core.base import BaseModule
from core.logging_utils import append_jsonl
from core.models import HorizonSignal, ModuleScore, RiskReport, TradeRecommendation
from ml.multi_timeframe import HORIZON_DAYS, MultiTimeframeModeler
from risk.command_center import RiskCommandCenter

STALE_BADGE_HOURS = 0.5


class TradingPFOrchestrator:
    def __init__(
        self,
        modules: list[BaseModule],
        risk_engine: RiskCommandCenter,
        scoring_cfg: dict[str, Any],
        max_workers: int = 8,
    ) -> None:
        self.modules = modules
        self.risk_engine = risk_engine
        self.scoring_cfg = scoring_cfg
        self.max_workers = max_workers
        self.horizon_modeler = MultiTimeframeModeler()

    def _run_modules(self, ticker: str, bundle: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(module.evaluate, ticker, bundle): module for module in self.modules}
            for fut in as_completed(futures):
                module = futures[fut]
                score = fut.result()
                out[module.name] = score
        return out

    def _institutional_score(self, module_scores: dict[str, Any]) -> float:
        weights = self.scoring_cfg.get("weights", {})
        total_w = 0.0
        weighted = 0.0
        for key, score in module_scores.items():
            weight = float(weights.get(key, 0.0))
            weighted += weight * float(score.value) * float(score.confidence)
            total_w += weight
        if total_w <= 0:
            return 0.0
        return float(np.clip((weighted / total_w) * 10.0, 0.0, 100.0))

    @staticmethod
    def _buy_category(score_0_100: float) -> tuple[str, str]:
        if score_0_100 >= 70:
            return "Strong Buy", "#16a34a"
        if score_0_100 >= 58:
            return "Buy", "#22c55e"
        if score_0_100 >= 46:
            return "Accumulate", "#0ea5e9"
        if score_0_100 >= 35:
            return "Watch", "#2563eb"
        return "Avoid", "#1e3a8a"

    @staticmethod
    def _category_for_ticker(ticker: str, sector: str, category_label: str = "") -> str:
        unknown_tokens = {"unknown", "n/a", "na", "none", "null", "other", "unclassified", "uncategorized"}
        if category_label:
            label = category_label.strip()
            if label and label.lower() not in unknown_tokens:
                return label
        defense = {"LMT", "NOC", "RTX", "GD", "BA"}
        technology = {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "AVGO", "ORCL", "CRM", "NFLX", "INTC", "QCOM", "ADBE", "XLK", "QQQ"}
        finance = {"JPM", "BAC", "WFC", "GS", "MS", "SCHW", "BLK", "XLF", "KBE", "KRE"}
        energy = {"XOM", "CVX", "COP", "SLB", "XLE", "USO"}
        healthcare = {"UNH", "LLY", "PFE", "JNJ", "MRK", "XLV"}
        industrials = {"CAT", "DE", "GE", "HON", "XLI"}
        broad_equity = {"SPY", "IWM", "DIA", "VTI", "XLY", "XLP", "XLU"}
        rates = {"TLT", "IEF", "^TNX", "^TYX", "^IRX"}
        commodities = {"GLD", "SLV"}
        crypto = {"BTC-USD", "ETH-USD"}

        if ticker in defense:
            return "Military"
        if ticker in technology:
            return "Technology"
        if ticker in finance:
            return "Finance"
        if ticker in energy:
            return "Energy"
        if ticker in healthcare:
            return "Healthcare"
        if ticker in industrials:
            return "Industrials"
        if ticker in broad_equity:
            return "Equities"
        if ticker in rates:
            return "Rates"
        if ticker in commodities:
            return "Commodities"
        if ticker in crypto:
            return "Crypto"
        s = (sector or "").lower()
        if s in unknown_tokens:
            s = ""
        if "tech" in s:
            return "Technology"
        if "financial" in s or "bank" in s:
            return "Finance"
        if "energy" in s:
            return "Energy"
        if "health" in s:
            return "Healthcare"
        if "equit" in s:
            return "Equities"
        if "crypto" in s:
            return "Crypto"
        if "rates" in s:
            return "Rates"
        return sector.title() if sector else "General"

    @staticmethod
    def _plain_reason(
        ticker: str,
        category: str,
        buy_category: str,
        module_scores: dict[str, Any],
        horizons: list[HorizonSignal],
    ) -> tuple[str, list[str]]:
        h1 = next((h for h in horizons if h.horizon == "1D"), horizons[0])
        tech = module_scores.get("technical")
        catalyst = module_scores.get("catalyst")
        macro = module_scores.get("macro_alignment")
        vol = module_scores.get("volatility_adjustment")
        themes = catalyst.metadata.get("themes", []) if catalyst else []
        ro = float(macro.metadata.get("risk_on_off_score", 0.5)) if macro else 0.5
        tech_v = float(tech.value) if tech else 5.0
        cat_v = float(catalyst.metadata.get("catalyst_strength_score", catalyst.value)) if catalyst else 5.0
        vol_v = float(vol.value) if vol else 5.0

        narrative: list[str] = []
        narrative.append(f"Status: {ticker} is a {buy_category} in {category}.")

        if "geopolitical_conflict" in themes and "defense" in category.lower():
            narrative.append("Why now: conflict headlines can increase demand for defense companies.")
        elif "energy_supply" in themes and "energy" in category.lower():
            narrative.append("Why now: supply-risk headlines can support energy prices and energy stocks.")
        elif "rates_inflation" in themes:
            narrative.append("Why now: rates/inflation headlines can move this stock in the short term.")
        else:
            narrative.append("Why now: recent news flow is active and is helping drive this setup.")

        if tech_v >= 6:
            trend_note = "trend is supportive"
        elif tech_v >= 4:
            trend_note = "trend is mixed"
        else:
            trend_note = "trend is weak"
        narrative.append(f"What supports it: news score {cat_v:.1f}/10 and {trend_note}.")

        narrative.append(
            f"Near-term view: 1-day probability is {h1.probability*100:.1f}% with expected range {h1.target_low:.2f} to {h1.target_high:.2f}."
        )
        narrative.append(
            "Risk check: if headlines cool off or the market turns risk-off, this can downgrade quickly."
        )
        return " ".join(narrative), narrative

    @staticmethod
    def _horizon_signals(
        price: float,
        vol: float,
        probs: dict[str, float],
        target_ranges: dict[str, tuple[float, float]],
    ) -> list[HorizonSignal]:
        out: list[HorizonSignal] = []
        for h, days in HORIZON_DAYS.items():
            p = float(probs.get(h, 0.5))
            sigma = max(vol, 0.008) * np.sqrt(days)
            tgt_low, tgt_high = target_ranges.get(h, (price * (1 - 1.3 * sigma), price * (1 + 1.3 * sigma)))
            upside = (tgt_high / price - 1) * 100
            downside = (tgt_low / price - 1) * 100
            risk_cls = "LOW" if sigma < 0.02 else "MEDIUM" if sigma < 0.04 else "HIGH"
            out.append(
                HorizonSignal(
                    horizon=h,
                    probability=p,
                    target_low=float(tgt_low),
                    target_high=float(tgt_high),
                    upside_pct=float(upside),
                    downside_pct=float(downside),
                    risk_classification=risk_cls,
                )
            )
        return out

    def generate_recommendations(
        self,
        universe: list[str],
        bundles: dict[str, dict[str, Any]],
        portfolio_returns: np.ndarray,
        exposures: dict[str, dict[str, float]],
        confidence_threshold: float = 0.0,
        risk_tolerance: float = 0.5,
    ) -> tuple[list[TradeRecommendation], RiskReport]:
        risk_report = self.risk_engine.evaluate(portfolio_returns, exposures)
        recs: list[TradeRecommendation] = []
        for ticker in universe:
            bundle = bundles[ticker]
            module_scores = self._run_modules(ticker, bundle)
            institutional_score = self._institutional_score(module_scores)
            catalyst_themes = module_scores.get("catalyst", ModuleScore(module="catalyst", value=0, confidence=0)).metadata.get("themes", [])
            defense_tickers = {"LMT", "NOC", "RTX", "GD", "BA"}
            if "geopolitical_conflict" in catalyst_themes and ticker in defense_tickers:
                institutional_score = float(min(100.0, institutional_score + 8.0))
            if "energy_supply" in catalyst_themes and ticker in {"XLE", "CVX", "XOM", "USO"}:
                institutional_score = float(min(100.0, institutional_score + 5.0))
            current_price = float(bundle["daily"]["close"].iloc[-1])
            vol = float(bundle["daily"]["close"].pct_change().dropna().tail(30).std())
            model_out = self.horizon_modeler.fit_predict(bundle["daily"], institutional_score)
            buy_category, buy_color = self._buy_category(institutional_score)
            quality = bundle.get("data_quality", {})
            mock_fields = list(quality.get("mock_fields", []))
            core_market_mock_fields = {"daily", "intraday"}
            mock_data_used = bool(quality.get("is_mock", False)) and any(f in core_market_mock_fields for f in mock_fields)
            stale_age_hours = float(quality.get("stale_age_hours", 0.0))
            stale_data_used = bool(quality.get("is_stale", False)) and stale_age_hours >= STALE_BADGE_HOURS

            horizons = self._horizon_signals(current_price, vol, model_out.probabilities, model_out.target_ranges)
            weights = self.scoring_cfg.get("weights", {})
            explainability = {
                k: {
                    "raw_score": float(v.value),
                    "confidence": float(v.confidence),
                    "weight": float(weights.get(k, 0.0)),
                    "weighted_contribution": float(weights.get(k, 0.0) * v.value * v.confidence),
                }
                for k, v in module_scores.items()
            }
            trend = bundle["daily"]["close"].tail(40).values.astype(float)
            if len(trend) > 1:
                min_v, max_v = float(np.min(trend)), float(np.max(trend))
                denom = max(max_v - min_v, 1e-9)
                spark = [float((x - min_v) / denom) for x in trend]
            else:
                spark = [0.5]
            narrative = (
                f"{ticker} composite score {institutional_score:.1f} ({buy_category}); "
                f"risk-first allocation applies regime and tail constraints. "
                f"Data source quality: {'cached old data used' if stale_data_used else ('fallback/mock used' if mock_data_used else 'live feeds')}."
            )
            category = self._category_for_ticker(
                ticker,
                str(bundle.get("sector", "general")),
                str(bundle.get("category_label", "")),
            )
            plain_reason, plain_reason_points = self._plain_reason(ticker, category, buy_category, module_scores, horizons)
            risk_metrics = {
                "daily_volatility": vol,
                "var_95": risk_report.portfolio_var_95,
                "expected_shortfall_95": risk_report.expected_shortfall_95,
                "horizon_accuracy": model_out.historical_accuracy,
                "rolling_win_rate": model_out.rolling_win_rate,
                "probability_confidence": model_out.confidence,
            }
            if confidence_threshold > 0 and max(h.probability for h in horizons) < confidence_threshold:
                continue
            rec = TradeRecommendation(
                ticker=ticker,
                name=str(bundle.get("company_name", ticker)),
                category=category,
                current_price=current_price,
                institutional_score=institutional_score,
                buy_category=buy_category,
                buy_color=buy_color,
                mock_data_used=mock_data_used,
                mock_fields=mock_fields,
                stale_data_used=stale_data_used,
                stale_age_hours=stale_age_hours,
                action="BUY" if buy_category in {"Strong Buy", "Buy"} else "WATCH" if buy_category == "Accumulate" else "AVOID",
                module_scores=module_scores,
                horizons=horizons,
                narrative=narrative + f" Risk tolerance setting={risk_tolerance:.2f}.",
                plain_reason=plain_reason,
                plain_reason_points=plain_reason_points,
                risk_metrics=risk_metrics,
                explainability=explainability,
                sparkline=spark,
            )
            recs.append(rec)
            append_jsonl("logs/predictions.jsonl", asdict(rec))

        recs.sort(key=lambda x: x.institutional_score, reverse=True)
        return recs, risk_report
