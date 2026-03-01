from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Protocol

from data.feature_store import FeatureStore
from data.providers import DataProviderConfig, LiveYahooFREDProvider, MockDataProvider
from data.schemas import MarketBundle, normalize_bundle_dict


class Provider(Protocol):
    def load_bundle(self, ticker: str) -> dict: ...


@dataclass(slots=True)
class IngestionConfig:
    provider: str
    daily_lookback_days: int
    intraday_lookback_days: int
    intraday_interval_minutes: int
    random_seed: int


class ProviderRegistry:
    @staticmethod
    def build(cfg: IngestionConfig) -> Provider:
        provider_key = cfg.provider.lower()
        provider_cfg = DataProviderConfig(
            daily_lookback_days=cfg.daily_lookback_days,
            intraday_lookback_days=cfg.intraday_lookback_days,
            intraday_interval_minutes=cfg.intraday_interval_minutes,
            random_seed=cfg.random_seed,
        )
        if provider_key in {"mock", "paper", "simulated"}:
            return MockDataProvider(provider_cfg)
        if provider_key in {"live", "live_yahoo_fred", "yahoo", "yfinance", "polygon", "ibkr"}:
            return LiveYahooFREDProvider(provider_cfg)
        raise ValueError(f"Unknown data provider: {cfg.provider}")


class IngestionPipeline:
    def __init__(self, provider: Provider, feature_store: FeatureStore, max_workers: int = 6) -> None:
        self.provider = provider
        self.feature_store = feature_store
        self.max_workers = max(1, max_workers)

    def fetch_universe(self, universe: list[str]) -> dict[str, MarketBundle]:
        out: dict[str, MarketBundle] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(universe) or 1)) as ex:
            futs = {ex.submit(self.provider.load_bundle, ticker): ticker for ticker in universe}
            for fut in as_completed(futs):
                ticker = futs[fut]
                raw = fut.result()
                dq = dict(raw.get("data_quality", {}))
                mock_fields = list(dq.get("mock_fields", []))
                used_old = False
                replaced_fields: set[str] = set()
                stale_hours = 0.0
                if mock_fields and self.feature_store.exists("bundle_cache", ticker):
                    try:
                        cached = self.feature_store.load_pickle("bundle_cache", ticker)
                        cached_bundle = cached.get("bundle", {})
                        saved_at = cached.get("saved_at")
                        if saved_at:
                            saved_ts = datetime.fromisoformat(saved_at)
                            stale_hours = max(
                                0.0,
                                (datetime.now(timezone.utc) - saved_ts.astimezone(timezone.utc)).total_seconds() / 3600.0,
                            )
                        for fld in mock_fields:
                            if fld in cached_bundle:
                                raw[fld] = cached_bundle[fld]
                                used_old = True
                                replaced_fields.add(fld)
                    except Exception:
                        pass
                if used_old:
                    remaining_mock_fields = [f for f in mock_fields if f not in replaced_fields]
                    raw["data_quality"] = {
                        **dq,
                        "is_mock": len(remaining_mock_fields) > 0,
                        "mock_fields": remaining_mock_fields,
                        "is_stale": True,
                        "stale_age_hours": stale_hours,
                    }
                else:
                    raw["data_quality"] = {**dq, "is_stale": False, "stale_age_hours": 0.0}

                normalized = normalize_bundle_dict(raw, ticker=ticker)
                out[ticker] = normalized
                self.feature_store.save_frame("daily", ticker, normalized.daily)
                self.feature_store.save_frame("intraday", ticker, normalized.intraday)
                self.feature_store.save_frame("options", ticker, normalized.options_chain)
                self.feature_store.save_json("macro", ticker, normalized.macro)
                self.feature_store.save_pickle(
                    "bundle_cache",
                    ticker,
                    {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "bundle": raw,
                    },
                )
        return out
