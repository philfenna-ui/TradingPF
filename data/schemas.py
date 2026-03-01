from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.exceptions import DataError


REQUIRED_DAILY_COLUMNS = {"close", "volume", "vwap", "atr_proxy"}
REQUIRED_INTRADAY_COLUMNS = {"close", "volume"}
REQUIRED_OPTIONS_COLUMNS = {"strike", "type", "open_interest", "volume", "iv", "premium"}
REQUIRED_MACRO_KEYS = {
    "fed_funds_rate",
    "cpi_yoy",
    "gdp_qoq",
    "yield_2y",
    "yield_10y",
    "yield_30y",
    "liquidity_index",
    "vix",
    "vvix",
    "dxy",
}


@dataclass(slots=True)
class NewsEvent:
    timestamp: datetime
    headline: str
    sentiment: float
    sector: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketBundle:
    ticker: str
    daily: pd.DataFrame
    intraday: pd.DataFrame
    macro: dict[str, float]
    options_chain: pd.DataFrame
    dark_pool: dict[str, float]
    futures: dict[str, float]
    crypto_funding: dict[str, float]
    yield_curve: dict[str, float]
    news: list[NewsEvent]
    macro_news: list[NewsEvent]
    sector: str
    asset_class: str
    company_name: str = ""
    category_label: str = ""
    data_quality: dict[str, Any] = field(default_factory=dict)
    source_timestamps: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not REQUIRED_DAILY_COLUMNS.issubset(self.daily.columns):
            missing = REQUIRED_DAILY_COLUMNS - set(self.daily.columns)
            raise DataError(f"{self.ticker}: daily data missing columns: {sorted(missing)}")
        if not REQUIRED_INTRADAY_COLUMNS.issubset(self.intraday.columns):
            missing = REQUIRED_INTRADAY_COLUMNS - set(self.intraday.columns)
            raise DataError(f"{self.ticker}: intraday data missing columns: {sorted(missing)}")
        if not REQUIRED_OPTIONS_COLUMNS.issubset(self.options_chain.columns):
            missing = REQUIRED_OPTIONS_COLUMNS - set(self.options_chain.columns)
            raise DataError(f"{self.ticker}: options data missing columns: {sorted(missing)}")
        if not REQUIRED_MACRO_KEYS.issubset(self.macro.keys()):
            missing = REQUIRED_MACRO_KEYS - set(self.macro.keys())
            raise DataError(f"{self.ticker}: macro data missing keys: {sorted(missing)}")
        if self.daily.empty or self.intraday.empty:
            raise DataError(f"{self.ticker}: price frames must not be empty")
        if len(self.news) == 0 and len(self.macro_news) == 0:
            raise DataError(f"{self.ticker}: at least one news item is required")


def normalize_bundle_dict(bundle: dict[str, Any], ticker: str) -> MarketBundle:
    daily = bundle["daily"].copy()
    intraday = bundle["intraday"].copy()
    if isinstance(daily.index, pd.DatetimeIndex):
        if daily.index.tz is not None:
            daily.index = daily.index.tz_convert("UTC").tz_localize(None)
        daily = daily.sort_index()
    if isinstance(intraday.index, pd.DatetimeIndex):
        if intraday.index.tz is not None:
            intraday.index = intraday.index.tz_convert("UTC").tz_localize(None)
        intraday = intraday.sort_index()

    events = [
        NewsEvent(
            timestamp=pd.Timestamp(item["timestamp"]).to_pydatetime(),
            headline=str(item["headline"]),
            sentiment=float(item.get("sentiment", 0.0)),
            sector=str(item.get("sector", "unknown")),
            source=str(item.get("source", "unknown")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in bundle.get("news", [])
    ]
    macro_events = [
        NewsEvent(
            timestamp=pd.Timestamp(item["timestamp"]).to_pydatetime(),
            headline=str(item["headline"]),
            sentiment=float(item.get("sentiment", 0.0)),
            sector=str(item.get("sector", "macro")),
            source=str(item.get("source", "unknown")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in bundle.get("macro_news", [])
    ]
    obj = MarketBundle(
        ticker=ticker,
        daily=daily,
        intraday=intraday,
        macro=dict(bundle["macro"]),
        options_chain=bundle["options_chain"].copy(),
        dark_pool=dict(bundle["dark_pool"]),
        futures=dict(bundle["futures"]),
        crypto_funding=dict(bundle["crypto_funding"]),
        yield_curve=dict(bundle["yield_curve"]),
        news=events,
        macro_news=macro_events,
        sector=str(bundle.get("sector", "unknown")),
        asset_class=str(bundle.get("asset_class", "unknown")),
        company_name=str(bundle.get("company_name", ticker)),
        category_label=str(bundle.get("category_label", "")),
        data_quality=dict(bundle.get("data_quality", {})),
        source_timestamps=dict(bundle.get("source_timestamps", {})),
    )
    obj.validate()
    return obj
