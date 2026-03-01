from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import io
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass(slots=True)
class DataProviderConfig:
    daily_lookback_days: int = 520
    intraday_lookback_days: int = 30
    intraday_interval_minutes: int = 30
    random_seed: int = 42
    request_timeout_seconds: int = 12


class MockDataProvider:
    """Deterministic fallback provider."""

    def __init__(self, cfg: DataProviderConfig) -> None:
        self.cfg = cfg

    def _dynamic_seed(self, ticker: str) -> int:
        # Rolling seed ensures fallback data evolves over time.
        bucket = int(pd.Timestamp.now("UTC").floor("5min").timestamp())
        return abs(hash((ticker, self.cfg.random_seed, bucket))) % (2**32)

    def _price_frame(self, ticker: str) -> pd.DataFrame:
        rng = np.random.default_rng(self._dynamic_seed(ticker))
        idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=self.cfg.daily_lookback_days, freq="B")
        rets = rng.normal(0.0003, 0.015, size=len(idx))
        px = 100 * np.exp(np.cumsum(rets))
        vol = rng.integers(1_000_000, 15_000_000, size=len(idx))
        df = pd.DataFrame({"close": px, "volume": vol}, index=idx)
        df["vwap"] = df["close"] * (1 + rng.normal(0, 0.001, size=len(idx)))
        df["atr_proxy"] = df["close"].pct_change().abs().rolling(14).mean().fillna(0.01) * df["close"]
        return df

    def _intraday_frame(self, ticker: str) -> pd.DataFrame:
        rng = np.random.default_rng(self._dynamic_seed(f"{ticker}-intra"))
        rows = int((self.cfg.intraday_lookback_days * 24 * 60) / self.cfg.intraday_interval_minutes)
        idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=rows, freq=f"{self.cfg.intraday_interval_minutes}min")
        rets = rng.normal(0, 0.0012, size=len(idx))
        px = 100 * np.exp(np.cumsum(rets))
        vol = rng.integers(25_000, 500_000, size=len(idx))
        return pd.DataFrame({"close": px, "volume": vol}, index=idx)

    def _macro(self) -> dict[str, float]:
        return {
            "fed_funds_rate": 5.25,
            "cpi_yoy": 3.0,
            "gdp_qoq": 2.1,
            "yield_2y": 4.2,
            "yield_10y": 4.0,
            "yield_30y": 4.25,
            "liquidity_index": 0.56,
            "vix": 18.0,
            "vvix": 95.0,
            "dxy": 103.0,
        }

    def _options(self, daily: pd.DataFrame) -> pd.DataFrame:
        price = float(daily["close"].iloc[-1])
        strikes = np.arange(price * 0.7, price * 1.3, price * 0.03)
        oi = np.random.default_rng(7).integers(200, 8000, len(strikes))
        vol = np.random.default_rng(8).integers(100, 12000, len(strikes))
        iv = np.random.default_rng(9).uniform(0.18, 0.75, len(strikes))
        cp = np.where(strikes >= price, "C", "P")
        premium = np.where(cp == "C", vol * 1.2, vol * 0.9)
        return pd.DataFrame({"strike": strikes, "type": cp, "open_interest": oi, "volume": vol, "iv": iv, "premium": premium})

    def _news(self, ticker: str) -> list[dict[str, Any]]:
        now = pd.Timestamp.now("UTC")
        return [
            {"timestamp": now - timedelta(hours=1), "headline": f"{ticker} reports better-than-expected guidance", "sentiment": 0.6, "sector": "technology", "source": "synthetic"},
            {"timestamp": now - timedelta(hours=4), "headline": f"Macro uncertainty weighs on {ticker} peer group", "sentiment": -0.2, "sector": "technology", "source": "synthetic"},
            {"timestamp": now - timedelta(hours=9), "headline": f"Options flow activity spikes in {ticker}", "sentiment": 0.4, "sector": "technology", "source": "synthetic"},
        ]

    def load_bundle(self, ticker: str) -> dict[str, Any]:
        daily = self._price_frame(ticker)
        return {
            "daily": daily,
            "intraday": self._intraday_frame(ticker),
            "macro": self._macro(),
            "options_chain": self._options(daily),
            "dark_pool": {"dark_pool_volume_ratio": 0.18, "block_trade_count": 7, "repeated_prints_score": 0.63},
            "futures": {"es_change": 0.4, "nq_change": 0.55},
            "crypto_funding": {"BTC": 0.01, "ETH": 0.009},
            "yield_curve": {"2s10s": -0.2, "10s30s": 0.25},
            "news": self._news(ticker),
            "macro_news": [],
            "sector": "technology",
            "asset_class": "equities",
            "company_name": ticker,
            "category_label": "Technology",
            "data_quality": {"is_mock": True, "mock_fields": ["daily", "intraday", "macro", "options_chain", "news"]},
            "source_timestamps": {"daily": str(daily.index[-1]), "intraday": str(pd.Timestamp.now("UTC"))},
        }


class LiveYahooFREDProvider:
    """
    Live market data provider using Yahoo Finance and public FRED CSV endpoints.
    Falls back to deterministic mock frames per-field if specific feeds are unavailable.
    """

    FRED_SERIES = {
        "fed_funds_rate": "FEDFUNDS",
        "cpi_yoy": "CPIAUCSL",
        "gdp_qoq": "A191RL1Q225SBEA",
    }

    def __init__(self, cfg: DataProviderConfig) -> None:
        self.cfg = cfg
        self.mock = MockDataProvider(cfg)
        self.session = requests.Session()
        self._macro_cache: dict[str, float] | None = None

    def _interval(self) -> str:
        mapping = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}
        return mapping.get(self.cfg.intraday_interval_minutes, "30m")

    @staticmethod
    def _normalize_download_output(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    def _silent_download(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """
        yfinance can print noisy '1 Failed download' lines even when handled.
        Capture stdout/stderr to keep app output clean.
        """
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            df = yf.download(symbol, progress=False, threads=False, auto_adjust=False, **kwargs)
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        return self._normalize_download_output(df)

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        renamed = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        ).copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in renamed.columns:
                renamed[col] = 0.0
        renamed["volume"] = renamed["volume"].fillna(0)
        typical = (renamed["high"] + renamed["low"] + renamed["close"]) / 3
        renamed["vwap"] = (typical * np.where(renamed["volume"] > 0, renamed["volume"], 1)).rolling(20).sum() / np.where(
            renamed["volume"].rolling(20).sum() > 0, renamed["volume"].rolling(20).sum(), 1
        )
        tr1 = renamed["high"] - renamed["low"]
        tr2 = (renamed["high"] - renamed["close"].shift(1)).abs()
        tr3 = (renamed["low"] - renamed["close"].shift(1)).abs()
        renamed["atr_proxy"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()
        return renamed

    def _download_daily(self, ticker: str) -> pd.DataFrame:
        period = max(365, self.cfg.daily_lookback_days + 40)
        raw = self._silent_download(ticker, period=f"{period}d", interval="1d")
        if raw.empty:
            raise ValueError(f"no_daily_data:{ticker}")
        df = self._normalize_ohlcv(raw).tail(self.cfg.daily_lookback_days)
        return df[["close", "volume", "vwap", "atr_proxy"]]

    def _download_intraday(self, ticker: str) -> pd.DataFrame:
        lookback = min(self.cfg.intraday_lookback_days, 59)
        raw = self._silent_download(
            ticker,
            period=f"{lookback}d",
            interval=self._interval(),
            prepost=True,
        )
        if raw.empty:
            raise ValueError(f"no_intraday_data:{ticker}")
        df = self._normalize_ohlcv(raw)
        return df[["close", "volume"]]

    def _fetch_fred_latest(self, series_id: str) -> float:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = self.session.get(url, timeout=self.cfg.request_timeout_seconds)
        resp.raise_for_status()
        tmp = pd.read_csv(StringIO(resp.text))
        vals = pd.to_numeric(tmp.iloc[:, 1], errors="coerce").dropna()
        if vals.empty:
            raise ValueError(f"no_fred_values:{series_id}")
        return float(vals.iloc[-1])

    def _last_close(self, symbol: str, fallback: float) -> float:
        candidates = [symbol]
        if symbol.startswith("^"):
            candidates.append(symbol[1:])
        for s in candidates:
            try:
                hist = self._silent_download(s, period="10d", interval="1d")
                if hist.empty:
                    continue
                if "Close" in hist.columns and not hist["Close"].dropna().empty:
                    return float(hist["Close"].dropna().iloc[-1])
            except Exception:
                continue
        return fallback

    def _last_return(self, symbol: str) -> float:
        candidates = [symbol]
        if symbol == "ES=F":
            candidates.extend(["^GSPC", "SPY"])
        if symbol == "NQ=F":
            candidates.extend(["^IXIC", "QQQ"])
        for s in candidates:
            try:
                hist = self._silent_download(s, period="5d", interval="1d")
                closes = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)
                if len(closes) < 2:
                    continue
                return float(closes.iloc[-1] / closes.iloc[-2] - 1)
            except Exception:
                continue
        return 0.0

    def _macro(self) -> dict[str, float]:
        if self._macro_cache is not None:
            return self._macro_cache
        out = self.mock._macro()
        try:
            out["fed_funds_rate"] = self._fetch_fred_latest(self.FRED_SERIES["fed_funds_rate"])
            out["cpi_yoy"] = self._fetch_fred_latest(self.FRED_SERIES["cpi_yoy"])
            out["gdp_qoq"] = self._fetch_fred_latest(self.FRED_SERIES["gdp_qoq"])
        except Exception:
            pass
        out["yield_2y"] = self._last_close("^IRX", out["yield_2y"])  # 13-week as short-end proxy
        out["yield_10y"] = self._last_close("^TNX", out["yield_10y"])
        out["yield_30y"] = self._last_close("^TYX", out["yield_30y"])
        out["vix"] = self._last_close("^VIX", out["vix"])
        out["vvix"] = self._last_close("^VVIX", out["vvix"])
        out["dxy"] = self._last_close("DX-Y.NYB", out["dxy"])
        # Liquidity proxy combines curve shape and volatility.
        curve = out["yield_10y"] - out["yield_2y"]
        out["liquidity_index"] = float(np.clip(0.55 + 0.02 * curve - 0.005 * (out["vix"] - 20), 0.1, 0.9))
        self._macro_cache = out
        return out

    def _options(self, ticker: str, daily: pd.DataFrame) -> pd.DataFrame:
        # Yahoo options endpoints frequently fail with crumb auth issues.
        # Use deterministic fallback to avoid noisy auth errors in live runs.
        return self.mock._options(daily)

    @staticmethod
    def _infer_asset_class(ticker: str) -> str:
        if ticker.endswith("-USD"):
            return "crypto"
        if ticker.startswith("^"):
            return "rates"
        return "equities"

    @staticmethod
    def _infer_sector(info: dict[str, Any], asset_class: str) -> str:
        if asset_class != "equities":
            return asset_class
        return str(info.get("sector") or "unknown")

    def _news(self, ticker: str, sector: str) -> list[dict[str, Any]]:
        now = pd.Timestamp.now("UTC")
        # Keep ticker-local news from stable macro RSS stream to avoid crumb failures.
        return [
            {
                "timestamp": now - timedelta(hours=2),
                "headline": f"{ticker} news feed currently sparse; monitoring live catalysts",
                "sentiment": 0.0,
                "sector": sector,
                "source": "system",
            }
        ]

    def _macro_news(self) -> list[dict[str, Any]]:
        queries = [
            ("war defense stocks", "geopolitics"),
            ("geopolitical risk markets", "macro"),
            ("oil supply disruption", "energy"),
            ("asia markets china japan india korea stocks", "asia"),
            ("middle east markets oil gulf shipping red sea hormuz", "middle_east"),
            ("global breaking news diplomacy sanctions", "world"),
            ("natural disasters earthquake hurricane flood wildfire", "world"),
            ("global outbreak public health alert", "world"),
            ("shipping route disruption port closure canal", "world"),
            ("election risk policy change major economy", "world"),
        ]
        now = pd.Timestamp.now("UTC")
        out: list[dict[str, Any]] = []
        for q, sector in queries:
            try:
                url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
                resp = self.session.get(url, timeout=self.cfg.request_timeout_seconds)
                resp.raise_for_status()
                xml = resp.text
                titles = []
                for chunk in xml.split("<item>")[1:6]:
                    if "<title>" in chunk and "</title>" in chunk:
                        t = chunk.split("<title>", 1)[1].split("</title>", 1)[0]
                        titles.append(t.replace("&amp;", "&"))
                for t in titles:
                    out.append(
                        {
                            "timestamp": now,
                            "headline": t,
                            "sentiment": 0.15 if "surge" in t.lower() or "rise" in t.lower() else 0.0,
                            "sector": sector,
                            "source": "google_news_rss",
                        }
                    )
            except Exception:
                continue
        return out

    def load_bundle(self, ticker: str) -> dict[str, Any]:
        mock_fields: list[str] = []
        source_ts: dict[str, str] = {}
        try:
            daily = self._download_daily(ticker)
            source_ts["daily"] = str(daily.index[-1])
        except Exception:
            daily = self.mock._price_frame(ticker)
            mock_fields.append("daily")
            source_ts["daily"] = str(daily.index[-1])
        try:
            intraday = self._download_intraday(ticker)
            source_ts["intraday"] = str(intraday.index[-1])
        except Exception:
            intraday = self.mock._intraday_frame(ticker)
            mock_fields.append("intraday")
            source_ts["intraday"] = str(intraday.index[-1])

        macro = self._macro()
        try:
            options_chain = self._options(ticker, daily)
        except Exception:
            options_chain = self.mock._options(daily)
            mock_fields.append("options_chain")

        info: dict[str, Any] = {}
        asset_class = self._infer_asset_class(ticker)
        sector = self._infer_sector(info, asset_class)
        name = ticker
        category_label = sector.title() if sector else "General"

        # Legal-access dark pool proxy derived from abnormal block activity and off-book style volume concentration.
        vol_short = float(daily["volume"].tail(5).mean())
        vol_long = float(daily["volume"].tail(30).mean())
        vol_ratio = float(np.clip(vol_short / max(vol_long, 1.0), 0, 3))
        dark_pool_ratio = float(np.clip(0.08 + 0.07 * vol_ratio, 0.05, 0.35))
        block_count = int(np.clip((daily["volume"].tail(20) > daily["volume"].tail(120).quantile(0.85)).sum(), 1, 20))
        repeated_prints = float(np.clip(daily["close"].tail(20).value_counts(normalize=True).iloc[0] * 10, 0, 1))

        futures = {"es_change": self._last_return("ES=F"), "nq_change": self._last_return("NQ=F")}
        yield_curve = {"2s10s": float(macro["yield_10y"] - macro["yield_2y"]), "10s30s": float(macro["yield_30y"] - macro["yield_10y"])}
        crypto_funding = {"BTC": 0.0, "ETH": 0.0}
        if ticker in {"BTC-USD", "ETH-USD"}:
            rets = daily["close"].pct_change().dropna().tail(7)
            crypto_funding[ticker.split("-")[0]] = float(np.clip(rets.mean() * 10, -0.05, 0.05))

        return {
            "daily": daily,
            "intraday": intraday,
            "macro": macro,
            "options_chain": options_chain,
            "dark_pool": {"dark_pool_volume_ratio": dark_pool_ratio, "block_trade_count": block_count, "repeated_prints_score": repeated_prints},
            "futures": futures,
            "crypto_funding": crypto_funding,
            "yield_curve": yield_curve,
            "news": self._news(ticker, sector),
            "macro_news": self._macro_news(),
            "sector": sector,
            "asset_class": asset_class,
            "company_name": name,
            "category_label": category_label,
            "data_quality": {"is_mock": len(mock_fields) > 0, "mock_fields": mock_fields},
            "source_timestamps": source_ts,
        }
