from __future__ import annotations

from pathlib import Path
import sys
import json
import os
import re
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, url_for
import yfinance as yf
import requests

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import run_pipeline
from core.config import load_config
from execution.broker import PaperBroker

app = Flask(__name__)
DEFAULT_CONFIG = str((BASE_DIR / "config" / "default.yaml").resolve())
CONFIG_ENV_VAR = "TRADING_PF_CONFIG"
LAST_DATA: dict | None = None
LAST_CONTROL_KEY: str | None = None
WATCHLIST_PATH = BASE_DIR / "logs" / "watchlist.json"
LAST_PAYLOAD_PATH = BASE_DIR / "logs" / "last_full_payload.json"
DISCOVERY_CACHE_PATH = BASE_DIR / "logs" / "discovery_top1000_cache.json"
ALL_SIGNALS = ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"]
DEFAULT_INCLUDED_SIGNALS = ["Strong Buy", "Buy", "Accumulate"]
DISCOVERY_UNIVERSE = [
    "SPY","QQQ","IWM","DIA","VTI","TLT","IEF","LQD","HYG","GLD","SLV","USO","XLE","XLF","XLK","XLV","XLI","XLP","XLY","XLU",
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","AVGO","ORCL","CRM","NFLX","INTC","QCOM","ADBE",
    "JPM","BAC","WFC","GS","MS","SCHW","BLK",
    "LMT","NOC","RTX","GD","BA",
    "XOM","CVX","COP","SLB",
    "UNH","LLY","PFE","JNJ","MRK",
    "CAT","DE","GE","HON",
    "BTC-USD","ETH-USD",
]


def _load_discovery_from_cache(max_age_hours: float = 12.0) -> list[str] | None:
    if not DISCOVERY_CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(DISCOVERY_CACHE_PATH.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(str(raw.get("saved_at", "")))
        if datetime.now(timezone.utc) - saved_at.astimezone(timezone.utc) > timedelta(hours=max_age_hours):
            return None
        symbols = [str(x).upper() for x in raw.get("symbols", []) if str(x).strip()]
        return symbols if symbols else None
    except Exception:
        return None


def _save_discovery_cache(symbols: list[str]) -> None:
    try:
        DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"saved_at": datetime.now(timezone.utc).isoformat(), "symbols": symbols}
        DISCOVERY_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _fetch_us_listed_symbols() -> list[str]:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    symbols: set[str] = set()
    # Official Nasdaq symbol directories (NASDAQ + NYSE/AMEX/others).
    feeds = [
        ("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "nasdaq"),
        ("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "other"),
    ]
    for url, kind in feeds:
        try:
            resp = session.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
            if len(lines) < 3:
                continue
            # Skip header and footer lines.
            for ln in lines[1:-1]:
                parts = ln.split("|")
                if kind == "nasdaq":
                    if len(parts) < 7:
                        continue
                    sym = parts[0].strip().upper()
                    test_issue = parts[3].strip().upper()
                    is_etf = parts[6].strip().upper()
                    if test_issue == "Y" or is_etf == "Y":
                        continue
                else:
                    if len(parts) < 7:
                        continue
                    sym = parts[0].strip().upper()
                    is_etf = parts[4].strip().upper()
                    test_issue = parts[6].strip().upper()
                    if test_issue == "Y" or is_etf == "Y":
                        continue
                # Keep common equity-like tickers; drop obvious derivatives/structured symbols.
                if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", sym):
                    continue
                if any(x in sym for x in ["$", "^", "/"]):
                    continue
                symbols.add(sym)
        except Exception:
            continue
    return sorted(symbols)


def _fetch_top_1000_most_active() -> list[str]:
    # Use Nasdaq screener as the primary broad-company universe source (1000 symbols).
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    out: list[str] = []
    seen: set[str] = set()
    for offset in (0, 1000, 2000):
        try:
            resp = session.get(
                "https://api.nasdaq.com/api/screener/stocks",
                params={"tableonly": "true", "limit": 1000, "offset": offset},
                timeout=15,
                headers=headers,
            )
            resp.raise_for_status()
            rows = ((((resp.json() or {}).get("data") or {}).get("table") or {}).get("rows") or [])
            if not rows:
                break
            for r in rows:
                sym = str(r.get("symbol", "")).upper().strip()
                if not sym or sym in seen:
                    continue
                if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", sym):
                    continue
                seen.add(sym)
                out.append(sym)
                if len(out) >= 1000:
                    return out
        except Exception:
            break
    if len(out) >= 1000:
        return out[:1000]

    # Fallback: rank broad US listed company universe by Yahoo regular market volume.
    candidates = _fetch_us_listed_symbols()
    if not candidates:
        return out
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    # Rank candidates by regular market volume via Yahoo quote endpoint.
    ranked: list[tuple[str, float]] = []
    syms = sorted(candidates)
    for i in range(0, len(syms), 200):
        chunk = syms[i : i + 200]
        try:
            resp = session.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": ",".join(chunk)},
                timeout=10,
                headers=headers,
            )
            resp.raise_for_status()
            res = (((resp.json() or {}).get("quoteResponse") or {}).get("result") or [])
            for q in res:
                sym = str(q.get("symbol", "")).upper().strip()
                qtype = str(q.get("quoteType", "")).upper().strip()
                vol = float(q.get("regularMarketVolume") or 0.0)
                if sym and qtype == "EQUITY":
                    ranked.append((sym, vol))
        except Exception:
            continue

    ranked.sort(key=lambda x: x[1], reverse=True)
    out = [s for s, _ in ranked if s]
    # Keep unique in-ranked order.
    uniq: list[str] = []
    seen: set[str] = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
        if len(uniq) >= 1000:
            break
    if uniq:
        return uniq
    return out


def _get_discovery_universe_top1000() -> list[str]:
    cached = _load_discovery_from_cache(max_age_hours=12.0)
    if cached:
        return cached
    fetched = _fetch_top_1000_most_active()
    if fetched:
        _save_discovery_cache(fetched)
        return fetched
    # Safe fallback if upstream screener is unavailable/rate-limited.
    return DISCOVERY_UNIVERSE


def _empty_dashboard_data() -> dict:
    return {
        "headline_brief": {"us": [], "europe": [], "world": [], "summary": "", "actions": [], "major_events": []},
        "data_freshness": {"last_data_refresh": "Not refreshed yet"},
        "model_diagnostics": {"model_version": "N/A", "model_drift_warning": False},
        "risk_report": {
            "regime": "Unknown",
            "tail_risk_probability": 0.0,
            "sector_exposure": {},
            "asset_class_exposure": {},
        },
        "cross_asset_intelligence": {"risk_on_off_score": 0.0, "correlation_30d": {}},
        "recommendations": [],
        "search_results": [],
        "horizon_top": {},
        "accuracy_panel": {},
        "performance_validation": {"sharpe_by_signal_type": {}, "max_drawdown_by_signal_type": {}},
        "alt_rankings": {"conservative": [], "aggressive": []},
        "data_variation_warning": "",
        "controls": {},
    }


def _save_last_payload(data: dict) -> None:
    try:
        LAST_PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAST_PAYLOAD_PATH.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _load_last_payload() -> dict | None:
    if not LAST_PAYLOAD_PATH.exists():
        return None
    try:
        raw = json.loads(LAST_PAYLOAD_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        return None
    return None


def _run_with_signal_refill(
    *,
    config_path: str,
    confidence_threshold: float,
    risk_tolerance: float,
    disabled_modules: list[str],
    ranking_profile: str,
    include_signals: list[str],
    signal_search_mode: bool,
) -> dict:
    if not signal_search_mode:
        return run_pipeline(
            config_path=config_path,
            confidence_threshold=confidence_threshold,
            risk_tolerance=risk_tolerance,
            disabled_modules=disabled_modules,
            ranking_profile=ranking_profile,
            include_signals=include_signals,
        )

    # Discovery mode: scan top 1,000 most active companies (cached), then rank/filter.
    discovery_universe = _get_discovery_universe_top1000()
    first = run_pipeline(
        config_path=config_path,
        confidence_threshold=0.0,
        risk_tolerance=risk_tolerance,
        disabled_modules=disabled_modules,
        ranking_profile=ranking_profile,
        include_signals=include_signals,
        universe_override=discovery_universe,
    )
    return first


def _load_watchlist() -> list[dict]:
    if not WATCHLIST_PATH.exists():
        return []
    try:
        items = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
        changed = False
        for it in items:
            if "watch_id" not in it:
                it["watch_id"] = uuid4().hex
                changed = True
        if changed:
            _save_watchlist(items)
        return items
    except Exception:
        return []


def _save_watchlist(items: list[dict]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _latest_price(ticker: str, fallback: float) -> float:
    try:
        h = yf.Ticker(ticker).history(period="5d", interval="1d")
        if h.empty:
            return fallback
        return float(h["Close"].dropna().iloc[-1])
    except Exception:
        return fallback


def _fallback_suggestions(q: str) -> list[dict]:
    ql = q.lower()
    try:
        cfg = load_config(DEFAULT_CONFIG)
        universe = [str(t).upper() for t in cfg.data.get("universe", [])]
    except Exception:
        universe = []
    out = []
    for s in universe:
        if ql in s.lower():
            out.append({"symbol": s, "name": s, "type": "fallback"})
    return out[:12]


def _watchlist_view() -> list[dict]:
    out = []
    for row in _load_watchlist():
        watched = float(row.get("watched_price", 0.0))
        current = _latest_price(str(row.get("ticker", "")), watched)
        pct = ((current / watched - 1) * 100) if watched > 0 else 0.0
        out.append(
            {
                "ticker": row.get("ticker"),
                "watch_id": row.get("watch_id"),
                "name": row.get("name", row.get("ticker")),
                "watched_price": watched,
                "current_price": current,
                "pct_change": pct,
                "watched_at": row.get("watched_at"),
            }
        )
    return out


def _render_dashboard(data, config_path, error, controls, ticker_search, trade_message: str | None = None):
    return render_template(
        "dashboard.html",
        data=data,
        config_path=config_path,
        error=error,
        controls=controls,
        ticker_search=ticker_search,
        trade_message=trade_message,
        watchlist=_watchlist_view(),
    )


def _controls_key(
    confidence_threshold: float,
    risk_tolerance: float,
    ranking_profile: str,
    disabled_modules: list[str],
    include_signals: list[str],
) -> str:
    return json.dumps(
        {
            "c": round(confidence_threshold, 4),
            "r": round(risk_tolerance, 4),
            "p": ranking_profile,
            "d": sorted(disabled_modules),
            "s": sorted(include_signals),
        },
        sort_keys=True,
    )


def _resolve_config_path(raw_path: str | None) -> str:
    env_cfg = os.getenv(CONFIG_ENV_VAR, "").strip()
    src = (raw_path or "").strip() or env_cfg or DEFAULT_CONFIG
    # Normalize Windows-style separators for Linux hosts.
    src = src.replace("\\", "/")
    p = Path(src).expanduser()
    if p.is_absolute():
        return str(p)
    # Resolve relative paths against project root first, then current working directory.
    project_resolved = (BASE_DIR / p).resolve()
    if project_resolved.exists():
        return str(project_resolved)
    return str((Path.cwd() / p).resolve())


@app.get("/")
def index():
    global LAST_DATA, LAST_CONTROL_KEY
    controls = {
        "confidence_threshold": 0.5,
        "risk_tolerance": 0.5,
        "ranking_profile": "balanced",
        "disabled_modules": [],
        "include_signals": DEFAULT_INCLUDED_SIGNALS.copy(),
        "signal_search_mode": False,
    }
    try:
        if LAST_DATA is None:
            LAST_DATA = _load_last_payload()
        if LAST_DATA is not None:
            return _render_dashboard(LAST_DATA, DEFAULT_CONFIG, None, controls, "")
        data = _empty_dashboard_data()
        return _render_dashboard(data, DEFAULT_CONFIG, None, controls, "")
    except Exception as exc:
        return _render_dashboard(None, DEFAULT_CONFIG, str(exc), controls, ""), 500


@app.post("/run")
def run_dashboard():
    global LAST_DATA, LAST_CONTROL_KEY
    config_path = _resolve_config_path(request.form.get("config_path", DEFAULT_CONFIG))
    confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
    risk_tolerance = float(request.form.get("risk_tolerance", 0.5))
    ranking_profile = str(request.form.get("ranking_profile", "balanced"))
    disabled_modules = request.form.getlist("disabled_modules")
    include_signals = request.form.getlist("include_signals")
    if not include_signals:
        include_signals = DEFAULT_INCLUDED_SIGNALS.copy()
    ticker_search = str(request.form.get("ticker_search", "")).strip()
    override = [x.strip().upper() for x in ticker_search.replace(";", ",").split(",") if x.strip()]
    signal_search_mode = set(include_signals) != set(DEFAULT_INCLUDED_SIGNALS)
    ctl_key = _controls_key(confidence_threshold, risk_tolerance, ranking_profile, disabled_modules, include_signals)
    try:
        if override and LAST_DATA is not None and LAST_CONTROL_KEY == ctl_key:
            data = dict(LAST_DATA)
        else:
            data = _run_with_signal_refill(
                config_path=config_path,
                confidence_threshold=confidence_threshold,
                risk_tolerance=risk_tolerance,
                disabled_modules=disabled_modules,
                ranking_profile=ranking_profile,
                include_signals=include_signals,
                signal_search_mode=signal_search_mode,
            )
            LAST_DATA = data
            LAST_CONTROL_KEY = ctl_key
            _save_last_payload(data)
        if override:
            searched = run_pipeline(
                config_path=config_path,
                confidence_threshold=confidence_threshold,
                risk_tolerance=risk_tolerance,
                disabled_modules=disabled_modules,
                ranking_profile=ranking_profile,
                universe_override=override,
                include_signals=include_signals,
            )
            data["search_results"] = searched.get("recommendations", [])
        else:
            data["search_results"] = []
            LAST_DATA = data
            _save_last_payload(data)
        return _render_dashboard(
            data,
            config_path,
            None,
            {
                "confidence_threshold": confidence_threshold,
                "risk_tolerance": risk_tolerance,
                "ranking_profile": ranking_profile,
                "disabled_modules": disabled_modules,
                "include_signals": include_signals,
                "signal_search_mode": signal_search_mode,
            },
            ticker_search,
        )
    except Exception as exc:
        return _render_dashboard(
            None,
            config_path,
            str(exc),
            {
                "confidence_threshold": confidence_threshold,
                "risk_tolerance": risk_tolerance,
                "ranking_profile": ranking_profile,
                "disabled_modules": disabled_modules,
                "include_signals": include_signals,
                "signal_search_mode": signal_search_mode,
            },
            ticker_search,
        ), 500


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/run")
def api_run():
    payload = request.json if request.is_json else {}
    config_path = _resolve_config_path(payload.get("config_path", DEFAULT_CONFIG))
    ticker_search = str(payload.get("ticker_search", "")).strip()
    include_signals = list(payload.get("include_signals", DEFAULT_INCLUDED_SIGNALS))
    override = [x.strip().upper() for x in ticker_search.replace(";", ",").split(",") if x.strip()]
    signal_search_mode = set(include_signals) != set(DEFAULT_INCLUDED_SIGNALS)
    data = _run_with_signal_refill(
        config_path=config_path,
        confidence_threshold=float(payload.get("confidence_threshold", 0.5)),
        risk_tolerance=float(payload.get("risk_tolerance", 0.5)),
        disabled_modules=list(payload.get("disabled_modules", [])),
        ranking_profile=str(payload.get("ranking_profile", "balanced")),
        include_signals=include_signals,
        signal_search_mode=signal_search_mode,
    )
    if override:
        searched = run_pipeline(
            config_path=config_path,
            confidence_threshold=float(payload.get("confidence_threshold", 0.5)),
            risk_tolerance=float(payload.get("risk_tolerance", 0.5)),
            disabled_modules=list(payload.get("disabled_modules", [])),
            ranking_profile=str(payload.get("ranking_profile", "balanced")),
            universe_override=override,
            include_signals=include_signals,
        )
        data["search_results"] = searched.get("recommendations", [])
    else:
        data["search_results"] = []
    data["signal_search_mode"] = signal_search_mode
    return jsonify(data)


@app.get("/scan/<ticker>")
def scan_ticker(ticker: str):
    global LAST_DATA, LAST_CONTROL_KEY
    t = ticker.strip().upper()
    include_signals = DEFAULT_INCLUDED_SIGNALS.copy()
    data = run_pipeline(config_path=DEFAULT_CONFIG, universe_override=[t], include_signals=include_signals)
    LAST_DATA = data
    _save_last_payload(data)
    LAST_CONTROL_KEY = _controls_key(0.5, 0.5, "balanced", [], include_signals)
    return _render_dashboard(
        data,
        DEFAULT_CONFIG,
        None,
        {"confidence_threshold": 0.5, "risk_tolerance": 0.5, "ranking_profile": "balanced", "disabled_modules": [], "include_signals": include_signals, "signal_search_mode": False},
        t,
    )


@app.get("/api/latest")
def api_latest():
    snapshot = BASE_DIR / "logs" / "dashboard_snapshot.json"
    if not snapshot.exists():
        return jsonify({"error": "snapshot_not_found"}), 404
    return jsonify(json.loads(snapshot.read_text(encoding="utf-8")))


@app.get("/api/ticker_suggest")
def api_ticker_suggest():
    q = str(request.args.get("q", "")).strip()
    if len(q) < 1:
        return jsonify([])
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": 12, "newsCount": 0},
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        raw = resp.json()
        quotes = raw.get("quotes", [])
        out = []
        for it in quotes:
            sym = str(it.get("symbol", "")).strip()
            if not sym:
                continue
            name = str(it.get("shortname") or it.get("longname") or sym)
            typ = str(it.get("quoteType") or "unknown")
            exch = str(it.get("exchDisp") or "")
            out.append({"symbol": sym, "name": name, "type": typ, "exchange": exch})
        if out:
            return jsonify(out[:12])
    except Exception:
        pass
    return jsonify(_fallback_suggestions(q))


@app.post("/api/refresh_ticker")
def api_refresh_ticker():
    global LAST_DATA
    payload = request.json if request.is_json else {}
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not ticker:
        return jsonify({"ok": False, "message": "Ticker is required."}), 400
    try:
        config_path = _resolve_config_path(payload.get("config_path", DEFAULT_CONFIG))
        confidence_threshold = float(payload.get("confidence_threshold", 0.5))
        risk_tolerance = float(payload.get("risk_tolerance", 0.5))
        ranking_profile = str(payload.get("ranking_profile", "balanced"))
        disabled_modules = list(payload.get("disabled_modules", []))
        include_signals = list(payload.get("include_signals", DEFAULT_INCLUDED_SIGNALS)) or DEFAULT_INCLUDED_SIGNALS

        refreshed = run_pipeline(
            config_path=config_path,
            confidence_threshold=confidence_threshold,
            risk_tolerance=risk_tolerance,
            disabled_modules=disabled_modules,
            ranking_profile=ranking_profile,
            universe_override=[ticker],
            include_signals=include_signals,
        )
        recs = refreshed.get("recommendations", [])
        if not recs:
            return jsonify({"ok": False, "message": f"No refreshed recommendation available for {ticker}."}), 404

        rec = recs[0]
        stale = bool(rec.get("stale_data_used", False))
        if stale:
            age = float(rec.get("stale_age_hours", 0.0))
            msg = (
                f"{ticker} is still using cached data ({age:.1f}h old). "
                "Live source may be rate-limited, unavailable, or market feed unchanged."
            )
        else:
            msg = f"{ticker} refreshed with latest available data."

        # Keep current in-memory payload aligned where possible.
        if LAST_DATA is not None:
            for key in ("recommendations", "search_results"):
                arr = LAST_DATA.get(key) or []
                for i, item in enumerate(arr):
                    if str(item.get("ticker", "")).upper() == ticker:
                        arr[i] = rec
                LAST_DATA[key] = arr

        return jsonify({"ok": True, "ticker": ticker, "stale": stale, "message": msg, "recommendation": rec})
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Refresh failed for {ticker}: {exc}"}), 500


@app.post("/paper_trade")
def paper_trade():
    global LAST_DATA
    ticker = str(request.form.get("ticker", "")).strip().upper()
    side = str(request.form.get("side", "BUY")).strip().upper()
    qty = float(request.form.get("qty", 1))
    controls = {
        "confidence_threshold": float(request.form.get("confidence_threshold", 0.5)),
        "risk_tolerance": float(request.form.get("risk_tolerance", 0.5)),
        "ranking_profile": str(request.form.get("ranking_profile", "balanced")),
        "disabled_modules": request.form.getlist("disabled_modules"),
        "include_signals": request.form.getlist("include_signals") or DEFAULT_INCLUDED_SIGNALS.copy(),
        "signal_search_mode": False,
    }
    ticker_search = str(request.form.get("ticker_search", "")).strip()
    if LAST_DATA is None:
        LAST_DATA = run_pipeline(config_path=DEFAULT_CONFIG, confidence_threshold=controls["confidence_threshold"], risk_tolerance=controls["risk_tolerance"])
    rec = next((r for r in (LAST_DATA.get("recommendations") or []) if r.get("ticker") == ticker), None)
    if rec is None:
        return _render_dashboard(LAST_DATA, DEFAULT_CONFIG, f"Ticker {ticker} not found in current recommendations.", controls, ticker_search), 400
    cfg = load_config(DEFAULT_CONFIG)
    broker = PaperBroker(cfg.execution)
    atr = float(rec.get("module_scores", {}).get("technical", {}).get("metadata", {}).get("atr", rec.get("current_price", 1.0) * 0.01))
    ticket = broker.build_ticket(ticker=ticker, side=side, qty=qty, price=float(rec["current_price"]), atr=atr)
    receipt = broker.submit(ticket, confirm=True)
    return _render_dashboard(LAST_DATA, DEFAULT_CONFIG, None, controls, ticker_search, trade_message=f"Paper order filled: {receipt['side']} {receipt['qty']} {receipt['ticker']} at {receipt['fill_price']:.2f}")


@app.post("/watchlist/add")
def watchlist_add():
    global LAST_DATA
    ticker = str(request.form.get("ticker", "")).strip().upper()
    name = str(request.form.get("name", ticker)).strip()
    watched_price_form = float(request.form.get("watched_price", 0))
    controls = {
        "confidence_threshold": float(request.form.get("confidence_threshold", 0.5)),
        "risk_tolerance": float(request.form.get("risk_tolerance", 0.5)),
        "ranking_profile": str(request.form.get("ranking_profile", "balanced")),
        "disabled_modules": request.form.getlist("disabled_modules"),
        "include_signals": request.form.getlist("include_signals") or DEFAULT_INCLUDED_SIGNALS.copy(),
        "signal_search_mode": False,
    }
    ticker_search = str(request.form.get("ticker_search", "")).strip()
    items = _load_watchlist()
    watched_price = _latest_price(ticker, watched_price_form)
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = False
    for it in items:
        if it.get("ticker") == ticker:
            it["name"] = name
            it["watched_price"] = watched_price
            it["watched_at"] = now_iso
            it["watch_id"] = it.get("watch_id", uuid4().hex)
            updated = True
            break
    if not updated:
        items.append(
            {
                "watch_id": uuid4().hex,
                "ticker": ticker,
                "name": name,
                "watched_price": watched_price,
                "watched_at": now_iso,
            }
        )
    _save_watchlist(items)
    if LAST_DATA is None:
        LAST_DATA = _load_last_payload() or _empty_dashboard_data()
    return _render_dashboard(
        LAST_DATA,
        DEFAULT_CONFIG,
        None,
        controls,
        ticker_search,
        trade_message=f"{ticker} watch price {'updated' if updated else 'added'} at {watched_price:.2f}.",
    )


@app.post("/watchlist/remove")
def watchlist_remove():
    global LAST_DATA
    ticker = str(request.form.get("ticker", "")).strip().upper()
    watch_id = str(request.form.get("watch_id", "")).strip()
    controls = {
        "confidence_threshold": float(request.form.get("confidence_threshold", 0.5)),
        "risk_tolerance": float(request.form.get("risk_tolerance", 0.5)),
        "ranking_profile": str(request.form.get("ranking_profile", "balanced")),
        "disabled_modules": request.form.getlist("disabled_modules"),
        "include_signals": request.form.getlist("include_signals") or DEFAULT_INCLUDED_SIGNALS.copy(),
        "signal_search_mode": False,
    }
    ticker_search = str(request.form.get("ticker_search", "")).strip()
    items = _load_watchlist()
    if watch_id:
        items = [x for x in items if str(x.get("watch_id", "")) != watch_id]
    else:
        items = [x for x in items if x.get("ticker") != ticker]
    _save_watchlist(items)
    if LAST_DATA is None:
        LAST_DATA = _load_last_payload() or _empty_dashboard_data()
    return _render_dashboard(LAST_DATA, DEFAULT_CONFIG, None, controls, ticker_search, trade_message=f"{ticker} removed from watchlist.")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
