from __future__ import annotations

from pathlib import Path
import sys
import json
import os
from datetime import datetime, timezone
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
ALL_SIGNALS = ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"]
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
DISCOVERY_UNIVERSE_EXTENDED = DISCOVERY_UNIVERSE + [
    "COST","WMT","HD","LOW","MCD","SBUX","KO","PEP","NKE","DIS","CMCSA","T","VZ","TMUS",
    "SHOP","UBER","ABNB","PYPL","INTU","NOW","SNOW","PLTR","PANW","CRWD","ZS",
    "MRK","ABBV","BMY","AMGN","GILD","ISRG","DHR","TMO","MDT","SYK",
    "AXP","BK","C","USB","PNC","COF","SPGI","ICE","CME",
    "EOG","MPC","PSX","VLO","OXY","KMI","ET","ENB",
    "DAL","UAL","AAL","LUV","UPS","FDX","CSX","NSC","UNP",
    "BABA","PDD","TSM","ASML","SAP","SHEL","BP","RDSA.AS","EEM","EFA",
]


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

    # Fast refill mode: single discovery pass (main pipeline now backfills nearest signals).
    first = run_pipeline(
        config_path=config_path,
        confidence_threshold=0.0,
        risk_tolerance=risk_tolerance,
        disabled_modules=disabled_modules,
        ranking_profile=ranking_profile,
        include_signals=include_signals,
        universe_override=DISCOVERY_UNIVERSE,
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
        "include_signals": ALL_SIGNALS.copy(),
        "signal_search_mode": False,
    }
    try:
        data = run_pipeline(
            config_path=DEFAULT_CONFIG,
            confidence_threshold=0.5,
            risk_tolerance=0.5,
            include_signals=controls["include_signals"],
        )
        LAST_DATA = data
        LAST_CONTROL_KEY = _controls_key(
            controls["confidence_threshold"],
            controls["risk_tolerance"],
            controls["ranking_profile"],
            controls["disabled_modules"],
            controls["include_signals"],
        )
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
        include_signals = ALL_SIGNALS.copy()
    ticker_search = str(request.form.get("ticker_search", "")).strip()
    override = [x.strip().upper() for x in ticker_search.replace(";", ",").split(",") if x.strip()]
    signal_search_mode = set(include_signals) != set(ALL_SIGNALS)
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
    include_signals = list(payload.get("include_signals", ALL_SIGNALS))
    override = [x.strip().upper() for x in ticker_search.replace(";", ",").split(",") if x.strip()]
    signal_search_mode = set(include_signals) != set(ALL_SIGNALS)
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
    include_signals = ALL_SIGNALS.copy()
    data = run_pipeline(config_path=DEFAULT_CONFIG, universe_override=[t], include_signals=include_signals)
    LAST_DATA = data
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
        "include_signals": request.form.getlist("include_signals") or ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"],
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
        "include_signals": request.form.getlist("include_signals") or ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"],
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
        LAST_DATA = run_pipeline(config_path=DEFAULT_CONFIG)
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
        "include_signals": request.form.getlist("include_signals") or ["Strong Buy", "Buy", "Accumulate", "Watch", "Avoid"],
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
        LAST_DATA = run_pipeline(config_path=DEFAULT_CONFIG)
    return _render_dashboard(LAST_DATA, DEFAULT_CONFIG, None, controls, ticker_search, trade_message=f"{ticker} removed from watchlist.")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
