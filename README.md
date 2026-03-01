# TRADING PF

Production-grade, modular, institutional-style quantitative intelligence and trading research platform in Python.

## Scope

This project implements a research-driven decision engine with controlled execution gates:

- Cross-asset macro intelligence
- Institutional flow proxies (options + dark pool + liquidity)
- AI news clustering and catalyst scoring
- Volatility mispricing research
- Statistical arbitrage signal engine
- Market regime + black swan anomaly monitoring
- RL-style allocation research (research mode)
- Risk-first portfolio command center
- Compliant broker execution interface (manual confirmation supported)
- Walk-forward backtesting and performance metrics
- Continuous retraining decision workflow
- Dashboard snapshot generation

## Project Structure

```
TRADING_PF/
  core/
  data/
  technical/
  ml/
  news/
  options/
  darkpool/
  liquidity/
  macro/
  volatility/
  pairs/
  geo/
  rl/
  risk/
  execution/
  dashboard/
  backtest/
  config/
  logs/
  main.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --config config/default.yaml
```

## Run In Browser

```bash
python -m pip install -r requirements.txt
python web_app.py
```

Then open:

`http://127.0.0.1:5000`

Use the `Refresh` button to execute the full engine and view results in-browser.
The dashboard now embeds the launch command and runs with built-in default config (no full config path typing required).
You can search/scan custom tickers directly from the dashboard and click any ticker in results to open a focused scan view.
Paper trading can be submitted directly from the web dashboard on BUY candidates.

Default config uses live market/macro ingestion:

- Market/Intraday/Options: Yahoo Finance
- Macro (Fed Funds, CPI, GDP): FRED public series endpoints
- Automatic fallback: deterministic mock feed if a specific symbol/feed is temporarily unavailable

## Deploy Online (Render)

This repository is now deployment-ready with:

- `wsgi.py`
- `Procfile`
- `render.yaml`
- `Dockerfile`

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Deploy TRADING PF web dashboard"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 2. Create Render Web Service

1. Go to Render dashboard.
2. Click `New` -> `Web Service`.
3. Connect your GitHub repo.
4. Render should auto-detect `render.yaml`.
5. Deploy.

If you configure manually, use:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn wsgi:app --workers 2 --threads 8 --timeout 240 --bind 0.0.0.0:$PORT`

### 3. Open your public URL

Render provides a URL like:

`https://trading-pf.onrender.com`

Open it in browser and click `Refresh`.

### 4. Optional custom domain

In Render service settings:

1. Add custom domain
2. Add DNS records at your domain registrar
3. Wait for SSL provisioning

## Deploy With Docker (Any Host)

Build and run locally:

```bash
docker build -t trading-pf .
docker run -p 5000:5000 trading-pf
```

Open:

`http://127.0.0.1:5000`

## Config Path Notes

- Default config is `config/default.yaml`.
- For cloud/runtime override, set env var:

`TRADING_PF_CONFIG=config/default.yaml`

- Windows-style paths are normalized automatically in the app (fixes `config\default.yaml` issues on Linux hosts).

## Outputs

Generated under `logs/`:

- `system.log`: runtime logs
- `predictions.jsonl`: model recommendations and confidence
- `premarket_scans.jsonl`: tactical scanner outputs
- `orders.jsonl`: execution audit trail
- `outcomes.jsonl`: prediction outcome tracking
- `retraining_events.jsonl`: retraining decision logs
- `dashboard_snapshot.json`: consolidated dashboard-ready payload
- `data_store/`: persisted validated datasets and features

## Production Notes

- Thread-safe module execution via lock-protected modules and thread pool orchestration
- Config-driven behavior via `config/default.yaml`
- Risk validation runs before execution routing
- Execution path enforces broker-level compliance toggles
- All recommendations include confidence and risk fields
- Research modules are intentionally decoupled for easy replacement with live institutional feeds
- Local files in `logs/` and `data_store/` are ephemeral on many cloud free tiers; for permanent history/watchlist, move to a database.

## API Endpoints

- `GET /health`
- `POST /run` (HTML form execution)
- `POST /api/run` (JSON execution payload)
- `GET /api/latest` (latest dashboard snapshot)

## Dashboard Enhancements

- Color-coded buy recommendation bands (blue -> green)
- Standardized glossary term pop-up boxes for key institutional metrics
- Top 20 recommendations with market category labels (e.g., Military, Technology, Finance)
- Plain-English "Why this trade?" explanations
- Ticker search, clickable ticker drilldowns, and in-dashboard paper-trade buttons

## Tests

```bash
python -m pytest -q
```
