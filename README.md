# Project Trident Research Engine

Public snapshot of the Project Trident crypto research engine, taken before the dedicated backtesting work was added.

## What This Repo Does

- Pulls crypto market and context data from public and key-based APIs.
- Builds deterministic technical, fundamental, news, macro, calendar, and sentiment features.
- Produces `decision:trident` outputs for symbol-level market scenarios.
- Stores research state in local DuckDB tables for repeatable analysis.
- Includes semi-simulated paper-trading workflows for research and operator review.
- Supports optional GPT-assisted narrative fields while keeping the core feature and decision path deterministic.

## Financial Disclaimer

Project Trident is a research engine only. It does not provide financial advice, investment advice, trading advice, or buy/sell recommendations. The decision outputs, feature scores, and semi-simulated paper workflows are experimental research artifacts and should not be used as the sole basis for financial decisions. Cryptocurrency markets are volatile, and any trading or investment decision is your responsibility.

## What Is Intentionally Excluded

- Dedicated backtesting modules and historical backtest output.
- Runtime databases, writer locks, and generated `var/` artifacts.
- Real API keys or private environment files.
- Live brokerage or exchange execution.
- Exact exchange-grade fill modeling.

## API Requirements

| API | Env Var | Required? | Used For |
| --- | --- | --- | --- |
| Kraken public REST | `KRAKEN_BASE` | Required | Crypto OHLCV candles and market history. No API key required. |
| CoinGecko public API | `COINGECKO_BASE` | Required | Crypto fundamentals, supply, market cap, volume, and ranking data. No API key required for the public endpoint. |
| NewsAPI.ai / Event Registry | `NEWSAPI_AI_KEY` | Optional | Crypto, macro, geopolitical, and policy news ingestion. Needed for full news features. |
| FRED | `FRED_API_KEY` | Optional | Fed liquidity and macro series. Needed for macro liquidity context. |
| Finnhub | `FINNHUB_API_KEY` | Optional | Economic calendar events. Needed for calendar context. |
| Alternative.me Fear & Greed | none | Optional | Crypto sentiment index. No API key required. |
| OpenAI | `OPENAI_API_KEY` | Optional | Scenario explanations and narrative fields when GPT output is enabled. |

The minimum useful setup is Kraken plus CoinGecko endpoint access. Optional keys deepen the research context but are not required for basic market/fundamental feature generation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the API keys you want to enable, then initialize the database:

```bash
python backend/cli.py trident:doctor
python backend/cli.py trident:init
```

Legacy aliases are still available as `env:check` and `db:migrate`.

## Research Workflow

Pull the full available data stack for one symbol. Missing optional API keys will skip or fail those enrichment steps unless you run with `--strict`.

```bash
python backend/cli.py trident:prime --symbol BTC-USD
```

Legacy alias: `backfill:all`.

Run individual data pulls when you only need one layer:

```bash
python backend/cli.py crypto:backfill --symbol BTC-USD --interval 1h --lookback 2000
python backend/cli.py fundamentals:pull --symbols BTC-USD
python backend/cli.py news:pull --symbols BTC-USD
python backend/cli.py fed:pull
python backend/cli.py sentiment:pull
python backend/cli.py calendar:pull
```

Inspect feature outputs:

```bash
python backend/cli.py tech:features --symbol BTC-USD
python backend/cli.py fundamentals:features --symbol BTC-USD
python backend/cli.py news:features --symbol BTC-USD
python backend/cli.py all:features --symbol BTC-USD
```

Generate the Trident decision payload:

```bash
python backend/cli.py decision:trident --symbol BTC-USD --interval 1h
```

Enable GPT narrative fields only when desired:

```bash
TRIDENT_USE_GPT=true python backend/cli.py decision:trident --symbol BTC-USD --interval 1h
```

## Minimum Viable Run

This path uses the required public Kraken and CoinGecko endpoints only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python backend/cli.py trident:init
python backend/cli.py crypto:backfill --symbol BTC-USD --interval 1h --lookback 2000
python backend/cli.py fundamentals:pull --symbols BTC-USD
python backend/cli.py all:features --symbol BTC-USD
python backend/cli.py decision:trident --symbol BTC-USD --interval 1h
```

## Semi-Simulated Paper Workflow

The `paper:*` commands provide a research workflow around the decision engine. They simulate entries, exits, fees, slippage, and replay checks using local market data, but they are not an exact exchange simulator and should not be treated as production execution logic.

Initialize the local paper state:

```bash
python backend/cli.py paper:init --reset
```

Run the paper workflow in preview mode:

```bash
python backend/cli.py paper:run --dry-run
```

Run the semi-simulated paper workflow:

```bash
python backend/cli.py paper:run
```

Mark open positions and replay missed exits from candle data:

```bash
python backend/cli.py paper:mark
```

Review paper state:

```bash
python backend/cli.py paper:report --daily
```

## Determinism Notes

- Core features and decision gates are deterministic.
- GPT output is optional and can be disabled with `TRIDENT_USE_GPT=false`.
- Local database files and generated artifacts are ignored by git.
- Re-running with the same database, config, and timestamps is intended to produce repeatable research outputs.

## Validation

```bash
PYTHONPATH=. python -m backend.tests.test_phase5_decision
PYTHONPATH=. python -m backend.tests.test_phase4_confidence
PYTHONPATH=. python -m backend.tests.test_phase4_horizon_alignment
```
