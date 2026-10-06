# IDX Next-Day Return Prediction

Predicting tomorrow's return for **ASII** (Astra International), **BBRI** (Bank Rakyat Indonesia) and
**TLKM** (Telkom Indonesia), and testing whether the predictions would beat simply holding the stock.

**Live dashboard:** https://haikalfr04.github.io/Stocks-Prediction/

The dashboard is a static site on GitHub Pages. A GitHub Actions job runs every weekday after
the IDX close: it downloads fresh prices, retrains the models, regenerates the results as JSON
and redeploys the site. No server is needed.

## Why this project is set up the way it is

Many stock-prediction projects report impressive results that come from avoidable mistakes.
This project is built to avoid them:

| Common mistake | What this project does instead |
|---|---|
| Predicting the price level, which looks great because tomorrow's price ≈ today's price | Predicts the **next-day log return** |
| Random train/test split or scaling fit on all data (leakage) | **Walk-forward validation**: train on the past, predict the next month, retrain, repeat. A test checks that features never use future data |
| No baseline | Compares against **random walk, historical mean and momentum** baselines |
| Accuracy only | **Backtest** with IDX fees (0.15% buy, 0.25% sell, long-or-cash), compared with buy & hold |

The honest result is that daily returns are very hard to predict. That finding is reported
directly instead of being hidden.

## Pipeline

```
Yahoo Finance ──► features.py ──► walk-forward (evaluate.py) ──► backtest.py ──► site/data/*.json ──► GitHub Pages
                  lags, volatility,   Random walk / Hist. mean /     long-or-cash      static dashboard
                  RSI, MACD, volume   Momentum / Ridge / LightGBM    with fees         (Plotly)
```

| Module | Purpose |
|---|---|
| `src/stockpred/data.py` | Downloads adjusted OHLCV with a CSV cache; synthetic prices for offline runs |
| `src/stockpred/features.py` | Feature engineering (each feature uses only data up to day *t*) and the target |
| `src/stockpred/models.py` | Baselines, Ridge and LightGBM |
| `src/stockpred/evaluate.py` | Walk-forward splits, out-of-sample metrics (directional accuracy, R² vs random walk, IC) |
| `src/stockpred/backtest.py` | Long-or-cash backtest with fees, Sharpe ratio, drawdown |
| `src/stockpred/pipeline.py` | Runs everything and writes JSON for the site |
| `site/` | Static dashboard (HTML, CSS, JS) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

python -m stockpred.pipeline --out site/data             # real data from Yahoo Finance
python -m stockpred.pipeline --out site/data --synthetic # offline demo data

python -m http.server -d site 8000                       # open http://localhost:8000
```

## Deploying to GitHub Pages

1. In the repository go to **Settings → Pages** and set **Source** to **GitHub Actions**.
2. Push to `main`. The `Update predictions and deploy site` workflow tests, runs the pipeline
   and deploys. After that it runs automatically every weekday, and you can also start it
   manually from the **Actions** tab.

## Possible extensions

- Add an LSTM or Transformer model as a deep-learning comparison
- Add news sentiment features (IndoBERT)
- Predict volatility, which is more predictable than direction
- Use quantile regression for prediction intervals

---

For education and portfolio purposes only. This is not investment advice.
