"""End-to-end run: data -> features -> walk-forward models -> backtest -> JSON for the site.

Usage:
    python -m stockpred.pipeline --out site/data
    python -m stockpred.pipeline --out site/data --synthetic   # offline demo data
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .backtest import backtest, performance, signal_from_prediction
from .data import download_prices, synthetic_prices
from .evaluate import forecast_metrics, rolling_hit_rate, walk_forward_predict
from .features import build_dataset
from .models import MODELS, lightgbm

RECENT_DAYS = 120


def _clean(obj):
    """Make an object JSON-safe: NaN/inf -> null, floats rounded to keep files small."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _dates(index: pd.Index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def analyze_ticker(ticker: str, name: str, prices: pd.DataFrame) -> dict:
    X, y = build_dataset(prices)
    labeled_y = y.dropna()

    predictions: dict[str, pd.Series] = {}
    results: dict[str, dict] = {}
    for spec in MODELS:
        pred = walk_forward_predict(X, y, spec.factory, config.MIN_TRAIN_DAYS, config.RETRAIN_EVERY)
        predictions[spec.key] = pred
        bt = backtest(signal_from_prediction(pred), labeled_y)
        results[spec.key] = {
            "label": spec.label,
            "baseline": spec.baseline,
            "forecast": forecast_metrics(labeled_y, pred),
            "strategy": performance(bt),
            "equity": bt["equity"],
        }

    test_index = predictions["lightgbm"].index
    actual = labeled_y.reindex(test_index)
    buy_hold = backtest(pd.Series(1.0, index=test_index), actual)

    # Live forecast: fit on all labeled history, predict the latest (unlabeled) row.
    final = lightgbm().fit(X[y.notna()], labeled_y)
    latest_pred = float(final.predict(X.iloc[[-1]])[0])
    importance = (
        pd.Series(final.booster_.feature_importance("gain"), index=X.columns)
        .sort_values(ascending=False)
        .head(12)
    )

    recent = test_index[-RECENT_DAYS:]
    hit = rolling_hit_rate(actual, predictions["lightgbm"])

    return {
        "ticker": ticker,
        "symbol": ticker.removesuffix(".JK"),
        "name": name,
        "data_start": prices.index[0].strftime("%Y-%m-%d"),
        "data_end": prices.index[-1].strftime("%Y-%m-%d"),
        "test_start": test_index[0].strftime("%Y-%m-%d"),
        "n_test_days": len(test_index),
        "last_close": float(prices["Close"].iloc[-1]),
        "prices": {"dates": _dates(prices.index), "close": prices["Close"].tolist()},
        "models": [
            {"key": k, **{f: v for f, v in r.items() if f != "equity"}} for k, r in results.items()
        ],
        "buy_hold": performance(buy_hold),
        "equity": {
            "dates": _dates(test_index),
            "buy_hold": buy_hold["equity"].tolist(),
            **{k: results[k]["equity"].reindex(test_index).tolist() for k in ("lightgbm", "ridge")},
        },
        "recent": {
            "dates": _dates(recent),
            "actual": actual.reindex(recent).tolist(),
            "lightgbm": predictions["lightgbm"].reindex(recent).tolist(),
            "ridge": predictions["ridge"].reindex(recent).tolist(),
        },
        "hit_rate": {"window": 63, "dates": _dates(hit.index), "lightgbm": hit.tolist()},
        "feature_importance": [
            {"feature": f, "gain": float(g / importance.sum())} for f, g in importance.items()
        ],
        "forecast": {
            "as_of": X.index[-1].strftime("%Y-%m-%d"),
            "model": "lightgbm",
            "predicted_log_return": latest_pred,
            "signal": "LONG" if latest_pred > 0 else "CASH",
        },
    }


def run(out_dir: Path, synthetic: bool = False, cache_dir: Path | None = Path("data/raw")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic": synthetic,
        "settings": {
            "min_train_days": config.MIN_TRAIN_DAYS,
            "retrain_every": config.RETRAIN_EVERY,
            "buy_fee": config.BUY_FEE,
            "sell_fee": config.SELL_FEE,
        },
        "tickers": [],
    }
    for i, (ticker, name) in enumerate(config.TICKERS.items()):
        print(f"[{ticker}] loading prices", flush=True)
        prices = (
            synthetic_prices(seed=i)
            if synthetic
            else download_prices(ticker, config.START_DATE, cache_dir)
        )
        print(f"[{ticker}] {len(prices)} rows, running walk-forward", flush=True)
        result = analyze_ticker(ticker, name, prices)
        symbol = result["symbol"]
        (out_dir / f"{symbol}.json").write_text(json.dumps(_clean(result), separators=(",", ":")))
        lgbm = next(m for m in result["models"] if m["key"] == "lightgbm")
        summary["tickers"].append(
            {
                "symbol": symbol,
                "name": name,
                "data_end": result["data_end"],
                "forecast": result["forecast"],
                "lightgbm_directional_accuracy": lgbm["forecast"]["directional_accuracy"],
                "lightgbm_sharpe": lgbm["strategy"]["sharpe"],
                "buy_hold_sharpe": result["buy_hold"]["sharpe"],
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    print(f"Wrote results to {out_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("site/data"))
    parser.add_argument("--synthetic", action="store_true", help="use synthetic prices (offline)")
    args = parser.parse_args()
    run(args.out, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
