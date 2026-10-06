"""Walk-forward experiment for 2026: train on earlier data, predict every trading day in 2026.

Usage:
    python -m src.run

Writes metrics, per-day predictions, a comparison table and figures to results/.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .backtest import backtest, performance, signal_from_prediction
from .config import BUY_FEE, RESULTS_DIR, RETRAIN_EVERY, SELL_FEE, STOCKS, TEST_START
from .data import add_data_args, load_from_args
from .evaluate import forecast_metrics, walk_forward_predict
from .features import build_dataset
from .models import MODELS, lightgbm
from .plots import plt

# Months with fewer test days than this (for example the current month) are left out of
# the monthly accuracy chart.
MIN_DAYS_PER_MONTH = 10


def run_stock(code: str, prices: pd.DataFrame, test_start: str = TEST_START) -> dict:
    X, y = build_dataset(prices)
    labeled = y.dropna()

    # Row t predicts the next trading day, so the test period is defined by that date.
    next_date = pd.Series(prices.index[1:], index=prices.index[:-1])
    target_date = next_date.reindex(labeled.index)
    n_train = int((target_date < pd.Timestamp(test_start)).sum())
    if n_train < 500 or n_train >= len(labeled):
        raise ValueError(f"{code}: not enough training ({n_train} rows) or test data")

    preds = {m.key: walk_forward_predict(X, y, m.factory, n_train, RETRAIN_EVERY) for m in MODELS}
    idx = preds["lightgbm"].index
    actual = labeled.reindex(idx)
    dates = pd.DatetimeIndex(target_date.reindex(idx).to_numpy(), name="date")

    models, equity = {}, {}
    for m in MODELS:
        bt = backtest(signal_from_prediction(preds[m.key]), actual)
        models[m.key] = {
            "label": m.label,
            "baseline": m.baseline,
            "forecast": forecast_metrics(actual, preds[m.key]),
            "strategy": performance(bt),
        }
        equity[m.key] = bt["equity"].to_numpy()
    bh = backtest(pd.Series(1.0, index=idx), actual)
    equity["buy_hold"] = bh["equity"].to_numpy()

    # Turn return predictions into next-day closing price predictions, using the actual
    # IDX prices (not dividend-adjusted) so the numbers match what investors saw.
    raw = prices["RawClose"]
    prev_close = raw.reindex(idx).to_numpy()
    actual_close = raw.reindex(dates).to_numpy()
    pred_close = prev_close * np.exp(preds["lightgbm"].to_numpy())

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mape(a, b):
        return float(np.mean(np.abs(a - b) / a))

    daily = pd.DataFrame(
        {
            "prev_close": prev_close,
            "actual_close": actual_close,
            "pred_close_lightgbm": pred_close,
            "actual_return": actual.to_numpy(),
            **{f"pred_return_{k}": v.to_numpy() for k, v in preds.items()},
            **{f"equity_{k}": v for k, v in equity.items()},
        },
        index=dates,
    )

    moved = daily["actual_return"] != 0
    hit = ((daily["pred_return_lightgbm"] > 0) == (daily["actual_return"] > 0))[moved]
    by_month = hit.groupby(hit.index.to_period("M"))
    monthly = by_month.mean()[by_month.size() >= MIN_DAYS_PER_MONTH]

    final = lightgbm().fit(X[y.notna()], labeled)
    importance = pd.Series(final.booster_.feature_importance("gain"), index=X.columns)

    return {
        "code": code,
        "company": STOCKS[code][1],
        "n_train": n_train,
        "test_start": f"{dates[0]:%Y-%m-%d}",
        "test_end": f"{dates[-1]:%Y-%m-%d}",
        "n_test": len(idx),
        "models": models,
        "buy_hold": performance(bh),
        "price": {
            "rmse_lightgbm": rmse(actual_close, pred_close),
            "rmse_naive": rmse(actual_close, prev_close),
            "mape_lightgbm": mape(actual_close, pred_close),
            "mape_naive": mape(actual_close, prev_close),
        },
        "monthly_accuracy": {str(k): float(v) for k, v in monthly.items()},
        "feature_importance": (importance / importance.sum()).sort_values(ascending=False).to_dict(),
        "_daily": daily,
    }


def best_baseline(r: dict) -> tuple[str, dict]:
    candidates = [(k, m) for k, m in r["models"].items()
                  if m["baseline"] and not math.isnan(m["forecast"]["directional_accuracy"])]
    return max(candidates, key=lambda km: km[1]["forecast"]["directional_accuracy"])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def _num(x: float, digits: int = 2) -> str:
    return "-" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def summary_table(results: list[dict]) -> str:
    lines = [
        "| Stock | Direction accuracy (LightGBM) | Best baseline | Price RMSE (LightGBM) | Price RMSE (naive) "
        "| Strategy return | Buy & hold return |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lg = r["models"]["lightgbm"]
        _, bb = best_baseline(r)
        lines.append(
            f"| {r['code']} | {_pct(lg['forecast']['directional_accuracy'])} "
            f"| {_pct(bb['forecast']['directional_accuracy'])} ({bb['label']}) "
            f"| Rp{r['price']['rmse_lightgbm']:,.0f} | Rp{r['price']['rmse_naive']:,.0f} "
            f"| {_pct(lg['strategy']['total_return'], sign=True)} | {_pct(r['buy_hold']['total_return'], sign=True)} |"
        )
    return "\n".join(lines) + "\n"


def model_table(r: dict) -> str:
    lines = [
        "| Model | Direction accuracy | R² vs random walk | MAE | Sharpe | Return | Max drawdown | Trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, m in r["models"].items():
        f, s = m["forecast"], m["strategy"]
        name = f"**{m['label']}**" if key == "lightgbm" else m["label"] + (" (baseline)" if m["baseline"] else "")
        lines.append(
            f"| {name} | {_pct(f['directional_accuracy'])} | {_pct(f['r2_vs_random_walk'], 2)} "
            f"| {_pct(f['mae'], 2)} | {_num(s['sharpe'])} | {_pct(s['total_return'], sign=True)} "
            f"| {_pct(s['max_drawdown'])} | {s['trades']} |"
        )
    bh = r["buy_hold"]
    lines.append(
        f"| *Buy & hold* | - | - | - | {_num(bh['sharpe'])} | {_pct(bh['total_return'], sign=True)} "
        f"| {_pct(bh['max_drawdown'])} | 1 |"
    )
    return "\n".join(lines) + "\n"


def comparison_markdown(results: list[dict], synthetic: bool) -> str:
    r0 = results[0]
    parts = []
    if synthetic:
        parts.append("> **Synthetic data.** These numbers come from random-walk prices, not real market data.\n")
    parts += [
        f"Test period: {r0['test_start']} to {r0['test_end']} ({r0['n_test']} trading days). "
        f"Models are retrained every {RETRAIN_EVERY} trading days. "
        f"Fees: {BUY_FEE:.2%} buy, {SELL_FEE:.2%} sell.\n",
        "## Summary\n",
        summary_table(results),
    ]
    for r in results:
        parts += [f"\n## {r['code']} - {r['company']}\n", model_table(r)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_model_comparison(results: list[dict], path: Path) -> None:
    keys = [m.key for m in MODELS if m.key != "random_walk"]
    labels = [results[0]["models"][k]["label"] for k in keys]
    x = np.arange(len(keys))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, (r, color) in enumerate(zip(results, plots.STOCK_COLORS)):
        values = [r["models"][k]["forecast"]["directional_accuracy"] for k in keys]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, values, width=width * 0.9, color=color, label=r["code"])
    ax.axhline(0.5, color=plots.ACTUAL, lw=0.8, ls="--", label="50% = coin flip")
    ax.set_xticks(x, labels)
    all_values = [r["models"][k]["forecast"]["directional_accuracy"] for r in results for k in keys]
    finite = [v for v in all_values if not math.isnan(v)]
    ax.set_ylim(min(0.4, min(finite) - 0.03), max(0.62, max(finite) + 0.05))
    ax.yaxis.set_major_formatter(plots.percent)
    ax.set_title("Direction accuracy in 2026 (out-of-sample)")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predictions(r: dict, path: Path) -> None:
    d = r["_daily"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(d.index, d["actual_close"], color=plots.ACTUAL, lw=1.6, label="Actual close")
    ax1.plot(d.index, d["pred_close_lightgbm"], color=plots.LIGHTGBM, lw=1.0, label="LightGBM prediction")
    ax1.yaxis.set_major_formatter(plots.rupiah)
    ax1.set_title(f"{r['code']} 2026: closing price, actual vs predicted one day ahead")
    ax1.legend(loc="best", ncol=2)
    ax2.bar(d.index, d["actual_return"], color=plots.NEUTRAL_BAR, width=1.0, label="Actual return")
    ax2.plot(d.index, d["pred_return_lightgbm"], color=plots.LIGHTGBM, lw=1.2, label="LightGBM prediction")
    ax2.axhline(0, color=plots.BENCHMARK, lw=0.6)
    ax2.yaxis.set_major_formatter(plots.percent1)
    ax2.set_title("Daily return: actual vs predicted")
    ax2.legend(loc="upper left", ncol=2)
    ax2.xaxis.set_major_formatter(plots.month)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity(results: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(9, 3), sharey=True)
    lines = (("equity_buy_hold", plots.BENCHMARK, "Buy & hold", "--"),
             ("equity_ridge", plots.RIDGE, "Ridge strategy", "-"),
             ("equity_lightgbm", plots.LIGHTGBM, "LightGBM strategy", "-"))
    for ax, r in zip(axes, results):
        d = r["_daily"]
        for col, color, label, ls in lines:
            ax.plot(d.index, d[col], color=color, lw=1.2, ls=ls, label=label)
        ax.axhline(1, color=plots.BENCHMARK, lw=0.6)
        ax.set_title(r["code"])
        ax.xaxis.set_major_locator(plots.mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(plots.month)
    axes[0].yaxis.set_major_formatter(plots.FuncFormatter(lambda v, _p: f"Rp{v:.2f}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Value of Rp1 invested at the start of 2026, after fees", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_monthly_accuracy(results: list[dict], path: Path) -> None:
    months = sorted(set().union(*(r["monthly_accuracy"] for r in results)))
    x = np.arange(len(months))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, (r, color) in enumerate(zip(results, plots.STOCK_COLORS)):
        values = [r["monthly_accuracy"].get(m, np.nan) for m in months]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, values, width=width * 0.9, color=color, label=r["code"])
    ax.axhline(0.5, color=plots.ACTUAL, lw=0.8, ls="--", label="50% = coin flip")
    ax.set_xticks(x, [pd.Period(m).strftime("%b") for m in months])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plots.percent)
    ax.set_title("LightGBM direction accuracy per month, 2026")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(results: list[dict], path: Path, top: int = 8) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(9, 3))
    for ax, r in zip(axes, results):
        imp = pd.Series(r["feature_importance"]).head(top)[::-1]
        ax.barh(imp.index, imp.to_numpy(), color=plots.LIGHTGBM, height=0.65)
        ax.set_title(r["code"])
        ax.xaxis.set_major_formatter(plots.percent)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("LightGBM feature importance (share of total gain)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def save_results(results: list[dict], out: Path, synthetic: bool) -> None:
    figures = out / "figures"
    predictions = out / "predictions"
    figures.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)

    metrics = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic": synthetic,
        "settings": {"test_start": TEST_START, "retrain_every": RETRAIN_EVERY,
                     "buy_fee": BUY_FEE, "sell_fee": SELL_FEE},
        "stocks": {r["code"]: {k: v for k, v in r.items() if not k.startswith("_")} for r in results},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float).replace("NaN", "null"))
    (out / "comparison.md").write_text(comparison_markdown(results, synthetic))
    for r in results:
        r["_daily"].round(6).to_csv(predictions / f"{r['code']}_2026.csv")

    plots.setup()
    plot_model_comparison(results, figures / "model_comparison.png")
    for r in results:
        plot_predictions(r, figures / f"{r['code']}_2026_predictions.png")
    plot_equity(results, figures / "equity_curves.png")
    plot_monthly_accuracy(results, figures / "monthly_accuracy.png")
    plot_feature_importance(results, figures / "feature_importance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(RESULTS_DIR + ("_synthetic" if args.synthetic else ""))

    data = load_from_args(args)
    results = []
    for code, prices in data.items():
        print(f"{code}: {len(prices)} rows, running walk-forward ...", flush=True)
        results.append(run_stock(code, prices))
    save_results(results, out, args.synthetic)
    print()
    print(summary_table(results))
    print(f"Saved metrics, predictions and figures to {out}/")


if __name__ == "__main__":
    main()
