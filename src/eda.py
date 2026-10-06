"""Data exploration: summary statistics, price history, return distribution, autocorrelation.

Usage:
    python -m src.eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import RESULTS_DIR, STOCKS, TEST_START, TRADING_DAYS
from .data import add_data_args, load_from_args
from .plots import plt

MAX_LAG = 10


def stats(code: str, prices: pd.DataFrame) -> dict:
    close = prices["Close"]
    ret = np.log(close).diff().dropna()
    before = close[close.index < TEST_START]
    in_2026 = close[close.index >= TEST_START]
    return {
        "code": code,
        "company": STOCKS[code][1],
        "first_date": f"{close.index[0]:%Y-%m-%d}",
        "last_date": f"{close.index[-1]:%Y-%m-%d}",
        "trading_days": len(close),
        "last_close": float(close.iloc[-1]),
        "mean_daily_return": float(ret.mean()),
        "annual_volatility": float(ret.std() * np.sqrt(TRADING_DAYS)),
        "share_of_unchanged_days": float((ret == 0).mean()),
        "return_2026": float(in_2026.iloc[-1] / before.iloc[-1] - 1) if len(in_2026) and len(before) else None,
        "autocorrelation": {lag: float(ret.autocorr(lag)) for lag in range(1, MAX_LAG + 1)},
        "autocorr_95pct_band": float(1.96 / np.sqrt(len(ret))),
    }


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Stock | Period | Trading days | Last close | Annual volatility | Unchanged days | Return 2026 | Lag-1 autocorrelation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        r2026 = f"{r['return_2026']:+.1%}" if r["return_2026"] is not None else "-"
        lines.append(
            f"| {r['code']} | {r['first_date']} to {r['last_date']} | {r['trading_days']:,} "
            f"| Rp{r['last_close']:,.0f} | {r['annual_volatility']:.1%} | {r['share_of_unchanged_days']:.1%} "
            f"| {r2026} | {r['autocorrelation'][1]:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot_prices(data: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(len(data), 1, figsize=(9, 7), sharex=True)
    for ax, (code, prices), color in zip(axes, data.items(), plots.STOCK_COLORS):
        close = prices["Close"]
        ax.plot(close.index, close.to_numpy(), color=color, lw=1.1)
        ax.axvspan(pd.Timestamp(TEST_START), close.index[-1], color=plots.TEST_SHADE, alpha=0.08, lw=0)
        ax.set_title(f"{code} - {STOCKS[code][1]}")
        ax.yaxis.set_major_formatter(plots.rupiah)
    axes[0].text(pd.Timestamp(TEST_START), 1.0, " test period (2026)", va="top", fontsize=8,
                 color=plots.ACTUAL, transform=axes[0].get_xaxis_transform())
    fig.suptitle("Adjusted closing price", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_return_distribution(data: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(9, 2.8), sharey=True)
    for ax, (code, prices), color in zip(axes, data.items(), plots.STOCK_COLORS):
        ret = np.log(prices["Close"]).diff().dropna()
        ax.hist(ret.clip(-0.1, 0.1), bins=80, color=color)
        ax.set_title(code)
        ax.xaxis.set_major_formatter(plots.percent)
    axes[0].set_ylabel("Number of days")
    fig.suptitle("Distribution of daily log returns (clipped at ±10%)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_autocorrelation(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    lags = np.arange(1, MAX_LAG + 1)
    width = 0.8 / len(rows)
    for i, (r, color) in enumerate(zip(rows, plots.STOCK_COLORS)):
        values = [r["autocorrelation"][lag] for lag in lags]
        ax.bar(lags + (i - (len(rows) - 1) / 2) * width, values, width=width * 0.9, color=color, label=r["code"])
    band = max(r["autocorr_95pct_band"] for r in rows)
    ax.axhspan(-band, band, color=plots.BENCHMARK, alpha=0.15, lw=0, label="95% band for pure noise")
    ax.axhline(0, color=plots.ACTUAL, lw=0.6)
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (trading days)")
    ax.set_title("Autocorrelation of daily returns", pad=24)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(RESULTS_DIR + ("_synthetic" if args.synthetic else ""))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plots.setup()

    data = load_from_args(args)
    rows = [stats(code, prices) for code, prices in data.items()]
    (out / "eda_stats.json").write_text(json.dumps(rows, indent=2))
    (out / "eda_stats.md").write_text(to_markdown(rows))
    plot_prices(data, out / "figures" / "price_history.png")
    plot_return_distribution(data, out / "figures" / "return_distribution.png")
    plot_autocorrelation(rows, out / "figures" / "autocorrelation.png")
    print(to_markdown(rows))
    print(f"Saved statistics and figures to {out}/")


if __name__ == "__main__":
    main()
