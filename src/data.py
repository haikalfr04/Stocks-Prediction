"""Loads the price Excel file created by src/download.py, plus synthetic prices for testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, STOCKS

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_prices(path: str | Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Read one sheet per stock and return dividend/split-adjusted OHLCV prices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download` first.")
    sheets = pd.read_excel(path, sheet_name=list(STOCKS), index_col=0, parse_dates=True)
    return {code: clean_prices(df) for code, df in sheets.items()}


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Adj Close" in df.columns:
        # Scale Open/High/Low/Close by the Adj Close factor so that returns do not show
        # an artificial drop on ex-dividend dates.
        factor = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
    df = df[COLUMNS].astype(float).dropna()
    df = df[df["Volume"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def synthetic_prices(seed: int = 0, end: str = "2026-10-02", n_days: int = 2900) -> pd.DataFrame:
    """Random-walk prices with volatility clustering, for tests and offline dry runs.

    The returns are random by construction, so a correct pipeline should find no edge here.
    """
    rng = np.random.default_rng(seed)
    vol = np.empty(n_days)
    ret = np.empty(n_days)
    vol[0] = 0.018
    for t in range(n_days):
        if t:
            vol[t] = np.sqrt(0.000004 + 0.9 * vol[t - 1] ** 2 + 0.08 * ret[t - 1] ** 2)
        ret[t] = 0.0003 + vol[t] * rng.standard_normal()
    close = 4000 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.008, n_days))
    open_ = close * np.exp(rng.normal(0, 0.004, n_days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(17, 0.4, n_days) * (1 + 20 * np.abs(ret))
    idx = pd.bdate_range(end=end, periods=n_days, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=Path(DATA_PATH), help="Excel file with prices")
    parser.add_argument("--synthetic", action="store_true",
                        help="use synthetic random-walk prices (dry run without internet)")


def load_from_args(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.synthetic:
        return {code: synthetic_prices(seed=i) for i, code in enumerate(STOCKS)}
    return load_prices(args.data)
