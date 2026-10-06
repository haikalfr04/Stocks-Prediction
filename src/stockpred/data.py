"""Price data loading: Yahoo Finance with a local CSV cache, plus a synthetic fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(ticker: str, start: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download daily split/dividend-adjusted OHLCV.

    On success the result is written to ``cache_dir``. If the download fails and a
    cached copy exists, the cached copy is returned instead.
    """
    cache = Path(cache_dir) / f"{ticker}.csv" if cache_dir else None
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = clean_prices(df)
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")
    except Exception:
        if cache is not None and cache.exists():
            return load_csv(cache)
        raise
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_prices(df)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].astype(float).dropna()
    # Yahoo sometimes emits zero-volume rows for IDX holidays and trading halts.
    df = df[df["Volume"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def synthetic_prices(seed: int = 0, n_days: int = 2600, start: str = "2015-01-02") -> pd.DataFrame:
    """Random-walk prices with volatility clustering, for offline tests and demos.

    By construction the returns are unpredictable, so a well-behaved pipeline should
    not find a real edge here.
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
    idx = pd.bdate_range(start, periods=n_days, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )
