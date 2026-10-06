"""Membaca data harga dari file Excel hasil scripts/download_data.py, plus data sintetis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_excel(path: Path, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Baca satu sheet per kode saham dan kembalikan harga OHLCV yang sudah disesuaikan."""
    sheets = pd.read_excel(path, sheet_name=codes, index_col=0, parse_dates=True)
    return {code: clean_prices(df) for code, df in sheets.items()}


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Adj Close" in df.columns:
        # Sesuaikan Open/High/Low/Close dengan faktor dividen & split dari Adj Close,
        # supaya return tidak "loncat" pada tanggal cum-dividen.
        factor = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * factor
    df = df[COLUMNS].astype(float).dropna()
    df = df[df["Volume"] > 0]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Tanggal"
    return df.sort_index()


def synthetic_prices(seed: int = 0, end: str = "2026-10-02", n_days: int = 2900) -> pd.DataFrame:
    """Harga random walk dengan volatilitas berkelompok, untuk uji coba tanpa internet.

    Return-nya acak, jadi pipeline yang benar seharusnya TIDAK menemukan pola di sini.
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
    idx = pd.bdate_range(end=end, periods=n_days, name="Tanggal")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )
