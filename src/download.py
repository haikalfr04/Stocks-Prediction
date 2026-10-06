"""Download daily prices for ASII, BBRI and TLKM from Yahoo Finance into an Excel file.

Usage:
    python -m src.download
    python -m src.download --start 2015-01-01 --end 2026-10-06 --out data/idx_prices.xlsx

The Excel file has one sheet per stock (ASII, BBRI, TLKM) with the columns Date, Open, High,
Low, Close, Adj Close and Volume, plus an "Info" sheet describing the download.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, START_DATE, STOCKS

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    # auto_adjust=False keeps both prices: Close is the price shown on the IDX, and
    # Adj Close is adjusted for dividends and stock splits (used to compute returns).
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the internet connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[COLUMNS].dropna()
    df = df[df["Volume"] > 0]  # Yahoo lists some holidays and trading halts with zero volume
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=START_DATE, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="last date, exclusive (default: today)")
    parser.add_argument("--out", type=Path, default=Path(DATA_PATH))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    info = []
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for code, (ticker, name) in STOCKS.items():
            df = download(ticker, args.start, args.end)
            df.to_excel(writer, sheet_name=code)
            info.append({
                "Code": code,
                "Yahoo ticker": ticker,
                "Company": name,
                "First date": df.index[0].date(),
                "Last date": df.index[-1].date(),
                "Trading days": len(df),
            })
            print(f"{code}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        info_df = pd.DataFrame(info)
        info_df.to_excel(writer, sheet_name="Info", index=False)
        notes = pd.DataFrame({"Notes": [
            "Source: Yahoo Finance (yfinance Python package)",
            f"Downloaded: {datetime.now():%Y-%m-%d %H:%M}",
            "Close = actual closing price; Adj Close = adjusted for dividends and stock splits",
        ]})
        notes.to_excel(writer, sheet_name="Info", index=False, startrow=len(info_df) + 2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
