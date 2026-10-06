"""Download daily prices for ASII, BBRI and TLKM from Yahoo Finance into an Excel file.

Usage:
    python -m src.download
    python -m src.download --start 2015-01-01 --end 2026-09-30 --out data/idx_prices.xlsx

The data window is fixed (START_DATE to END_DATE in src/config.py), so the results can be
reproduced. The Excel file has one sheet per stock (ASII, BBRI, TLKM) with the columns Date,
Open, High, Low, Close, Adj Close and Volume, plus an "Info" sheet describing the download.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, END_DATE, START_DATE, STOCKS

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    import yfinance as yf

    # yfinance treats `end` as exclusive, so add one day to include END_DATE itself.
    # auto_adjust=False keeps both Close (the price shown on the IDX) and Adj Close
    # (adjusted for dividends and stock splits, used to compute returns).
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_exclusive, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the internet connection.")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance versions add a ticker level
        df.columns = df.columns.get_level_values(0)
    return df


def save_excel(data: dict[str, pd.DataFrame], path: str | Path = DATA_PATH,
               start: str = START_DATE, end: str = END_DATE) -> None:
    """Save one sheet per stock plus an Info sheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for code, df in data.items():
            df = df[COLUMNS].dropna()
            df = df[df["Volume"] > 0]  # Yahoo lists some holidays and trading halts with zero volume
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            df.index.name = "Date"
            df = df.loc[start:end]
            df.to_excel(writer, sheet_name=code)
            ticker, name = STOCKS[code]
            info.append({
                "Code": code,
                "Yahoo ticker": ticker,
                "Company": name,
                "First date": df.index[0].date(),
                "Last date": df.index[-1].date(),
                "Trading days": len(df),
            })
        info_df = pd.DataFrame(info)
        info_df.to_excel(writer, sheet_name="Info", index=False)
        notes = pd.DataFrame({"Notes": [
            "Source: Yahoo Finance (yfinance Python package)",
            f"Requested period: {start} to {end}",
            f"Downloaded: {datetime.now():%Y-%m-%d %H:%M}",
            "Close = actual closing price; Adj Close = adjusted for dividends and stock splits",
        ]})
        notes.to_excel(writer, sheet_name="Info", index=False, startrow=len(info_df) + 2)
    print(f"Saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=START_DATE, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="last date, included (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, default=Path(DATA_PATH))
    args = parser.parse_args()

    data = {}
    for code, (ticker, _name) in STOCKS.items():
        data[code] = download(ticker, args.start, args.end)
        print(f"{code}: {len(data[code])} rows")
    save_excel(data, args.out, args.start, args.end)


if __name__ == "__main__":
    main()
