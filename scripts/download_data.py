"""Unduh data harga harian saham ASII, BBRI, dan TLKM dari Yahoo Finance ke file Excel.

Script ini berdiri sendiri (tidak butuh paket proyek), jadi bisa juga dijalankan di
Google Colab.

Cara pakai:
    pip install yfinance pandas openpyxl
    python scripts/download_data.py
    python scripts/download_data.py --start 2015-01-01 --out data/data_saham.xlsx

Hasilnya satu file Excel dengan satu sheet per saham (ASII, BBRI, TLKM) berisi kolom
Tanggal, Open, High, Low, Close, Adj Close, Volume, ditambah sheet "Keterangan".
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

SAHAM = {
    "ASII": ("ASII.JK", "Astra International Tbk"),
    "BBRI": ("BBRI.JK", "Bank Rakyat Indonesia (Persero) Tbk"),
    "TLKM": ("TLKM.JK", "Telkom Indonesia (Persero) Tbk"),
}
KOLOM = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def unduh(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    # auto_adjust=False: Close = harga asli seperti di IDX, Adj Close = harga yang sudah
    # disesuaikan dengan dividen dan stock split (dipakai untuk menghitung return).
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if df.empty:
        raise RuntimeError(f"Data {ticker} kosong. Cek koneksi internet atau kode saham.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[KOLOM].dropna()
    df = df[df["Volume"] > 0]  # buang hari libur/suspensi yang tercatat dengan volume 0
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Tanggal"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh data saham ke Excel")
    parser.add_argument("--start", default="2015-01-01", help="tanggal awal (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="tanggal akhir, eksklusif (default: hari ini)")
    parser.add_argument("--out", type=Path, default=Path("data/data_saham.xlsx"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ringkasan = []
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for kode, (ticker, nama) in SAHAM.items():
            print(f"Mengunduh {ticker} ...")
            df = unduh(ticker, args.start, args.end)
            df.to_excel(writer, sheet_name=kode)
            ringkasan.append(
                {
                    "Kode": kode,
                    "Ticker Yahoo": ticker,
                    "Nama": nama,
                    "Tanggal awal": df.index[0].date(),
                    "Tanggal akhir": df.index[-1].date(),
                    "Jumlah hari bursa": len(df),
                }
            )
            print(f"  {len(df)} baris, {df.index[0].date()} s.d. {df.index[-1].date()}")
        info = pd.DataFrame(ringkasan)
        info.to_excel(writer, sheet_name="Keterangan", index=False)
        pd.DataFrame(
            {
                "Catatan": [
                    "Sumber: Yahoo Finance (paket Python yfinance)",
                    f"Diunduh: {datetime.now():%Y-%m-%d %H:%M}",
                    "Close = harga penutupan asli; Adj Close = disesuaikan dividen & stock split",
                ]
            }
        ).to_excel(writer, sheet_name="Keterangan", index=False, startrow=len(info) + 2)
    print(f"Tersimpan di {args.out}")


if __name__ == "__main__":
    main()
