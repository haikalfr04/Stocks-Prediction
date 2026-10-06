"""Predict the next trading day's return and closing price for each stock.

Usage:
    python -m src.predict

LightGBM is trained on all available history and predicts the day after the last date in
the data. Read the 2026 evaluation in results/comparison.md before trusting these numbers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .config import RESULTS_DIR
from .data import add_data_args, load_from_args
from .features import build_dataset
from .models import lightgbm


def forecast(prices) -> dict:
    X, y = build_dataset(prices)
    model = lightgbm().fit(X[y.notna()], y.dropna())
    pred = float(model.predict(X.iloc[[-1]])[0])
    close = float(prices["RawClose"].iloc[-1])  # actual IDX price, not dividend-adjusted
    return {
        "as_of": f"{X.index[-1]:%Y-%m-%d}",
        "last_close": close,
        "predicted_return": math.expm1(pred),
        "predicted_close": close * math.exp(pred),
        "signal": "BUY" if pred > 0 else "CASH",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_args(parser)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(RESULTS_DIR + ("_synthetic" if args.synthetic else ""))
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Stock | Data as of | Last close | Predicted return | Predicted close | Signal |",
        "|---|---|---|---|---|---|",
    ]
    for code, prices in load_from_args(args).items():
        f = forecast(prices)
        lines.append(
            f"| {code} | {f['as_of']} | Rp{f['last_close']:,.0f} | {f['predicted_return']:+.2%} "
            f"| Rp{f['predicted_close']:,.0f} | {f['signal']} |"
        )
    table = "\n".join(lines) + "\n"
    (out / "next_day_forecast.md").write_text(table)
    print(table)
    print("For learning purposes only. This is not investment advice.")


if __name__ == "__main__":
    main()
