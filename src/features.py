"""Feature engineering.

Every feature on row t uses only information available at the close of day t.
The target on row t is the log return from close t to close t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = 5
WINDOWS = (5, 10, 20)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    close, high, low = prices["Close"], prices["High"], prices["Low"]
    ret = log_returns(close)
    f = pd.DataFrame(index=prices.index)

    for k in range(1, LAGS + 1):
        f[f"ret_lag{k}"] = ret.shift(k - 1)  # ret_lag1 is today's return
    for w in WINDOWS:
        f[f"ret_mean_{w}"] = ret.rolling(w).mean()
        f[f"vol_{w}"] = ret.rolling(w).std()

    f["vol_ratio_5_20"] = f["vol_5"] / f["vol_20"]
    f["dist_ma20"] = close / close.rolling(20).mean() - 1
    f["dist_ma50"] = close / close.rolling(50).mean() - 1
    f["rsi_14"] = rsi(close) / 100

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    f["hl_range"] = (high - low) / close
    f["close_in_range"] = (close - low) / (high - low).replace(0, np.nan)
    logvol = np.log(prices["Volume"])
    f["volume_z20"] = (logvol - logvol.rolling(20).mean()) / logvol.rolling(20).std()
    f["day_of_week"] = prices.index.dayofweek
    return f


def make_target(prices: pd.DataFrame) -> pd.Series:
    return log_returns(prices["Close"]).shift(-1).rename("target")


def build_dataset(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned on dates where all features exist.

    The last row has no target (NaN): it is the row used for the next-day forecast.
    """
    X = make_features(prices).replace([np.inf, -np.inf], np.nan).dropna()
    y = make_target(prices).reindex(X.index)
    return X, y
