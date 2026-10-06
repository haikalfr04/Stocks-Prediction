"""Long-or-cash backtest with IDX transaction costs (no short selling)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BUY_FEE, SELL_FEE, TRADING_DAYS


def backtest(
    position: pd.Series,
    next_log_return: pd.Series,
    buy_fee: float = BUY_FEE,
    sell_fee: float = SELL_FEE,
) -> pd.DataFrame:
    """Simulate holding ``position`` (0 or 1) from close t to close t+1.

    Fees are charged on the day the position changes.
    """
    position, next_log_return = position.align(next_log_return, join="inner")
    position = position.astype(float)
    prev = position.shift(1, fill_value=0.0)
    cost = np.where(position > prev, buy_fee, 0.0) + np.where(position < prev, sell_fee, 0.0)
    strat = position * np.expm1(next_log_return) - cost
    equity = (1 + strat).cumprod()
    return pd.DataFrame({"position": position, "return": strat, "equity": equity})


def signal_from_prediction(pred: pd.Series, threshold: float = 0.0) -> pd.Series:
    return (pred > threshold).astype(float)


def shift_test_pvalue(position: pd.Series, next_log_return: pd.Series) -> float:
    """How often a strategy with the same timing pattern but shifted dates does as well.

    The position series is shifted in a circle by every possible number of days. Each
    shifted copy is in the market on the same number of days and trades about as often,
    but its timing is unrelated to the model's predictions. The result is the share of
    shifted copies whose total return is at least the original's: a small value means the
    original timing is unlikely to be luck.
    """
    position, next_log_return = position.align(next_log_return, join="inner")
    observed = performance(backtest(position, next_log_return))["total_return"]
    values = position.to_numpy()
    shifted_returns = [
        performance(backtest(pd.Series(np.roll(values, k), index=position.index), next_log_return))["total_return"]
        for k in range(1, len(values))
    ]
    return float(np.mean(np.array(shifted_returns) >= observed - 1e-12))


def performance(bt: pd.DataFrame) -> dict[str, float]:
    r, equity = bt["return"], bt["equity"]
    years = len(r) / TRADING_DAYS
    std = r.std()
    entries = int(((bt["position"] > 0) & (bt["position"].shift(1, fill_value=0) == 0)).sum())
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "annual_volatility": float(std * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan"),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "exposure": float(bt["position"].mean()),
        "trades": entries,
    }
