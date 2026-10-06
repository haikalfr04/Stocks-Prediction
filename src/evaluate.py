"""Walk-forward validation and forecast metrics."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd


def walk_forward_splits(n: int, min_train: int, step: int) -> Iterator[tuple[int, int, int]]:
    """Yield (train_end, test_start, test_end): train on [0, train_end), test on the block.

    Training rows always come strictly before test rows. Row i's target is the return to
    day i+1, which is known by the close of test_start, so no gap is needed.
    """
    for start in range(min_train, n, step):
        yield start, start, min(start + step, n)


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    factory: Callable[[], object],
    min_train: int,
    step: int,
) -> pd.Series:
    """Out-of-sample predictions for every labeled row after the first training window."""
    labeled = y.notna()
    Xl, yl = X[labeled], y[labeled]
    preds = np.full(len(Xl), np.nan)
    for train_end, test_start, test_end in walk_forward_splits(len(Xl), min_train, step):
        model = factory()
        model.fit(Xl.iloc[:train_end], yl.iloc[:train_end])
        preds[test_start:test_end] = model.predict(Xl.iloc[test_start:test_end])
    return pd.Series(preds, index=Xl.index, name="pred").dropna()


def forecast_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    actual, pred = actual.align(pred, join="inner")
    err = actual - pred
    sse_zero = float((actual**2).sum())
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        # Out-of-sample R^2 against always predicting 0% (the random walk).
        "r2_vs_random_walk": 1 - float((err**2).sum()) / sse_zero if sse_zero else float("nan"),
        "directional_accuracy": directional_accuracy(actual, pred),
        "information_coefficient": float(pred.corr(actual, method="spearman"))
        if pred.nunique() > 1
        else float("nan"),
        "n_predictions": int(len(actual)),
    }


def directional_accuracy(actual: pd.Series, pred: pd.Series) -> float:
    """Share of days on which the predicted direction was right.

    Days with a zero return (unchanged close) have no direction and are skipped. A model
    that never predicts a direction (always 0) gets no score.
    """
    actual, pred = actual.align(pred, join="inner")
    if (pred == 0).all():
        return float("nan")
    mask = actual != 0
    return float(((pred[mask] > 0) == (actual[mask] > 0)).mean())


def rolling_hit_rate(actual: pd.Series, pred: pd.Series, window: int = 63) -> pd.Series:
    actual, pred = actual.align(pred, join="inner")
    hits = ((pred > 0) == (actual > 0)).astype(float)
    return hits.rolling(window).mean().dropna()
