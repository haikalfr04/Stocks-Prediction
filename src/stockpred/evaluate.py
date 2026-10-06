"""Validasi walk-forward dan metrik akurasi prediksi."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd


def walk_forward_splits(n: int, min_train: int, step: int) -> Iterator[tuple[int, int, int]]:
    """Hasilkan (train_end, test_start, test_end): latih pada [0, train_end), uji pada blok.

    Baris latih selalu sebelum baris uji. Target baris i adalah return ke hari i+1, yang
    sudah diketahui saat penutupan hari test_start, jadi tidak perlu jeda.
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
    """Prediksi out-of-sample untuk setiap baris setelah jendela latih pertama."""
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
        # R^2 out-of-sample terhadap random walk (selalu memprediksi 0%).
        "r2_vs_random_walk": 1 - float((err**2).sum()) / sse_zero if sse_zero else float("nan"),
        "directional_accuracy": directional_accuracy(actual, pred),
        "information_coefficient": float(pred.corr(actual, method="spearman"))
        if pred.nunique() > 1
        else float("nan"),
        "n_predictions": int(len(actual)),
    }


def directional_accuracy(actual: pd.Series, pred: pd.Series) -> float:
    """Persentase hari dengan arah prediksi benar.

    Hari dengan return aktual 0 (harga tidak berubah) tidak dihitung, karena tidak punya
    arah. Model yang tidak pernah memberi arah (selalu memprediksi 0) tidak diberi nilai.
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
