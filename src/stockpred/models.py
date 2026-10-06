"""Model baseline dan model machine learning, semuanya punya fit(X, y) / predict(X)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RandomWalk:
    """Prediksi return 0: harga besok sama dengan harga hari ini."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalk":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class HistoricalMean:
    """Prediksi rata-rata return harian pada data latih."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMean":
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class Momentum:
    """Prediksi bahwa return hari ini terulang besok."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Momentum":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["ret_lag1"].to_numpy()


def ridge() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0))


def lightgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    factory: Callable[[], object]
    baseline: bool


MODELS: list[ModelSpec] = [
    ModelSpec("random_walk", "Random walk", RandomWalk, True),
    ModelSpec("hist_mean", "Rata-rata historis", HistoricalMean, True),
    ModelSpec("momentum", "Momentum", Momentum, True),
    ModelSpec("ridge", "Regresi Ridge", ridge, False),
    ModelSpec("lightgbm", "LightGBM", lightgbm, False),
]
