import numpy as np
import pandas as pd

from src.evaluate import directional_accuracy, forecast_metrics, walk_forward_predict, walk_forward_splits


def test_splits_train_strictly_before_test_and_cover_everything():
    covered = []
    for train_end, test_start, test_end in walk_forward_splits(100, min_train=40, step=7):
        assert train_end <= test_start < test_end
        covered.extend(range(test_start, test_end))
    assert covered == list(range(40, 100))


class RecordingModel:
    seen_max = []

    def fit(self, X, y):
        self.last_train_date = X.index.max()
        return self

    def predict(self, X):
        RecordingModel.seen_max.append((self.last_train_date, X.index.min()))
        return np.zeros(len(X))


def test_walk_forward_never_trains_on_test_dates():
    idx = pd.bdate_range("2020-01-01", periods=120)
    X = pd.DataFrame({"a": np.arange(120.0)}, index=idx)
    y = pd.Series(np.random.default_rng(0).normal(size=120), index=idx)
    y.iloc[-1] = np.nan
    preds = walk_forward_predict(X, y, RecordingModel, min_train=50, step=10)
    assert len(preds) == 119 - 50
    assert all(train_max < test_min for train_max, test_min in RecordingModel.seen_max)


def test_metrics_perfect_and_random_walk():
    actual = pd.Series([0.01, -0.02, 0.03, -0.01])
    perfect = forecast_metrics(actual, actual)
    assert perfect["directional_accuracy"] == 1.0
    assert perfect["r2_vs_random_walk"] == 1.0
    zero = forecast_metrics(actual, actual * 0)
    assert zero["r2_vs_random_walk"] == 0.0


def test_directional_accuracy_ignores_flat_days_and_directionless_models():
    actual = pd.Series([0.01, 0.0, -0.02, 0.0])
    assert directional_accuracy(actual, pd.Series([0.1, 0.1, 0.1, 0.1])) == 0.5
    assert np.isnan(directional_accuracy(actual, pd.Series([0.0, 0.0, 0.0, 0.0])))
