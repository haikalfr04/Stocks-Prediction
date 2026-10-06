import numpy as np
import pandas as pd

from stockpred.data import synthetic_prices
from stockpred.features import build_dataset, make_features, make_target


def test_features_do_not_look_ahead():
    prices = synthetic_prices(seed=1, n_days=300)
    cut = 200
    full = make_features(prices)
    # Scramble everything after the cut: features up to the cut must be unchanged.
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(0).uniform(0.5, 1.5, (300 - cut - 1, 1))
    partial = make_features(tampered)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], partial.iloc[: cut + 1])


def test_target_is_next_day_log_return():
    prices = synthetic_prices(seed=2, n_days=50)
    y = make_target(prices)
    expected = np.log(prices["Close"].iloc[11] / prices["Close"].iloc[10])
    assert np.isclose(y.iloc[10], expected)
    assert np.isnan(y.iloc[-1])


def test_dataset_has_no_missing_features_and_one_unlabeled_row():
    X, y = build_dataset(synthetic_prices(seed=3, n_days=400))
    assert not X.isna().any().any()
    assert y.isna().sum() == 1 and np.isnan(y.iloc[-1])
