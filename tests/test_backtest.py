import numpy as np
import pandas as pd
import pytest

from stockpred.backtest import backtest, performance


def test_always_long_matches_compounded_return_minus_entry_fee():
    r = pd.Series(np.log([1.01, 0.99, 1.02]))
    bt = backtest(pd.Series(1.0, index=r.index), r, buy_fee=0.001, sell_fee=0.002)
    expected = (1.01 - 0.001) * 0.99 * 1.02
    assert bt["equity"].iloc[-1] == pytest.approx(expected)


def test_fees_charged_on_each_position_change():
    r = pd.Series(np.zeros(4))
    pos = pd.Series([1.0, 0.0, 1.0, 0.0])
    bt = backtest(pos, r, buy_fee=0.001, sell_fee=0.002)
    assert bt["return"].tolist() == pytest.approx([-0.001, -0.002, -0.001, -0.002])
    assert performance(bt)["trades"] == 2


def test_cash_position_earns_nothing():
    r = pd.Series(np.log([1.05, 0.9, 1.1]))
    bt = backtest(pd.Series(0.0, index=r.index), r)
    assert bt["equity"].tolist() == [1.0, 1.0, 1.0]
