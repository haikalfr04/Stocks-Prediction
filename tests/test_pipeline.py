import json

from stockpred import config
from stockpred.data import synthetic_prices
from stockpred.pipeline import _clean, analyze_ticker


def test_pipeline_produces_complete_json(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_DAYS", 300)
    monkeypatch.setattr(config, "RETRAIN_EVERY", 100)
    result = analyze_ticker("TEST.JK", "Test", synthetic_prices(seed=5, n_days=600))
    payload = json.loads(json.dumps(_clean(result), allow_nan=False))
    assert payload["symbol"] == "TEST"
    assert {m["key"] for m in payload["models"]} >= {"random_walk", "ridge", "lightgbm"}
    n = len(payload["equity"]["dates"])
    assert n > 0 and len(payload["equity"]["buy_hold"]) == n == len(payload["equity"]["lightgbm"])
    assert payload["forecast"]["signal"] in {"LONG", "CASH"}
