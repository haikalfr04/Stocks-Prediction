import json

import pandas as pd

from src.data import synthetic_prices
from src.run import run_stock, save_results


def test_walk_forward_covers_2026_and_writes_results(tmp_path):
    results = [run_stock(code, synthetic_prices(seed=i, n_days=900)) for i, code in enumerate(["ASII", "BBRI", "TLKM"])]
    r = results[0]
    assert pd.Timestamp(r["test_start"]) >= pd.Timestamp("2026-01-01")
    assert r["n_train"] + r["n_test"] == 900 - 49 - 1  # 49 rows warm-up for MA50, last row has no target
    assert r["models"]["random_walk"]["forecast"]["n_predictions"] == r["n_test"]

    save_results(results, tmp_path, synthetic=True)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["synthetic"] is True and set(metrics["stocks"]) == {"ASII", "BBRI", "TLKM"}
    assert "Synthetic data" in (tmp_path / "comparison.md").read_text()
    for name in ["model_comparison.png", "ASII_2026_predictions.png", "equity_curves.png"]:
        assert (tmp_path / "figures" / name).stat().st_size > 0
    daily = pd.read_csv(tmp_path / "predictions" / "TLKM_2026.csv", index_col=0, parse_dates=True)
    assert len(daily) == results[2]["n_test"]
