Test period: 2026-01-02 to 2026-09-30 (176 trading days). Models are retrained every 21 trading days. Fees: 0.15% buy, 0.25% sell.

*Up days*: share of days with a price change on which the price rose. *Chance band*: a coin flip stays within ±7.6% of 50% on 95% of runs of this length. *Shift test p-value*: share of time-shifted copies of the strategy (same days in the market, same number of trades, unrelated timing) that earned at least as much. Values below 0.05 suggest the timing is not luck.

## Summary

| Stock | Up days | Direction accuracy (LightGBM) | Best baseline | Price RMSE (LightGBM) | Price RMSE (naive) | Strategy return | Buy & hold return | Shift test p-value |
|---|---|---|---|---|---|---|---|---|
| ASII | 42.3% | 50.0% | 42.3% (Historical mean) | Rp171 | Rp168 | -27.0% | -28.0% | 0.56 |
| BBRI | 47.3% | 44.3% | 50.3% (Momentum) | Rp68 | Rp66 | -22.0% | -8.8% | 0.62 |
| TLKM | 48.1% | 50.6% | 50.6% (Momentum) | Rp85 | Rp86 | -5.6% | -28.9% | 0.15 |


## ASII - Astra International

| Model | Direction accuracy | R² vs random walk | MAE | Sharpe | Return | Max drawdown | Trades | Shift test p-value |
|---|---|---|---|---|---|---|---|---|
| Random walk (baseline) | - | 0.00% | 1.99% | - | +0.0% | 0.0% | 0 | - |
| Historical mean (baseline) | 42.3% | -0.05% | 1.99% | -0.83 | -28.0% | -38.1% | 1 | - |
| Momentum (baseline) | 37.5% | -147.70% | 3.33% | -6.33 | -65.4% | -68.0% | 55 | - |
| Ridge regression | 55.4% | -1.65% | 1.99% | -1.25 | -27.3% | -32.1% | 35 | 0.55 |
| **LightGBM** | 50.0% | -3.76% | 2.05% | -1.02 | -27.0% | -34.0% | 25 | 0.56 |
| *Buy & hold* | - | - | - | -0.83 | -28.0% | -38.1% | 1 | - |


## BBRI - Bank Rakyat Indonesia

| Model | Direction accuracy | R² vs random walk | MAE | Sharpe | Return | Max drawdown | Trades | Shift test p-value |
|---|---|---|---|---|---|---|---|---|
| Random walk (baseline) | - | 0.00% | 1.55% | - | +0.0% | 0.0% | 0 | - |
| Historical mean (baseline) | 47.3% | -0.12% | 1.55% | -0.24 | -8.8% | -30.5% | 1 | - |
| Momentum (baseline) | 50.3% | -105.92% | 2.16% | -1.17 | -16.9% | -20.3% | 44 | - |
| Ridge regression | 47.9% | -1.15% | 1.56% | -1.52 | -24.2% | -30.5% | 53 | 0.56 |
| **LightGBM** | 44.3% | -2.37% | 1.61% | -1.24 | -22.0% | -29.9% | 40 | 0.62 |
| *Buy & hold* | - | - | - | -0.24 | -8.8% | -30.5% | 1 | - |


## TLKM - Telkom Indonesia

| Model | Direction accuracy | R² vs random walk | MAE | Sharpe | Return | Max drawdown | Trades | Shift test p-value |
|---|---|---|---|---|---|---|---|---|
| Random walk (baseline) | - | 0.00% | 1.90% | - | +0.0% | 0.0% | 0 | - |
| Historical mean (baseline) | 48.1% | -0.10% | 1.91% | -0.86 | -28.9% | -40.4% | 1 | - |
| Momentum (baseline) | 50.6% | -99.59% | 2.69% | -1.15 | -23.5% | -27.0% | 42 | - |
| Ridge regression | 50.6% | 3.77% | 1.91% | -0.16 | -8.2% | -29.8% | 29 | 0.14 |
| **LightGBM** | 50.6% | 1.24% | 1.94% | -0.14 | -5.6% | -22.4% | 36 | 0.15 |
| *Buy & hold* | - | - | - | -0.86 | -28.9% | -40.4% | 1 | - |
