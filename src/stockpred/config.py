"""Project-wide settings."""

TICKERS: dict[str, str] = {
    "ASII.JK": "Astra International",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "TLKM.JK": "Telkom Indonesia",
}

START_DATE = "2015-01-01"

# Walk-forward validation: train on at least ~3 years, retrain every ~month.
MIN_TRAIN_DAYS = 750
RETRAIN_EVERY = 21

# Typical IDX retail broker fees. The sell side includes the 0.1% final income tax.
BUY_FEE = 0.0015
SELL_FEE = 0.0025

TRADING_DAYS = 252
