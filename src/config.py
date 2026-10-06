"""Project settings shared by every script."""

# Code (also the Excel sheet name) -> (Yahoo Finance ticker, company name)
STOCKS: dict[str, tuple[str, str]] = {
    "ASII": ("ASII.JK", "Astra International"),
    "BBRI": ("BBRI.JK", "Bank Rakyat Indonesia"),
    "TLKM": ("TLKM.JK", "Telkom Indonesia"),
}

START_DATE = "2015-01-01"
DATA_PATH = "data/idx_prices.xlsx"
RESULTS_DIR = "results"

# Test period: every trading day in 2026. Earlier data is used only for training.
TEST_START = "2026-01-01"

# During the test period the models are retrained about once a month (21 trading days).
RETRAIN_EVERY = 21

# Typical IDX retail broker fees. The sell fee includes the 0.1% final income tax.
BUY_FEE = 0.0015
SELL_FEE = 0.0025

TRADING_DAYS = 252
