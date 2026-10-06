"""Pengaturan proyek."""

# Kode saham -> nama perusahaan. Kode dipakai sebagai nama sheet di file Excel.
SAHAM: dict[str, str] = {
    "ASII": "Astra International",
    "BBRI": "Bank Rakyat Indonesia",
    "TLKM": "Telkom Indonesia",
}

# Periode uji: semua hari bursa di tahun 2026. Data sebelumnya hanya untuk pelatihan.
TEST_START = "2026-01-01"

# Model dilatih ulang kira-kira setiap bulan (21 hari bursa) selama periode uji.
RETRAIN_EVERY = 21

# Biaya transaksi broker ritel IDX yang umum. Biaya jual sudah termasuk PPh final 0,1%.
BUY_FEE = 0.0015
SELL_FEE = 0.0025

TRADING_DAYS = 252
