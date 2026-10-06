# Prediksi Return Harian ASII, BBRI, dan TLKM Selama 2026

Proyek ini menguji apakah return harian saham **ASII** (Astra International), **BBRI**
(Bank Rakyat Indonesia), dan **TLKM** (Telkom Indonesia) dapat diprediksi selama tahun 2026.
Hasilnya ditulis dalam laporan LaTeX berbahasa Indonesia (`laporan/laporan.tex`). Kode
programnya ikut dilampirkan di dalam laporan.

## Alur kerja

```
scripts/download_data.py  ──►  data/data_saham.xlsx  ──►  python -m stockpred.report  ──►  laporan/generated/  ──►  laporan.pdf
   (Yahoo Finance)              (1 sheet per saham)        (model, tabel, grafik)          (.tex + .pdf)          (latexmk)
```

### 1. Instal

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[download,dev]"
```

### 2. Unduh data ke Excel

```bash
python scripts/download_data.py
```

Hasilnya `data/data_saham.xlsx` dengan sheet `ASII`, `BBRI`, `TLKM`, dan `Keterangan`. Setiap
sheet berisi kolom Tanggal, Open, High, Low, Close, Adj Close, dan Volume sejak 2015. Script
ini berdiri sendiri, jadi bisa juga dijalankan di Google Colab
(`pip install yfinance pandas openpyxl`).

Opsi: `--start 2015-01-01`, `--end 2026-10-06` (eksklusif), dan `--out path/file.xlsx`.

### 3. Jalankan model dan buat tabel/grafik

```bash
python -m stockpred.report --excel data/data_saham.xlsx
```

Data sebelum 2026 dipakai untuk melatih model. Seluruh hari bursa tahun 2026 menjadi
periode uji dengan validasi *walk-forward*: model dilatih ulang setiap 21 hari bursa dan
hanya memakai data sebelum hari yang diprediksi. Semua tabel, grafik, dan paragraf hasil
ditulis ke `laporan/generated/`, sehingga angka di laporan selalu sesuai dengan data.

Untuk uji coba tanpa internet, pakai `--synthetic`. Laporan dari data sintetis otomatis
diberi peringatan "DATA SINTETIS".

### 4. Kompilasi laporan PDF

```bash
cd laporan
latexmk -pdf laporan.tex
```

Butuh TeX Live atau MiKTeX dengan paket `babel-indonesian`, `listings`, dan `booktabs`. Bisa
juga di Overleaf: unggah folder `laporan/` (termasuk `generated/`) beserta folder `scripts/`
dan `src/`, karena kode program dilampirkan dari sana. Jangan lupa ganti `\Penulis` dan
`\Institusi` di baris awal `laporan.tex`.

## Metodologi singkat

| Kesalahan umum | Yang dilakukan di proyek ini |
|---|---|
| Memprediksi level harga (terlihat akurat karena harga besok ≈ harga hari ini) | Memprediksi **log return** hari berikutnya, lalu dikonversi ke harga dan dibandingkan dengan prediksi naif |
| Split acak atau *scaling* dengan seluruh data (*data leakage*) | **Walk-forward**: latih dengan data masa lalu saja. Ada test otomatis yang memastikan fitur tidak memakai data masa depan |
| Tanpa pembanding | Dibandingkan dengan baseline **random walk, rata-rata historis, dan momentum** |
| Hanya melihat akurasi | **Backtest** strategi beli-atau-tunai dengan biaya IDX (beli 0,15%, jual 0,25%) dibandingkan dengan *buy & hold* |

## Struktur

| File | Isi |
|---|---|
| `scripts/download_data.py` | Unduh data dari Yahoo Finance ke Excel |
| `src/stockpred/config.py` | Daftar saham, awal periode uji, biaya transaksi |
| `src/stockpred/data.py` | Baca Excel dan sesuaikan harga dengan dividen/split |
| `src/stockpred/features.py` | Fitur teknikal (return lag, volatilitas, RSI, MACD, volume) |
| `src/stockpred/models.py` | Baseline, Ridge, LightGBM |
| `src/stockpred/evaluate.py` | Validasi walk-forward dan metrik |
| `src/stockpred/backtest.py` | Simulasi strategi dengan biaya |
| `src/stockpred/report.py` | Membuat tabel, grafik, dan paragraf untuk laporan |
| `laporan/laporan.tex` | Laporan LaTeX |
| `tests/` | Pengujian (`pytest`) |

---

Untuk tujuan pembelajaran dan portofolio, bukan saran investasi.
