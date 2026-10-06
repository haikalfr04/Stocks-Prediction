import pandas as pd

from stockpred.data import synthetic_prices
from stockpred.report import analisis, angka, persen, tulis_laporan


def test_format_angka_indonesia():
    assert angka(1234.5, 1) == "1.234,5"
    assert persen(-0.0123, 2) == r"$-$1,23\%"
    assert persen(0.05, 1, tanda=True) == r"+5,0\%"


def test_laporan_lengkap_dari_data_sintetis(tmp_path):
    hasil = [
        analisis(k, k, synthetic_prices(seed=i, n_days=900), "2026-01-01")
        for i, k in enumerate(["AAA", "BBB", "CCC"])
    ]
    h = hasil[0]
    assert h["uji_awal"] >= pd.Timestamp("2026-01-01")
    assert h["n_train"] + h["n_uji"] == 900 - 49 - 1  # 49 baris awal untuk MA50, baris akhir tanpa target
    tulis_laporan(hasil, tmp_path, sintetis=True)
    for nama in ["angka.tex", "tabel_ringkasan.tex", "hasil_AAA.tex", "harga.pdf", "prediksi_CCC.pdf"]:
        assert (tmp_path / nama).stat().st_size > 0
    assert r"\sintetistrue" in (tmp_path / "angka.tex").read_text()
