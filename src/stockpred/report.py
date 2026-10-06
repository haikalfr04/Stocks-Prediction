"""Hitung semua hasil untuk laporan: tabel LaTeX, grafik PDF, dan paragraf hasil.

Cara pakai:
    python -m stockpred.report --excel data/data_saham.xlsx
    python -m stockpred.report --synthetic      # uji coba tanpa data asli

Setelah itu kompilasi laporan:
    cd laporan && latexmk -pdf laporan.tex
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config  # noqa: E402
from .backtest import backtest, performance, signal_from_prediction  # noqa: E402
from .data import load_excel, synthetic_prices  # noqa: E402
from .evaluate import forecast_metrics, walk_forward_predict  # noqa: E402
from .features import build_dataset  # noqa: E402
from .models import MODELS, lightgbm  # noqa: E402

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]
BULAN_SINGKAT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

WARNA_AKTUAL = "#52514e"
WARNA_BUY_HOLD = "#8a8984"
WARNA_LGBM = "#2a78d6"
WARNA_RIDGE = "#eb6834"
WARNA_SAHAM = ["#2a78d6", "#eb6834", "#1baf7a"]
WARNA_UJI = "#2a78d6"


# ---------------------------------------------------------------------------
# Format angka gaya Indonesia (koma desimal, titik ribuan)
# ---------------------------------------------------------------------------

def _kosong(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def angka(x: float, d: int = 2) -> str:
    if _kosong(x):
        return "--"
    s = f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("$-$" if x < 0 and round(abs(x), d) != 0 else "") + s


def persen(x: float, d: int = 1, tanda: bool = False) -> str:
    if _kosong(x):
        return "--"
    plus = "+" if tanda and round(x * 100, d) > 0 else ""
    return plus + angka(x * 100, d) + r"\%"


def rupiah(x: float) -> str:
    return "Rp" + angka(x, 0)


def tanggal(ts: pd.Timestamp) -> str:
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"


def _fmt_bulan(x, _pos=None) -> str:
    d = matplotlib.dates.num2date(x)
    return BULAN_SINGKAT[d.month - 1]


# ---------------------------------------------------------------------------
# Analisis per saham
# ---------------------------------------------------------------------------

def analisis(kode: str, nama: str, prices: pd.DataFrame, test_start: str) -> dict:
    X, y = build_dataset(prices)
    labeled = y.dropna()

    # Untuk baris t, hari yang diprediksi adalah hari bursa berikutnya (t+1).
    next_date = pd.Series(prices.index[1:], index=prices.index[:-1])
    target_date = next_date.reindex(labeled.index)
    n_train = int((target_date < pd.Timestamp(test_start)).sum())
    if n_train < 500 or n_train >= len(labeled):
        raise ValueError(f"{kode}: data latih ({n_train} baris) atau data uji tidak cukup")

    preds = {
        s.key: walk_forward_predict(X, y, s.factory, n_train, config.RETRAIN_EVERY) for s in MODELS
    }
    idx = preds["lightgbm"].index
    actual = labeled.reindex(idx)
    tgl = pd.DatetimeIndex(target_date.reindex(idx).to_numpy())

    models = []
    for s in MODELS:
        bt = backtest(signal_from_prediction(preds[s.key]), actual)
        models.append({
            "key": s.key,
            "label": s.label,
            "baseline": s.baseline,
            "forecast": forecast_metrics(actual, preds[s.key]),
            "strategy": performance(bt),
            "equity": pd.Series(bt["equity"].to_numpy(), index=tgl),
        })
    bh = backtest(pd.Series(1.0, index=idx), actual)

    # Terjemahkan prediksi return menjadi prediksi harga penutupan besok.
    close_t = prices["Close"].reindex(idx)
    harga_aktual = pd.Series((close_t * np.exp(actual)).to_numpy(), index=tgl)
    harga_pred = pd.Series((close_t * np.exp(preds["lightgbm"])).to_numpy(), index=tgl)
    harga_naif = pd.Series(close_t.to_numpy(), index=tgl)

    def rmse(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()))

    def mape(a, b):
        return float((abs(a - b) / a).mean())

    bergerak = (actual != 0).to_numpy()
    hit = pd.Series(((preds["lightgbm"] > 0) == (actual > 0)).to_numpy(), index=tgl)[bergerak]

    # Model final: latih pada seluruh data, prediksi hari bursa setelah data terakhir.
    final = lightgbm().fit(X[y.notna()], labeled)
    pred_besok = float(final.predict(X.iloc[[-1]])[0])
    importance = pd.Series(final.booster_.feature_importance("gain"), index=X.columns)
    importance = (importance / importance.sum()).sort_values(ascending=False)

    return {
        "kode": kode,
        "nama": nama,
        "prices": prices,
        "n_train": n_train,
        "uji_awal": tgl[0],
        "uji_akhir": tgl[-1],
        "n_uji": len(idx),
        "models": models,
        "buy_hold": performance(bh),
        "buy_hold_equity": pd.Series(bh["equity"].to_numpy(), index=tgl),
        "return_aktual": pd.Series(actual.to_numpy(), index=tgl),
        "return_pred": pd.Series(preds["lightgbm"].to_numpy(), index=tgl),
        "harga_aktual": harga_aktual,
        "harga_pred": harga_pred,
        "harga": {
            "rmse_lgbm": rmse(harga_aktual, harga_pred),
            "rmse_naif": rmse(harga_aktual, harga_naif),
            "mape_lgbm": mape(harga_aktual, harga_pred),
            "mape_naif": mape(harga_aktual, harga_naif),
        },
        # Bulan dengan kurang dari 10 hari bursa (misalnya bulan berjalan) tidak dipakai.
        "hit_bulanan": hit.groupby(hit.index.month).mean()[hit.groupby(hit.index.month).size() >= 10],
        "importance": importance,
        "besok": {
            "per": X.index[-1],
            "close": float(prices["Close"].iloc[-1]),
            "return": pred_besok,
            "harga": float(prices["Close"].iloc[-1] * math.exp(pred_besok)),
        },
    }


def _model(h: dict, key: str) -> dict:
    return next(m for m in h["models"] if m["key"] == key)


def _baseline_terbaik(h: dict) -> dict:
    kandidat = [m for m in h["models"]
                if m["baseline"] and not _kosong(m["forecast"]["directional_accuracy"])]
    return max(kandidat, key=lambda m: m["forecast"]["directional_accuracy"])


# ---------------------------------------------------------------------------
# Tabel LaTeX
# ---------------------------------------------------------------------------

def tabel_statistik(hasil: list[dict]) -> str:
    baris = []
    for h in hasil:
        p = h["prices"]
        ret = np.log(p["Close"]).diff().dropna()
        tahun_ini = p.loc[p.index >= pd.Timestamp(config.TEST_START), "Close"]
        sebelum = p.loc[p.index < pd.Timestamp(config.TEST_START), "Close"].iloc[-1]
        baris.append(" & ".join([
            h["kode"],
            f"{p.index[0]:%d/%m/%Y}--{p.index[-1]:%d/%m/%Y}",
            angka(len(p), 0),
            rupiah(p["Close"].iloc[-1]),
            persen(ret.mean(), 3),
            persen(ret.std() * np.sqrt(config.TRADING_DAYS), 1),
            persen(tahun_ini.iloc[-1] / sebelum - 1, 1, tanda=True),
        ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{lcrrrrr}",
        r"\toprule",
        r"Saham & Periode data & Hari bursa & Harga terakhir & Rata-rata return & Volatilitas & Return 2026 \\",
        r" & & & & harian & tahunan & (s.d. data akhir) \\",
        r"\midrule",
        *baris,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def tabel_model(h: dict) -> str:
    baris = []
    for m in h["models"]:
        f, s = m["forecast"], m["strategy"]
        nama = rf"\textbf{{{m['label']}}}" if m["key"] == "lightgbm" else m["label"]
        if m["baseline"]:
            nama = nama + r"$^{\dagger}$"
        baris.append(" & ".join([
            nama,
            persen(f["directional_accuracy"]),
            persen(f["r2_vs_random_walk"], 2),
            persen(f["mae"], 2),
            angka(s["sharpe"]),
            persen(s["total_return"], 1, tanda=True),
            persen(s["max_drawdown"]),
            angka(s["trades"], 0),
        ]) + r" \\")
    bh = h["buy_hold"]
    baris.append(r"\midrule")
    baris.append(" & ".join([
        r"\textit{Buy \& hold}", "--", "--", "--", angka(bh["sharpe"]),
        persen(bh["total_return"], 1, tanda=True), persen(bh["max_drawdown"]), "1",
    ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Akurasi prediksi return} & \multicolumn{4}{c}{Strategi (setelah biaya)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"Model & Akurasi arah & $R^2_{\mathrm{RW}}$ & MAE & Sharpe & Return & Drawdown & Transaksi \\",
        r"\midrule",
        *baris,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def tabel_ringkasan(hasil: list[dict]) -> str:
    baris = []
    for h in hasil:
        lg = _model(h, "lightgbm")
        bb = _baseline_terbaik(h)
        hg = h["harga"]
        baris.append(" & ".join([
            h["kode"],
            persen(lg["forecast"]["directional_accuracy"]),
            persen(bb["forecast"]["directional_accuracy"]),
            rupiah(hg["rmse_lgbm"]),
            rupiah(hg["rmse_naif"]),
            persen(lg["strategy"]["total_return"], 1, tanda=True),
            persen(h["buy_hold"]["total_return"], 1, tanda=True),
        ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Akurasi arah} & \multicolumn{2}{c}{RMSE harga} & \multicolumn{2}{c}{Return 2026} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Saham & LightGBM & Baseline terbaik & LightGBM & Naif & Strategi & Buy \& hold \\",
        r"\midrule",
        *baris,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def tabel_prediksi(hasil: list[dict]) -> str:
    baris = []
    for h in hasil:
        b = h["besok"]
        sinyal = r"\textsc{beli}" if b["return"] > 0 else r"\textsc{tunai}"
        baris.append(" & ".join([
            h["kode"], tanggal(b["per"]), rupiah(b["close"]),
            persen(math.expm1(b["return"]), 2, tanda=True), rupiah(b["harga"]), sinyal,
        ]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{llrrrc}",
        r"\toprule",
        r"Saham & Data per & Harga penutupan & Prediksi return & Prediksi harga & Sinyal \\",
        r"\midrule",
        *baris,
        r"\bottomrule",
        r"\end{tabular}",
    ])


# ---------------------------------------------------------------------------
# Paragraf hasil (otomatis mengikuti angka)
# ---------------------------------------------------------------------------

def _banding(a: float, b: float, d: int = 3) -> str:
    if round(a, d) > round(b, d):
        return "lebih tinggi daripada"
    if round(a, d) < round(b, d):
        return "lebih rendah daripada"
    return "sama dengan"


def paragraf_saham(h: dict) -> str:
    lg = _model(h, "lightgbm")
    bb = _baseline_terbaik(h)
    acc = lg["forecast"]["directional_accuracy"]
    r2 = lg["forecast"]["r2_vs_random_walk"]
    hg = h["harga"]
    tr, bh = lg["strategy"]["total_return"], h["buy_hold"]["total_return"]

    teks = [
        f"Selama periode uji ({tanggal(h['uji_awal'])} sampai {tanggal(h['uji_akhir'])}, "
        f"{h['n_uji']} hari bursa), LightGBM menebak arah pergerakan harian {h['kode']} "
        f"dengan benar pada {persen(acc)} hari. Angka ini {_banding(acc, bb['forecast']['directional_accuracy'])} "
        f"baseline terbaik, yaitu {bb['label'].lower()} ({persen(bb['forecast']['directional_accuracy'])}).",
    ]
    if r2 > 0:
        teks.append(
            f"Nilai $R^2_{{\\mathrm{{RW}}}}$ sebesar {persen(r2, 2)} berarti galat kuadrat prediksi "
            f"return LightGBM sedikit lebih kecil daripada sekadar menebak return 0\\%."
        )
    else:
        teks.append(
            f"Nilai $R^2_{{\\mathrm{{RW}}}}$ sebesar {persen(r2, 2)} berarti galat kuadrat prediksi "
            f"return LightGBM justru lebih besar daripada sekadar menebak return 0\\% (random walk)."
        )
    lebih = "lebih kecil" if hg["rmse_lgbm"] < hg["rmse_naif"] else "tidak lebih kecil"
    teks.append(
        f"Jika diterjemahkan ke harga, RMSE prediksi harga penutupan besok adalah "
        f"{rupiah(hg['rmse_lgbm'])} (MAPE {persen(hg['mape_lgbm'], 2)}), {lebih} dibandingkan "
        f"prediksi naif \\emph{{harga besok = harga hari ini}} dengan RMSE {rupiah(hg['rmse_naif'])} "
        f"(MAPE {persen(hg['mape_naif'], 2)})."
    )
    hasil_strategi = "mengungguli" if tr > bh else "tertinggal dari"
    teks.append(
        f"Strategi yang hanya membeli {h['kode']} ketika LightGBM memprediksi kenaikan menghasilkan "
        f"return {persen(tr, 1, tanda=True)} setelah biaya transaksi, sehingga {hasil_strategi} "
        f"strategi \\emph{{buy \\& hold}} ({persen(bh, 1, tanda=True)}). Strategi ini melakukan "
        f"{lg['strategy']['trades']} kali pembelian dan berada di pasar pada "
        f"{persen(lg['strategy']['exposure'], 0)} hari."
    )
    return " ".join(teks)


def kalimat_ringkasan(hasil: list[dict]) -> tuple[str, str]:
    n = len(hasil)
    n_acc = sum(_model(h, "lightgbm")["forecast"]["directional_accuracy"]
                > _baseline_terbaik(h)["forecast"]["directional_accuracy"] for h in hasil)
    n_harga = sum(h["harga"]["rmse_lgbm"] < h["harga"]["rmse_naif"] for h in hasil)
    n_strat = sum(_model(h, "lightgbm")["strategy"]["total_return"] > h["buy_hold"]["total_return"]
                  for h in hasil)
    acc = [_model(h, "lightgbm")["forecast"]["directional_accuracy"] for h in hasil]
    ringkas = (
        f"Akurasi arah LightGBM pada periode uji berkisar antara {persen(min(acc))} dan "
        f"{persen(max(acc))}. LightGBM mengungguli baseline terbaik pada {n_acc} dari {n} saham, "
        f"menghasilkan galat harga lebih kecil daripada prediksi naif pada {n_harga} dari {n} saham, "
        f"dan strategi berbasis LightGBM mengalahkan \\emph{{buy \\& hold}} pada {n_strat} dari {n} saham."
    )
    kesimpulan = ringkas + (
        " Keunggulan yang konsisten pada ketiga ukuran tersebut sekaligus tidak ditemukan."
        if min(n_acc, n_harga, n_strat) < n
        else " Model unggul pada ketiga ukuran untuk semua saham, tetapi periode uji yang hanya "
        "satu tahun belum cukup untuk memastikan keunggulan ini bertahan."
    )
    return ringkas, kesimpulan


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------

def _gaya() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ae",
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "pdf.fonttype": 42,
    })


def _rp(x, _pos=None) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def _pct(x, _pos=None) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def grafik_harga(hasil: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(len(hasil), 1, figsize=(6.3, 5.6), sharex=True)
    for ax, h, warna in zip(axes, hasil, WARNA_SAHAM):
        p = h["prices"]["Close"]
        ax.plot(p.index, p.to_numpy(), color=warna, lw=1.1)
        ax.axvspan(h["uji_awal"], p.index[-1], color=WARNA_UJI, alpha=0.08, lw=0)
        ax.set_title(f"{h['kode']} - {h['nama']}", loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(_rp))
    axes[0].text(hasil[0]["uji_awal"], 1.0, " periode uji 2026", transform=axes[0].get_xaxis_transform(),
                 va="top", fontsize=7.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(out / "harga.pdf")
    plt.close(fig)


def grafik_prediksi(h: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(h["harga_aktual"].index, h["harga_aktual"].to_numpy(), color=WARNA_AKTUAL, lw=1.6,
             label="Harga aktual")
    ax1.plot(h["harga_pred"].index, h["harga_pred"].to_numpy(), color=WARNA_LGBM, lw=1.0,
             label="Prediksi LightGBM")
    ax1.yaxis.set_major_formatter(FuncFormatter(_rp))
    ax1.set_title("Harga penutupan: aktual vs prediksi (satu hari ke depan)", loc="left")
    ax1.legend(loc="best", ncol=2)

    r = h["return_aktual"]
    ax2.bar(r.index, r.to_numpy(), color="#c9c8c2", width=1.0, label="Return aktual")
    ax2.plot(h["return_pred"].index, h["return_pred"].to_numpy(), color=WARNA_LGBM, lw=1.2,
             label="Prediksi LightGBM")
    ax2.axhline(0, color="#8a8984", lw=0.6)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_title("Return harian: aktual vs prediksi", loc="left")
    ax2.legend(loc="upper left", ncol=2)
    ax2.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_bulan))
    fig.tight_layout()
    fig.savefig(out / f"prediksi_{h['kode']}.pdf")
    plt.close(fig)


def grafik_ekuitas(hasil: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(hasil), figsize=(6.3, 2.5), sharey=True)
    for ax, h in zip(axes, hasil):
        ax.plot(h["buy_hold_equity"].index, h["buy_hold_equity"].to_numpy(), color=WARNA_BUY_HOLD,
                lw=1.2, ls="--", label="Buy & hold")
        for key, warna, label in (("ridge", WARNA_RIDGE, "Strategi Ridge"),
                                  ("lightgbm", WARNA_LGBM, "Strategi LightGBM")):
            eq = _model(h, key)["equity"]
            ax.plot(eq.index, eq.to_numpy(), color=warna, lw=1.2, label=label)
        ax.axhline(1, color="#8a8984", lw=0.6)
        ax.set_title(h["kode"], loc="left")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_bulan))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"Rp{x:.2f}".replace(".", ",")))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "ekuitas.pdf")
    plt.close(fig)


def grafik_akurasi_bulanan(hasil: list[dict], out: Path) -> None:
    bulan = sorted(set().union(*(h["hit_bulanan"].index for h in hasil)))
    x = np.arange(len(bulan))
    lebar = 0.8 / len(hasil)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for i, (h, warna) in enumerate(zip(hasil, WARNA_SAHAM)):
        v = h["hit_bulanan"].reindex(bulan).to_numpy()
        ax.bar(x + (i - (len(hasil) - 1) / 2) * lebar, v, width=lebar * 0.9, color=warna, label=h["kode"])
    ax.axhline(0.5, color="#52514e", lw=0.8, ls="--", label="50% = tebakan acak")
    ax.set_xticks(x, [BULAN_SINGKAT[b - 1] for b in bulan])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))
    ax.legend(loc="upper left", ncol=len(hasil) + 1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "akurasi_bulanan.pdf")
    plt.close(fig)


def grafik_fitur(hasil: list[dict], out: Path, top: int = 8) -> None:
    fig, axes = plt.subplots(1, len(hasil), figsize=(6.3, 2.7))
    for ax, h in zip(axes, hasil):
        imp = h["importance"].head(top)[::-1]
        ax.barh(imp.index, imp.to_numpy(), color=WARNA_LGBM, height=0.65)
        ax.set_title(h["kode"], loc="left")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v * 100:.0f}%"))
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fitur.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tulis_laporan(hasil: list[dict], out: Path, sintetis: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tabel_statistik.tex").write_text(tabel_statistik(hasil))
    (out / "tabel_ringkasan.tex").write_text(tabel_ringkasan(hasil))
    (out / "tabel_prediksi.tex").write_text(tabel_prediksi(hasil))
    for h in hasil:
        (out / f"tabel_model_{h['kode']}.tex").write_text(tabel_model(h))
        (out / f"hasil_{h['kode']}.tex").write_text(paragraf_saham(h))
    ringkas, kesimpulan = kalimat_ringkasan(hasil)
    (out / "kesimpulan.tex").write_text(kesimpulan)

    h0 = hasil[0]
    angka_tex = [
        rf"\newcommand{{\TanggalDibuat}}{{{tanggal(pd.Timestamp(datetime.now()))}}}",
        rf"\newcommand{{\DataAwal}}{{{tanggal(min(h['prices'].index[0] for h in hasil))}}}",
        rf"\newcommand{{\DataAkhir}}{{{tanggal(max(h['prices'].index[-1] for h in hasil))}}}",
        rf"\newcommand{{\UjiAwal}}{{{tanggal(h0['uji_awal'])}}}",
        rf"\newcommand{{\UjiAkhir}}{{{tanggal(h0['uji_akhir'])}}}",
        rf"\newcommand{{\JumlahHariUji}}{{{h0['n_uji']}}}",
        rf"\newcommand{{\JumlahLatih}}{{{angka(h0['n_train'], 0)}}}",
        rf"\newcommand{{\SimpanganAcak}}{{{angka(100 * math.sqrt(0.25 / h0['n_uji']), 1)}}}",
        rf"\newcommand{{\RingkasanHasil}}{{{ringkas}}}",
        r"\newif\ifsintetis",
        r"\sintetistrue" if sintetis else r"\sintetisfalse",
    ]
    (out / "angka.tex").write_text("\n".join(angka_tex) + "\n")

    _gaya()
    grafik_harga(hasil, out)
    for h in hasil:
        grafik_prediksi(h, out)
    grafik_ekuitas(hasil, out)
    grafik_akurasi_bulanan(hasil, out)
    grafik_fitur(hasil, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat tabel dan grafik untuk laporan")
    sumber = parser.add_mutually_exclusive_group(required=True)
    sumber.add_argument("--excel", type=Path, help="file Excel hasil scripts/download_data.py")
    sumber.add_argument("--synthetic", action="store_true", help="pakai data sintetis (uji coba)")
    parser.add_argument("--out", type=Path, default=Path("laporan/generated"))
    args = parser.parse_args()

    kode = list(config.SAHAM)
    if args.synthetic:
        data = {k: synthetic_prices(seed=i) for i, k in enumerate(kode)}
    else:
        data = load_excel(args.excel, kode)

    hasil = []
    for k in kode:
        print(f"[{k}] {len(data[k])} baris, menjalankan walk-forward ...", flush=True)
        hasil.append(analisis(k, config.SAHAM[k], data[k], config.TEST_START))
    tulis_laporan(hasil, args.out, sintetis=args.synthetic)
    print(f"Selesai. Hasil tersimpan di {args.out}. Kompilasi: cd laporan && latexmk -pdf laporan.tex")


if __name__ == "__main__":
    main()
