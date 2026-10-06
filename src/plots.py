"""Shared chart style and colors."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ACTUAL = "#52514e"
BENCHMARK = "#8a8984"
NEUTRAL_BAR = "#c9c8c2"
LIGHTGBM = "#2a78d6"
RIDGE = "#eb6834"
STOCK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # ASII, BBRI, TLKM
TEST_SHADE = "#2a78d6"

rupiah = FuncFormatter(lambda x, _p: "Rp" + f"{x:,.0f}")
percent = FuncFormatter(lambda x, _p: f"{x * 100:.0f}%")
percent1 = FuncFormatter(lambda x, _p: f"{x * 100:.1f}%")
month = mdates.DateFormatter("%b")


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ae",
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    })
