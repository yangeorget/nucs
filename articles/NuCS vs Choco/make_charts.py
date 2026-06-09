###############################################################################
# Generates the benchmark charts for the "NuCS vs Choco" article.
# Run from this directory:  python make_charts.py
###############################################################################
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f77b4"   # Choco
ORANGE = "#ff7f0e"  # NuCS
GREEN = "#2ca02c"   # NuCS variant

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)


def chart(filename, title, xlabel, x, series):
    """series: list of (label, ydata, color, dnf_mask)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for label, y, color, dnf in series:
        ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=5, label=label)
        if dnf:
            for xi, yi, is_dnf in zip(x, y, dnf):
                if is_dnf:
                    ax.annotate("DNF", (xi, yi), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time (ms) — lower is better")
    ax.set_xticks(x)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.03)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# 1. All-interval series
chart(
    "all_interval_series.png",
    "All-interval series — time (ms) vs size",
    "series size n",
    [500, 1000, 2000, 4000, 8000, 16000],
    [
        ("Choco (BC, ff)", [240, 392, 950, 3261, 16595, 120340], BLUE, None),
        ("NuCS (BC, ff)", [225, 398, 972, 3352, 15153, 85236], ORANGE, None),
    ],
)

# 2. Schur's lemma
chart(
    "schur_lemma.png",
    "Schur's lemma — time (ms) vs size",
    "problem size n",
    [100, 200, 400, 800, 1600],
    [
        ("Choco (BC)", [197, 333, 702, 2701, 17864], BLUE, None),
        ("NuCS (BC)", [418, 542, 1050, 3118, 14592], ORANGE, None),
    ],
)

# 3. Latin square (n=50 plain models did not finish -> plotted as a spike to the top)
DNF = 900
chart(
    "latin_square.png",
    "Latin square — time (ms) vs size",
    "square size n",
    [20, 30, 40, 50],
    [
        ("Choco (BC)", [105, 126, 180, DNF], BLUE, [False, False, False, True]),
        ("NuCS (BC)", [376, 380, 397, DNF], ORANGE, [False, False, False, True]),
        ("NuCS (BC + redundant)", [412, 453, 547, 728], GREEN, None),
    ],
)

# 4. Magic sequence
chart(
    "magic_sequence.png",
    "Magic sequence — time (ms) vs size",
    "sequence size n",
    [100, 200, 300, 400],
    [
        ("Choco (AC)", [149, 307, 779, 1783], BLUE, None),
        ("NuCS (BC, r1)", [416, 710, 1451, 2913], ORANGE, None),
        ("NuCS (BC, r1 + r2)", [387, 455, 624, 942], GREEN, None),
    ],
)

# 5. Golomb ruler
chart(
    "golomb_ruler.png",
    "Golomb ruler — time (ms) vs number of marks",
    "marks",
    [9, 10, 11, 12],
    [
        ("Choco (enum. domains)", [202, 481, 6800, 65705], BLUE, None),
        ("NuCS (BC)", [438, 883, 12428, 128040], ORANGE, None),
        ("NuCS (custom consistency)", [414, 641, 6791, 67319], GREEN, None),
    ],
)
