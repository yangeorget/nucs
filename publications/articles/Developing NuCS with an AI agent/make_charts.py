"""
Makes the charts of the article from data/*.csv (see collect_data.py).

Usage, from this directory:

    uv run --with matplotlib python make_charts.py

The charts go to images/. The colors are the first three slots of a categorical palette that passes a color-vision
check (worst adjacent CVD delta E 9.2). Aqua is below 3:1 against the surface, so each series also has a direct label.
"""

import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # titles and values
INK_2 = "#52514e"  # axes, labels, notes
GRID = "#e4e3df"
BAND = "#eceae4"  # the agent period
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

COLLECTED = "2026-10-06"  # the day of collect_data.py
AGENT_START = "2026-05"  # the first commit with the trailer; the trailer is regular from 2026-07

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def read(name: str) -> dict[str, dict[str, str]]:
    with open(DATA / name) as f:
        return {row["month"]: row for row in csv.DictReader(f)}


def all_months(first: str, last: str) -> list[str]:
    """Every calendar month from first to last, so that a month with no commit shows as zero, not as a gap."""
    year, month = int(first[:4]), int(first[5:])
    months = []
    while f"{year:04d}-{month:02d}" <= last:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_ticks(ax: plt.Axes, months: list[str]) -> None:
    """A label every quarter, as a short month and year."""
    ticks = [i for i, m in enumerate(months) if m[5:] in ("01", "04", "07", "10")]
    ax.set_xticks(ticks, [date(int(months[i][:4]), int(months[i][5:]), 1).strftime("%b %Y") for i in ticks])
    ax.set_xlim(-0.7, len(months) - 0.3)


def agent_band(ax: plt.Axes, months: list[str], label: bool = True) -> None:
    start = months.index(AGENT_START) - 0.5
    ax.axvspan(start, len(months) - 0.5, color=BAND, zorder=0, linewidth=0)
    if label:
        ax.text(start + 0.3, 0.97, "with the agent", transform=ax.get_xaxis_transform(), va="top", color=INK_2,
                fontsize=9)


def stacked_bars(ax: plt.Axes, months: list[str], series: list[tuple[str, list[int], str]]) -> None:
    bottom = [0] * len(months)
    for label, values, color in series:
        # A white edge gives the 2 px gap between the segments of a stack.
        ax.bar(range(len(months)), values, bottom=bottom, width=0.75, color=color, edgecolor=SURFACE, linewidth=1.2,
               label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    # The legend text is in ink; its color patch carries the identity. The CSV in data/ is the table view.
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=9, labelcolor=INK_2)


def partial_month_note(months: list[str]) -> str:
    """A note for the last month, which is not complete when the data is collected."""
    last = date(int(months[-1][:4]), int(months[-1][5:]), 1).strftime("%B %Y")
    return f"{last} is not complete (data of {COLLECTED})."


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def chart_commits() -> None:
    data = read("commits_per_month.csv")
    months = all_months(min(data), max(data))
    get = lambda m, k: int(data[m][k]) if m in data else 0  # noqa: E731
    fig, ax = plt.subplots(figsize=(9, 4.2))
    agent_band(ax, months)
    stacked_bars(ax, months, [
        ("without the Claude trailer", [get(m, "without_trailer") for m in months], BLUE),
        ("with the Claude trailer", [get(m, "with_trailer") for m in months], ORANGE),
    ])
    ax.set_title("Commits per month", loc="left")
    ax.set_ylabel("commits")
    month_ticks(ax, months)
    ax.text(0, -0.2, "The trailer is regular from July 2026: before, it is a lower bound of the work of the agent. "
            + partial_month_note(months), transform=ax.transAxes, color=INK_2, fontsize=8.5)
    save(fig, "commits_per_month.png")


def chart_lines_changed() -> None:
    data = read("commits_per_month.csv")
    months = all_months(min(data), max(data))
    get = lambda m, k: int(data[m][k]) if m in data else 0  # noqa: E731
    fig, ax = plt.subplots(figsize=(9, 4.2))
    agent_band(ax, months)
    stacked_bars(ax, months, [
        ("code (nucs/)", [get(m, "lines_changed_code") for m in months], BLUE),
        ("tests", [get(m, "lines_changed_tests") for m in months], ORANGE),
        ("docs (.md, .rst)", [get(m, "lines_changed_docs") for m in months], AQUA),
    ])
    ax.set_title("Lines changed per month (added + deleted)", loc="left")
    ax.set_ylabel("lines")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    month_ticks(ax, months)
    ax.text(0, -0.2, "The data files (datasets/) are not counted. " + partial_month_note(months),
            transform=ax.transAxes, color=INK_2, fontsize=8.5)
    save(fig, "lines_changed_per_month.png")


def chart_tests() -> None:
    data = read("tests_per_month.csv")
    months = all_months(min(data), max(data))
    known = [i for i, m in enumerate(months) if m in data]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    agent_band(ax, months, label=False)
    ax.text(months.index(AGENT_START) - 0.2, 0.03, "with the agent", transform=ax.get_xaxis_transform(), color=INK_2,
            fontsize=9)
    values = [int(data[months[i]]["collected"]) for i in known]
    # A step line: the number of tests stays the same in the months with no commit.
    ax.step(known + [len(months) - 1], values + [values[-1]], where="post", color=BLUE, linewidth=2)
    ax.plot(known, values, "o", color=BLUE, markersize=4)
    first, last = months.index("2026-03"), len(months) - 1
    for i, offset, align in [(first, (0, 8), "center"), (last, (0, 8), "center")]:
        v = int(data[months[i]]["collected"])
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points", xytext=offset, ha=align, color=INK, fontsize=9)
    # The jump of September 2026 is mostly parametrized cases: say it, from the data.
    sep, aug = data["2026-09"], data["2026-08"]
    more_tests = int(sep["collected"]) - int(aug["collected"])
    more_functions = int(sep["test_functions"]) - int(aug["test_functions"])
    ax.annotate(f"Sep 2026: +{more_tests:,} tests from +{more_functions} test functions\n(mostly parametrized cases)",
                (months.index("2026-09"), 1500), textcoords="offset points", xytext=(-12, 0), ha="right", va="center",
                color=INK_2, fontsize=8.5)
    ax.set_title("Tests that pytest collects, at the end of each month", loc="left")
    ax.set_ylabel("tests (parametrized cases included)")
    month_ticks(ax, months)
    save(fig, "tests_per_month.png")


def chart_coverage() -> None:
    tests = read("tests_per_month.csv")
    data = read("coverage_per_month.csv")
    # Start one month before the first measure: the months before have no coverage data.
    first = all_months("2024-01", min(data))[-2]
    months = all_months(first, max(tests))
    known = [i for i, m in enumerate(months) if m in data]
    percent = [100 * (1 - int(data[months[i]]["missed"]) / int(data[months[i]]["statements"])) for i in known]
    statements = [int(data[months[i]]["statements"]) for i in known]

    def with_breaks(values: list[float]) -> tuple[list[float], list[float]]:
        """A NaN between two months that are not consecutive, so that no line joins months that were not measured."""
        xs, ys = [], []
        for n, (i, v) in enumerate(zip(known, values)):
            if n and i != known[n - 1] + 1:
                xs.append(i - 0.5)
                ys.append(float("nan"))
            xs.append(i)
            ys.append(v)
        return xs, ys

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax, values, fmt, title in [
        (top, percent, "{:.1f}%", "Line coverage of nucs/ by the tests"),
        (bottom, statements, "{:,}", "Size of nucs/ (statements)"),
    ]:
        agent_band(ax, months, label=ax is top)
        ax.plot(*with_breaks(values), color=BLUE, linewidth=2, marker="o", markersize=5)
        for i, v in [(known[0], values[0]), (known[-1], values[-1])]:
            ax.annotate(fmt.format(v), (i, v), textcoords="offset points", xytext=(0, 8), ha="center", color=INK,
                        fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
    top.set_ylim(min(percent) - 15, 100)
    bottom.set_ylim(0, max(statements) * 1.25)
    month_ticks(bottom, months)
    bottom.text(0, -0.42, "Measured without the JIT, as the CI does. February 2025 uses the dependencies of its time "
                "(Python 3.12, numba 0.61).", transform=bottom.transAxes, color=INK_2, fontsize=8.5)
    fig.suptitle("Coverage went up while the code grew", x=0.01, ha="left", fontsize=13, fontweight="bold")
    save(fig, "coverage_per_month.png")


if __name__ == "__main__":
    chart_commits()
    chart_lines_changed()
    chart_tests()
    chart_coverage()
