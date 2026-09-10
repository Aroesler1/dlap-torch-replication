"""Render the two deck charts from output/gan and output/ext_turnover results.json.

pptxgenjs can embed native OOXML charts, but only PowerPoint renders them - Keynote,
Google Slides and most PDF converters drop them silently. These PNGs are placed as
images instead, styled to the deck palette. No result is recomputed here: every value
is read straight out of results.json.

Usage:  python3 make_charts.py          (run from repo/deck, or anywhere - paths are resolved)
"""
import json, logging, os
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").disabled = True   # Calibri is absent off Windows; the fallback is fine
import matplotlib.pyplot as plt

NAVY, GOLD, GREY, LIGHT, RED = "#1E2A47", "#B8860B", "#5A6474", "#D8DEE9", "#9E2A2B"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
# repo/deck/make_charts.py -> repo/output
RES = os.path.join(os.path.dirname(HERE), "output")

plt.rcParams.update({
    "font.family": ["Calibri", "Helvetica Neue", "DejaVu Sans"],
    "axes.edgecolor": LIGHT, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "text.color": GREY, "axes.titlecolor": GREY,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)


def net_of_cost():
    """Annual test Sharpe vs one-way cost, baseline vs turnover-penalised. 3.95 x 2.05 in on the slide."""
    base = json.load(open(os.path.join(RES, "gan", "results.json")))["test"]["SR_annual_net_of_cost"]
    pen = json.load(open(os.path.join(RES, "ext_turnover", "results.json")))["test"]["SR_annual_net_of_cost"]
    x = [0, 10, 25, 50]
    yb = [base[f"{c}bps"] for c in x]
    yp = [pen[f"{c}bps"] for c in x]
    # cost at which the penalised model overtakes the baseline (linear between the 25 and 50 bp grid points)
    d25, d50 = yb[2] - yp[2], yb[3] - yp[3]
    cross = 25 + 25 * d25 / (d25 - d50)

    fig, ax = plt.subplots(figsize=(3.95, 2.05), dpi=300)
    ax.plot(x, yb, "-o", color=NAVY, linewidth=1.8, markersize=4, label="GAN (paper config)")
    ax.plot(x, yp, "-o", color=GOLD, linewidth=1.8, markersize=4, label="GAN + turnover penalty")
    ax.axvline(cross, color=GREY, linestyle="--", linewidth=0.8)
    ax.annotate(f"crossover\n≈{cross:.0f} bps", xy=(cross, 2.05), fontsize=6.5, color=GREY,
                ha="right", va="top", xytext=(cross - 1.5, 2.1))
    ax.set_title("Annual test Sharpe vs one-way cost", fontsize=8.5, pad=6)
    ax.set_xticks(x); ax.set_xticklabels([f"{c} bps" for c in x], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=1)
    style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "net_of_cost_sharpe.png"))
    plt.close(fig)
    return cross


def sharpe_by_year():
    """Annualised test Sharpe per calendar year. 6.3 x 3.0 in on the slide."""
    t = json.load(open(os.path.join(RES, "gan", "results.json")))["test"]
    ys = t["SR_annual_by_year"]
    yrs = sorted(ys)
    vals = [ys[y] for y in yrs]
    fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=300)
    ax.bar(range(len(yrs)), vals, color=[RED if v < 0 else NAVY for v in vals], width=0.72)
    ax.axhline(t["SR_annual"], color=GOLD, linestyle="--", linewidth=1.0)
    # the empty top-right of the panel is the only place this does not sit on a bar
    ax.text(0.60, 0.94, f"dashed line: full test window {t['SR_annual']:.2f}", transform=ax.transAxes,
            fontsize=7.5, color=GOLD, ha="left", va="top")
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xticks(range(len(yrs)))
    ax.set_xticklabels(yrs, rotation=90, fontsize=6.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title("Annualised test Sharpe, year by year", fontsize=9, pad=6)
    style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "sharpe_by_year.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c = net_of_cost()
    sharpe_by_year()
    print(f"wrote {OUT}/net_of_cost_sharpe.png (crossover {c:.1f} bps) and {OUT}/sharpe_by_year.png")
