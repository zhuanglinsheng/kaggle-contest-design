"""Draw Proposition 1's sufficient regions, NOT computed equilibrium boundaries.

All coordinates below are illustrative layout coordinates. The horizontal
ordering sigma_L < sigma_U, the cost-dependent finite-precision certificate,
the small-risk prize-exit classification, the decreasing 9/16 entry band,
and the zero-prize sets represent proved relationships. Neither
c_H(sigma) nor c_e(sigma) is numerically evaluated by this script.
"""
from pathlib import Path
import tempfile
import os

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="joint-design-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # Ordinal placement of symbolic thresholds; no parameter calibration.
    sigma_l, sigma_u, x_end = 2.05, 3.45, 4.50
    cf, y_end = 1.0, 9.55
    x = np.linspace(0, x_end, 700)
    high = 8.78 + 0.09*np.sin(1.2*x)
    middle = np.linspace(sigma_l, sigma_u, 350)
    entry = 0.98-0.24*(middle-sigma_l)/(sigma_u-sigma_l)
    lower = (9/16)*entry
    costs = np.linspace(cf/2, y_end, 600)
    direct_width = 1.45*np.sqrt(np.maximum(0, 1-(costs-cf/2)/(6.7*cf)))
    zero_full_width = .08*np.exp(.16*(costs-cf/2))
    risk_width = np.maximum(direct_width, zero_full_width)
    exit_width = np.sqrt(np.maximum(0, (7.8*cf-costs)/3))
    finite_width = np.minimum(risk_width, exit_width)
    exit_risk = np.linspace(0, 1.45, 250)
    exit_cost = 7.8*cf-3*exit_risk**2
    assert 0 < sigma_l < sigma_u
    assert np.all(high > 6*cf)
    assert np.all(np.diff(entry) < 0)
    assert exit_cost[0] > 6*cf
    assert np.all((lower > 0) & (lower < entry))

    fig, ax = plt.subplots(figsize=(9.6, 6.5))
    finite_color, full_color, zero_color = "#4679ad", "#cc8c3a", "#60957a"
    ax.fill_between(x, high, y_end, color=zero_color, alpha=0.24, zorder=1)
    ax.fill_between(
        middle, entry, np.interp(middle, x, high),
        color=zero_color, alpha=0.24, zorder=1,
    )
    ax.fill_betweenx(costs, exit_width, risk_width,
                     where=(exit_width < risk_width) & (costs < high.min()),
                     color=zero_color, alpha=0.24, zorder=1)
    ax.fill_betweenx(costs, 0, finite_width,
                     color=finite_color, alpha=0.28, zorder=2)
    ax.fill_between(middle, lower, entry,
                    color=full_color, alpha=0.48, zorder=3)
    ax.plot(x, high, color="#3f5348", linestyle="--", linewidth=1.5)
    ax.plot(middle, entry, color="#3f5348", linewidth=1.4)
    ax.plot(middle, lower, color="#90632b", linewidth=1.2)
    for boundary in [sigma_l, sigma_u]:
        ax.axvline(boundary, color="#aeb3b7", linestyle=":", linewidth=1, zorder=0)
    ax.plot(risk_width, costs, color=finite_color, linewidth=1.2)
    ax.plot(exit_risk, exit_cost, color="#3f5348", linewidth=1.3)
    ax.plot([0, finite_width[0]], [cf/2, cf/2], color=finite_color, linewidth=1.1)

    ax.text(0.53, 3.2, "Positive prize\nFinite precision\n"+r"$0\leq\lambda^*<\infty$",
            ha="center", va="center", linespacing=1.5)
    ax.text(2.65, 4.9, "Zero prize\nPrecision irrelevant",
            ha="center", va="center", linespacing=1.5)
    ax.text(1.4, 9.14, "Zero prize for every risk level",
            ha="center", va="center")
    ax.text(1.65, 3.8, "Unclassified", ha="center", va="center",
            rotation=90, color="#62676b")
    ax.text(3.98, 3.8, "Unclassified", ha="center", va="center",
            rotation=90, color="#62676b")
    ax.text(0.56, 0.18, "Unclassified", ha="center", va="center",
            color="#62676b", fontsize=10.5)
    ax.annotate(
        r"$\sigma_F(c)$", xy=(risk_width[260], costs[260]),
        xytext=(1.32, 5.5), fontsize=12,
        arrowprops={"arrowstyle": "-", "color": finite_color, "lw": 0.9},
    )
    ax.annotate(
        r"Prize exit: $c_0(\sigma)$", xy=(exit_risk[50], exit_cost[50]),
        xytext=(0.65, 7.73), fontsize=11,
        arrowprops={"arrowstyle": "-", "color": "#3f5348", "lw": 0.9},
    )
    ax.annotate(
        r"Positive prize: $\lambda^*=\infty$",
        xy=(2.75, np.interp(2.75, middle, 0.5*(entry+lower))),
        xytext=(2.73, 0.13), ha="center", fontsize=11,
        arrowprops={"arrowstyle": "-", "color": "#90632b", "lw": 1},
    )
    ax.annotate(
        r"$c_e(\sigma)$", xy=(sigma_u, entry[-1]), xytext=(3.67, 0.91),
        va="center", fontsize=12,
        arrowprops={"arrowstyle": "-", "color": "#3f5348", "lw": 0.9},
    )
    ax.annotate(
        r"$\frac{9}{16}c_e(\sigma)$", xy=(sigma_u, lower[-1]),
        xytext=(3.69, 0.39), va="center", fontsize=12,
        arrowprops={"arrowstyle": "-", "color": "#90632b", "lw": 0.9},
    )
    ax.text(3.49, 8.51, r"$c_H(\sigma)$", ha="left", va="center")
    ax.set_xlim(0, x_end)
    ax.set_ylim(0, y_end)
    ax.set_xticks([0, sigma_l, sigma_u],
                  ["0", r"$\sigma_L$", r"$\sigma_U$"])
    ax.set_yticks([0, cf/2, 6*cf], ["0", r"$c_F/2$", r"$6c_F$"])
    ax.set_xlabel(r"Innovation risk $\sigma$", labelpad=9)
    ax.set_ylabel(r"Effort cost $c$", labelpad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(r"Sufficient regions in the $(c,\sigma)$ plane", pad=17)
    fig.text(0.5, 0.018, "Schematic only: positions and curve shapes are not numerical boundaries.",
             ha="center", color="#62676b", fontsize=10.5)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(output/"joint_design_regions.pdf", bbox_inches="tight")
    fig.savefig(output/"joint_design_regions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved schematic PDF and PNG; no equilibrium or boundary estimation performed.")


if __name__ == "__main__":
    main()
