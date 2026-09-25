"""Run and plot the revised 90-day PaperJK7 synthetic baseline."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from pipeline.config import CACHE_DIR, MATPLOTLIB_CACHE_DIR, PAPER_DIR, SYNTHETIC_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np

from synthetic_data.synthetic_data import synthetic_data_simulation


SUMMARY_PATH = SYNTHETIC_DIR / "baseline_summary.json"
DATA_PATH = SYNTHETIC_DIR / "baseline_data.json"
FIGURE_PATH = PAPER_DIR / "synthetic_data.pdf"


def plot_ts_events(
    time_grid,
    series,
    events,
    *,
    colors,
    linestyles,
    ylim=(-20, 25),
    figsize=(10, 5),
    height_ratios=(3, 1),
):
    """Reproduce the original notebook layout used for Figure 3."""
    fig, (ax_main, ax_events) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax_main.set_ylim(*ylim)
    for (label, values), color, linestyle in zip(
        series.items(), colors, linestyles
    ):
        ax_main.plot(
            time_grid,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
        )
    ax_main.legend(frameon=False)

    x_range = ax_main.get_xlim()
    event_height = 0.2
    event_row = 0.0
    event_ticks = []
    event_labels = []
    for label, event_times in events.items():
        ax_events.hlines(
            y=event_row,
            xmin=x_range[0],
            xmax=x_range[1],
            color="black",
        )
        ax_events.vlines(
            x=event_times,
            ymin=event_row - event_height,
            ymax=event_row + event_height,
            color="black",
        )
        event_ticks.append(event_row)
        event_labels.append(label)
        event_row -= 0.5
    ax_events.set_xlim(x_range)
    ax_events.get_xaxis().set_visible(False)
    ax_events.set_yticks(event_ticks)
    ax_events.set_yticklabels(event_labels)
    return fig


def main() -> None:
    start = datetime(2026, 1, 1)
    end = start + timedelta(days=90)
    result = synthetic_data_simulation(
        theta=1.0,
        c_i=1.2,
        c_j=1.5,
        sigma=2.0,
        lamb=1.0,
        intensity_effort_ratio=15.0,
        hour_arrival_ub=1.0,
        start_time=start,
        end_time=end,
        time_unit=timedelta(hours=1),
        time_unit_2f=1 / 24,
        mu_0=0.0,
        seed_brownian=1000,
        seed_poisson=1001,
        seed_uniform=1002,
        seed_initial=1003,
    )
    (
        time_grid,
        effort_i,
        effort_j,
        real_gap,
        perceived_gap,
        observed_gap,
        submissions_i,
        submissions_j,
    ) = result

    summary = {
        "initial_latent_gap": float(real_gap[0]),
        "submissions_i": len(submissions_i),
        "submissions_j": len(submissions_j),
        "mean_effort_i": float(effort_i.mean()),
        "mean_effort_j": float(effort_j.mean()),
        "finite_paths": bool(
            all(map(lambda x: bool((abs(x) < float("inf")).all()), [effort_i, effort_j, real_gap, perceived_gap, observed_gap]))
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    event_idx = np.sort(
        np.concatenate(
            [
                np.ceil([(t - start).total_seconds() / 3600 for t in submissions_i]).astype(int) + 1,
                np.ceil([(t - start).total_seconds() / 3600 for t in submissions_j]).astype(int) + 1,
            ]
        )
    )
    obs_idx = np.unique(event_idx)
    obs_h = np.diff(np.concatenate(([1], obs_idx))) / 24
    data = {
        "theta": 1.0,
        "ratio": 15.0,
        "estimate_r": 0,
        "N_Delta": len(time_grid),
        "Delta2f": 1 / 24,
        "Ni": len(submissions_i),
        "Nj": len(submissions_j),
        "hat_t_i": [(t - start).total_seconds() / 3600 for t in submissions_i],
        "hat_t_j": [(t - start).total_seconds() / 3600 for t in submissions_j],
        "hat_y": observed_gap.tolist(),
        "N_obs": len(obs_idx),
        "obs_idx": obs_idx.tolist(),
        "obs_h": obs_h.tolist(),
    }
    DATA_PATH.write_text(json.dumps(data, indent=2) + "\n")

    fig = plot_ts_events(
        time_grid,
        series={
            r"real output gap ($y_t$)": real_gap[:-1],
            r"observed output gap ($\hat{y}_t$)": observed_gap[:-1],
            r"perceived output gap ($\tilde{y}_t$)": perceived_gap[:-1],
        },
        events={r"player $i$": submissions_i, r"player $j$": submissions_j},
        colors=["blue", "black", "black"],
        linestyles=["solid", "solid", "dashed"],
    )
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(json.dumps(summary, indent=2))
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
