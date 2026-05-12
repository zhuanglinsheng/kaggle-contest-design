#!/usr/bin/env python3
"""Compute counterfactual statistics matching Table tab-counterfactual-policy
in PaperJK.tex, for both joint optimisation and prize-only optimisation.

Reads the CSVs from counterfactual_results/ and counterfactual_prize_result/,
computes the same statistics as the paper's table, and appends a formatted
summary to the respective summary .md files.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parents[1]


def load_ok_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("status") == "ok"]


def ci_mean(x: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    n = len(x)
    m = float(np.mean(x))
    if np.std(x) == 0:
        return m, math.nan, math.nan
    se = float(scipy_stats.sem(x, nan_policy="omit"))
    lo, hi = scipy_stats.t.interval(confidence, df=n - 1, loc=m, scale=se)
    return m, lo, hi


def one_sided_ttest(series: np.ndarray, *, alternative: str = "greater") -> dict:
    """One-sample t-test that the mean is > 0 (greater) or < 0 (less)."""
    if np.std(series) == 0:
        return {"t_stat": math.nan, "p_value": math.nan, "n": len(series)}
    res = scipy_stats.ttest_1samp(series, 0.0, alternative=alternative)
    return {"t_stat": res.statistic, "p_value": res.pvalue, "n": len(series)}


def wilcoxon_p(series: np.ndarray, *, alternative: str = "greater") -> float:
    """Wilcoxon signed-rank p-value (one-sided)."""
    if np.std(series) == 0 or np.all(series == 0):
        return math.nan
    res = scipy_stats.wilcoxon(series, alternative=alternative)
    return res.pvalue


def build_stat_block(
    label: str,
    effort_pct: np.ndarray,
    prize_pct: np.ndarray,
    effort_abs: np.ndarray | None = None,
    prize_abs: np.ndarray | None = None,
) -> str:
    """Return a Markdown table section."""
    lines: list[str] = []
    lines.append(f"## {label}\n")

    # Compute statistics
    e_pos = int(np.sum(effort_pct > 0))
    p_pos = int(np.sum(prize_pct > 0))
    e_mean, e_ci_lo, e_ci_hi = ci_mean(effort_pct)
    p_mean, p_ci_lo, p_ci_hi = ci_mean(prize_pct)
    e_median = float(np.median(effort_pct))
    p_median = float(np.median(prize_pct))

    e_ttest = one_sided_ttest(effort_pct, alternative="greater")
    p_ttest = one_sided_ttest(prize_pct, alternative="greater")

    e_wilcox = wilcoxon_p(effort_pct, alternative="greater")
    p_wilcox = wilcoxon_p(prize_pct, alternative="greater")

    e_min = float(np.min(effort_pct))
    p_min = float(np.min(prize_pct))
    e_max = float(np.max(effort_pct))
    p_max = float(np.max(prize_pct))

    lines.append(
        "| Statistic | Effort improvement | Prize saving |\n"
        "|---|---|---|"
    )
    lines.append(
        f"| Contests with positive margin | {e_pos}/73 | {p_pos}/73 |"
    )
    lines.append(
        f"| Mean percentage change | {e_mean:.2f}% | {p_mean:.2f}% |"
    )
    lines.append(
        f"| Median percentage change | {e_median:.2f}% | {p_median:.2f}% |"
    )
    lines.append(
        f"| 95% CI for mean | [{e_ci_lo:.2f}%, {e_ci_hi:.2f}%] | "
        f"[{p_ci_lo:.2f}%, {p_ci_hi:.2f}%] |"
    )
    lines.append(
        f"| One-sided $t$-statistic | {e_ttest['t_stat']:.2f} | "
        f"{p_ttest['t_stat']:.2f} |"
    )
    lines.append(
        f"| One-sided $t$-test $p$-value | {e_ttest['p_value']:.2e} | "
        f"{p_ttest['p_value']:.2e} |"
    )
    lines.append(
        f"| Wilcoxon signed-rank $p$-value | {e_wilcox:.2e} | "
        f"{p_wilcox:.2e} |"
    )
    lines.append(
        f"| Minimum percentage change | {e_min:.2f}% | {p_min:.2f}% |"
    )
    lines.append(
        f"| Maximum percentage change | {e_max:.2f}% | {p_max:.2f}% |"
    )

    if effort_abs is not None and prize_abs is not None:
        ea_mean, ea_lo, ea_hi = ci_mean(effort_abs)
        pa_mean, pa_lo, pa_hi = ci_mean(prize_abs)
        lines.append(
            f"| Mean absolute change | {ea_mean:.2f} | {pa_mean:.2f} |"
        )
        lines.append(
            f"| 95% CI for absolute change | [{ea_lo:.2f}, {ea_hi:.2f}] | "
            f"[{pa_lo:.2f}, {pa_hi:.2f}] |"
        )

    return "\n".join(lines)


def main() -> None:
    # ---- Joint optimisation ----
    joint_csv = REPO / "counterfactual_results" / "counterfactual_summary.csv"
    joint_md = REPO / "counterfactual_results" / "counterfactual_summary.md"

    if joint_csv.exists():
        rows = load_ok_rows(joint_csv)
        effort_pct = np.array([float(r["M_max_change_pct"]) for r in rows])
        prize_pct = np.array([float(r["theta_reduction_pct_best"]) for r in rows])

        # Absolute effort change = M_max - M0
        effort_abs = np.array([
            float(r["M_max_with_theta_cap"]) - float(r["M0"]) for r in rows
        ])
        prize_abs = np.array([
            float(r["theta0"]) - float(r["theta_best"]) for r in rows
        ])

        block = build_stat_block(
            "Joint optimisation (theta + T)",
            effort_pct, prize_pct,
            effort_abs, prize_abs,
        )

        # Append to the existing md file
        existing = joint_md.read_text(encoding="utf-8")
        joint_md.write_text(
            existing.rstrip() + "\n\n" + block + "\n",
            encoding="utf-8",
        )
        print(f"Joint stats appended to {joint_md}")
    else:
        print(f"Joint CSV not found: {joint_csv}")

    # ---- Prize-only optimisation ----
    prize_csv = REPO / "counterfactual_prize_result" / "prize_only_summary.csv"
    prize_md = REPO / "counterfactual_prize_result" / "prize_only_summary.md"

    if prize_csv.exists():
        rows = load_ok_rows(prize_csv)

        # Effort improvement at fixed T: (M_max_at_T0 - M0)/M0
        effort_pct = np.array([
            100.0 * (float(r["M_max_at_T0"]) - float(r["M0"])) / float(r["M0"])
            if float(r["M0"]) != 0 else math.nan
            for r in rows
        ])
        prize_pct_raw = np.array([float(r["theta_reduction_pct_best"]) for r in rows])
        strict = np.array([
            r.get("strict_theta_reduction", "False") == "True" for r in rows
        ])

        effort_abs = np.array([
            float(r["M_max_at_T0"]) - float(r["M0"]) for r in rows
        ])
        prize_abs = np.array([
            float(r["theta0"]) - float(r["theta_best"]) for r in rows
        ])

        # Drop NaN (shouldn't have any, but be safe)
        valid = ~np.isnan(effort_pct) & ~np.isnan(prize_pct_raw)
        effort_pct = effort_pct[valid]
        prize_pct_raw = prize_pct_raw[valid]
        effort_abs = effort_abs[valid]
        prize_abs = prize_abs[valid]
        strict = strict[valid]

        # Use strict_theta_reduction to define positive-margin prize saving;
        # raw prize_pct can be micro-positively due to binary-search precision
        prize_pct: np.ndarray = np.where(strict, prize_pct_raw, 0.0)

        block = build_stat_block(
            "Prize-only optimisation (T fixed at T0)",
            effort_pct, prize_pct,
            effort_abs, prize_abs,
        )

        existing = prize_md.read_text(encoding="utf-8")
        prize_md.write_text(
            existing.rstrip() + "\n\n" + block + "\n",
            encoding="utf-8",
        )
        print(f"Prize-only stats appended to {prize_md}")
    else:
        print(f"Prize CSV not found: {prize_csv}")


if __name__ == "__main__":
    main()
