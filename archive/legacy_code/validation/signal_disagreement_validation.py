from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"
PAPER_DIR = ROOT / "paper"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))

from matplotlib import pyplot as plt

# JSON/PKL files exist for these contests, but they are not part of the
# current 73-contest main empirical table.
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def sign_array(values: np.ndarray) -> np.ndarray:
    return np.sign(values)


def load_validation_frame(
    data_dir: Path = DATA_DIR,
    *,
    max_n_delta: int = 2300,
) -> pd.DataFrame:
    rows = []
    for json_path in sorted(data_dir.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data["N_Delta"] > max_n_delta:
            continue

        with posterior_path.open("rb") as f:
            posterior = pickle.load(f)

        private_path = np.asarray(data["real_y"], dtype=float)
        public_path = np.asarray(data["hat_y"], dtype=float)
        diff = private_path - public_path
        path_discrepancy = float(np.sqrt(np.mean(diff**2)))
        path_discrepancy_l1 = float(np.mean(np.abs(diff)))
        path_sign_disagreement = float(np.mean(sign_array(private_path) != sign_array(public_path)))
        lambda_hat = float(np.mean(np.asarray(posterior["lambda"], dtype=float)))

        rows.append(
            {
                "contest_id": contest_id,
                "private_gap_T": float(private_path[-1]),
                "public_gap_T": float(public_path[-1]),
                "D_path": path_discrepancy,
                "D_path_l1": path_discrepancy_l1,
                "lambda_hat": lambda_hat,
                "log1p_D_path": float(np.log1p(path_discrepancy)),
                "log1p_D_path_l1": float(np.log1p(path_discrepancy_l1)),
                "log1p_lambda_hat": float(np.log1p(lambda_hat)),
            }
        )

    return pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)


def fit_discrepancy_model(df: pd.DataFrame, outcome: str = "log1p_D_path"):
    x = sm.add_constant(df["log1p_lambda_hat"])
    return sm.OLS(df[outcome], x).fit()


def model_row(model_name: str, fit, *, fit_stat_name: str) -> dict:
    coef = float(fit.params["log1p_lambda_hat"])
    stderr = float(fit.bse["log1p_lambda_hat"])
    pvalue = float(fit.pvalues["log1p_lambda_hat"])
    ci_low, ci_high = fit.conf_int().loc["log1p_lambda_hat"].tolist()
    if fit_stat_name == "R2":
        fit_stat = float(fit.rsquared)
    else:
        fit_stat = float(getattr(fit, "prsquared", np.nan))
    return {
        "outcome": model_name,
        "coefficient_log1p_lambda": coef,
        "std_error": stderr,
        "p_value": pvalue,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fit_stat_name": fit_stat_name,
        "fit_stat": fit_stat,
    }


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    ols = fit_discrepancy_model(df)
    rows.append(model_row("log(1 + D_path_l)", ols, fit_stat_name="R2"))
    ols_l1 = fit_discrepancy_model(df, "log1p_D_path_l1")
    rows.append(model_row("log(1 + D_path_l1)", ols_l1, fit_stat_name="R2"))

    spearman_d = stats.spearmanr(df["lambda_hat"], df["D_path"], nan_policy="omit")
    spearman_d_l1 = stats.spearmanr(df["lambda_hat"], df["D_path_l1"], nan_policy="omit")

    diagnostics = "\n".join(
        [
            f"n_contests: {len(df)}",
            f"mean_D_path: {df['D_path'].mean():.6g}",
            f"median_D_path: {df['D_path'].median():.6g}",
            f"spearman_lambda_D_path: rho={spearman_d.statistic:.6g}, p={spearman_d.pvalue:.6g}",
            f"spearman_lambda_D_path_l1: rho={spearman_d_l1.statistic:.6g}, p={spearman_d_l1.pvalue:.6g}",
        ]
    )
    return pd.DataFrame(rows), diagnostics


def save_disagreement_plot(df: pd.DataFrame, fit, output_path: Path) -> None:
    x = df["log1p_lambda_hat"].to_numpy(dtype=float)
    y = df["log1p_D_path"].to_numpy(dtype=float)
    order = np.argsort(x)
    x_grid = x[order]
    x_design = sm.add_constant(pd.Series(x_grid, name="log1p_lambda_hat"))
    y_fit = fit.predict(x_design)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(x, y, s=20, color="black", label="Contest")
    ax.plot(x_grid, y_fit, color="black", linestyle="--", label="OLS Fit")
    ax.set_xlabel(r"$\log(1+\hat\lambda)$")
    ax.set_ylabel(r"$\log(1+D^{path})$")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_validation_frame()
    summary, diagnostics = summarize(df)
    fit = fit_discrepancy_model(df)

    df.to_csv(RESULTS_DIR / "signal_disagreement_contest_level.csv", index=False)
    summary.to_csv(RESULTS_DIR / "signal_disagreement_summary.csv", index=False)
    (RESULTS_DIR / "signal_disagreement_diagnostics.txt").write_text(
        diagnostics + "\n",
        encoding="utf-8",
    )
    save_disagreement_plot(df, fit, PAPER_DIR / "validate_signal_disagreement.pdf")

    print(summary.to_string(index=False))
    print()
    print(diagnostics)
    print(f"\nWrote results to {RESULTS_DIR}")
    print(f"Wrote figure to {PAPER_DIR / 'validate_signal_disagreement.pdf'}")


if __name__ == "__main__":
    main()
