from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"

SPLIT = 0.75
EPS = 1e-10
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}

EARLY_PARAMETER_FILES = [
    RESULTS_DIR / "structural_time_split_holdout_parameters.csv",
    RESULTS_DIR / "structural_time_split_holdout_sampling_parameters.csv",
    RESULTS_DIR / "structural_time_split_parameters_split_0.75.csv",
]
FULL_PARAMETER_FILES = [
    RESULTS_DIR / "structural_full_parameters.csv",
]
MIN_EARLY_PARAMETER_CONTESTS = 60


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_posterior(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def event_counts(times: list[float], n_delta: int) -> np.ndarray:
    counts = np.zeros(n_delta, dtype=int)
    for t in times:
        idx = int(math.ceil(float(t)))
        if 0 <= idx < n_delta:
            counts[idx] += 1
    return counts


def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), EPS, None)
    y = np.asarray(y, dtype=float)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def fn_gamma(u: float) -> float:
    u = float(np.clip(u, -0.999999, 0.999999))
    return u / (1.0 - u**2) + np.arctanh(u)


def fn_invgamma(x: float) -> float:
    return math.atan(0.856 * x) * 2.0 / math.pi


def safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -50.0, 50.0)))


def fn_efforts(
    y: float,
    t: float,
    deadline: float,
    theta: float,
    sigma: float,
    c_i: float,
    c_j: float,
) -> tuple[float, float]:
    remaining = max(deadline - t, 1e-8)
    sigma2 = sigma**2
    w_i = theta / (sigma2 * c_i)
    w_j = theta / (sigma2 * c_j)

    exp_w_i = safe_exp(w_i)
    exp_w_j = safe_exp(w_j)
    exp_neg_w_i = safe_exp(-w_i)
    exp_neg_w_j = safe_exp(-w_j)
    rho_i = (exp_w_i + exp_neg_w_j - 2.0) / max(exp_w_i - exp_neg_w_j, EPS)
    rho_j = (exp_w_j + exp_neg_w_i - 2.0) / max(exp_w_j - exp_neg_w_i, EPS)
    rho_i = float(np.clip(rho_i, -0.999999, 0.999999))
    rho_j = float(np.clip(rho_j, -0.999999, 0.999999))

    gamma_rho_i = fn_gamma(rho_i)
    gamma_rho_j = fn_gamma(rho_j)
    y_stderr = max(sigma * math.sqrt(remaining), 1e-8)
    z = y / y_stderr
    density_y = math.exp(-0.5 * z**2) / (math.sqrt(2.0 * math.pi) * y_stderr)
    loc = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * (gamma_rho_i + gamma_rho_j) - gamma_rho_j
    rho_z = fn_invgamma(loc)
    k = sigma2 / 2.0 * (gamma_rho_i + gamma_rho_j) * (1.0 - rho_z**2)
    m_i = max(density_y * k * (1.0 + rho_z), EPS)
    m_j = max(density_y * k * (1.0 - rho_z), EPS)
    return m_i, m_j


def structural_intensity_from_parameters(data: dict, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    n_delta = int(data["N_Delta"])
    delta2f = float(data["Delta2f"])
    deadline = n_delta * delta2f
    theta = float(data["theta"])
    hat_y = np.asarray(data["hat_y"], dtype=float)
    mu_0 = float(params["mu_0"])
    c_i = float(params["c_i"])
    c_j = float(params["c_j"])
    sigma = float(params["sigma"])
    lambda_ = float(params["lambda"])
    r = float(params["r"])

    tilde_y = np.zeros(n_delta + 1, dtype=float)
    intensity_i = np.zeros(n_delta, dtype=float)
    intensity_j = np.zeros(n_delta, dtype=float)
    tilde_y[0] = mu_0
    for h in range(n_delta):
        t = h * delta2f
        m_i, m_j = fn_efforts(tilde_y[h], t, deadline, theta, sigma, c_i, c_j)
        intensity_i[h] = r * m_i / 24.0
        intensity_j[h] = r * m_j / 24.0
        kalman_gain = math.sqrt(max(lambda_, EPS)) * sigma * (hat_y[h] - tilde_y[h])
        tilde_y[h + 1] = tilde_y[h] + (m_i - m_j + kalman_gain) * delta2f
        if not np.isfinite(tilde_y[h + 1]):
            tilde_y[h + 1] = tilde_y[h]
    return np.clip(intensity_i, EPS, None), np.clip(intensity_j, EPS, None)


def load_parameter_file() -> tuple[pd.DataFrame | None, Path | None, str | None]:
    for path in EARLY_PARAMETER_FILES:
        if path.exists():
            frame = pd.read_csv(path)
            required = {"contest_id", "mu_0", "c_i", "c_j", "sigma", "lambda", "r"}
            if required.issubset(frame.columns) and frame["contest_id"].nunique() >= MIN_EARLY_PARAMETER_CONTESTS:
                return frame, path, "early_75_parameters"
    for path in FULL_PARAMETER_FILES:
        if path.exists():
            frame = pd.read_csv(path)
            required = {"contest_id", "mu_0", "c_i", "c_j", "sigma", "lambda", "r"}
            if required.issubset(frame.columns):
                return frame, path, "full_parameters_internal"
    return None, None, None


def structural_intensity(data: dict, posterior: dict, early_params: dict | None) -> tuple[np.ndarray, np.ndarray, str]:
    if early_params is not None:
        intensity_i, intensity_j = structural_intensity_from_parameters(data, early_params)
        return intensity_i, intensity_j, "early_75_parameters"
    intensity_i = np.asarray(posterior["intensity_i"], dtype=float).mean(axis=0)
    intensity_j = np.asarray(posterior["intensity_j"], dtype=float).mean(axis=0)
    return np.clip(intensity_i, EPS, None), np.clip(intensity_j, EPS, None), "full_posterior_fallback"


def build_player_hours() -> tuple[pd.DataFrame, str, str | None]:
    parameter_frame, parameter_path, parameter_source = load_parameter_file()
    parameter_lookup = {}
    if parameter_frame is not None:
        parameter_lookup = {
            int(row.contest_id): {
                "mu_0": row.mu_0,
                "c_i": row.c_i,
                "c_j": row.c_j,
                "sigma": row.sigma,
                "lambda": row["lambda"],
                "r": row.r,
            }
            for _, row in parameter_frame.iterrows()
        }

    rows = []
    sources = set()
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue
        data = load_json(json_path)
        if int(data["N_Delta"]) > 2300:
            continue
        posterior = load_posterior(posterior_path)
        n_delta = int(data["N_Delta"])
        cutoff = int(math.floor(SPLIT * n_delta))
        counts_i = event_counts(data["hat_t_i"], n_delta)
        counts_j = event_counts(data["hat_t_j"], n_delta)
        early_i = counts_i[:cutoff].sum()
        early_j = counts_j[:cutoff].sum()
        early_rate_i = (early_i + EPS) / max(cutoff, 1)
        early_rate_j = (early_j + EPS) / max(cutoff, 1)
        pooled_rate = (early_i + early_j + EPS) / max(2 * cutoff, 1)
        intensity_i, intensity_j, source = structural_intensity(
            data, posterior, parameter_lookup.get(contest_id)
        )
        if parameter_lookup.get(contest_id) is not None and parameter_source is not None:
            source = parameter_source
        sources.add(source)
        hat_y = np.asarray(data["hat_y"], dtype=float)
        for player, counts, struct, early_rate in [
            ("i", counts_i, intensity_i, early_rate_i),
            ("j", counts_j, intensity_j, early_rate_j),
        ]:
            for h in range(cutoff, n_delta):
                remaining = n_delta - h
                rows.append(
                    {
                        "contest_id": contest_id,
                        "player": player,
                        "hour": h,
                        "heldout_count": int(counts[h]),
                        "structural_intensity": float(struct[h]),
                        "constant_player_intensity": float(early_rate),
                        "constant_contest_intensity": float(pooled_rate),
                        "abs_public_gap": float(abs(hat_y[h])),
                        "log_remaining": float(np.log1p(remaining)),
                        "late_share": float(h / max(n_delta, 1)),
                        "theta": float(data["theta"]),
                        "public_split": float(data["percentage"]),
                        "max_daily_submit": float(data["max_daily_submit"]),
                    }
                )
    source_label = "+".join(sorted(sources))
    parameter_label = str(parameter_path) if parameter_path is not None else None
    return pd.DataFrame(rows), source_label, parameter_label


def fit_gap_benchmark(df: pd.DataFrame) -> pd.Series:
    x = df[["abs_public_gap", "log_remaining", "late_share", "theta", "public_split", "max_daily_submit"]]
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = (x - x.mean()) / x.std(ddof=0).replace(0.0, 1.0)
    x = sm.add_constant(x)
    y = df["heldout_count"].astype(float)
    try:
        fit = sm.GLM(y, x, family=sm.families.Poisson()).fit(maxiter=200, disp=0)
        pred = fit.predict(x)
    except Exception:
        pred = np.repeat(float(y.mean()), len(y))
    return pd.Series(np.clip(pred, EPS, None), index=df.index)


def summarize_scores(df: pd.DataFrame, source_label: str, early_label: str | None) -> pd.DataFrame:
    model_columns = {
        "Structural effort intensity": "structural_intensity",
        "Player early-rate benchmark": "constant_player_intensity",
        "Contest early-rate benchmark": "constant_contest_intensity",
        "Public-gap Poisson benchmark": "gap_poisson_intensity",
    }
    rows = []
    y = df["heldout_count"].to_numpy(dtype=float)
    for model, column in model_columns.items():
        mu = df[column].to_numpy(dtype=float)
        ll = poisson_loglik(y, mu)
        rows.append(
            {
                "model": model,
                "n_player_hours": int(len(df)),
                "n_submissions": int(y.sum()),
                "mean_log_score": float(ll.mean()),
                "total_log_score": float(ll.sum()),
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "mean_prediction": float(mu.mean()),
                "source": source_label,
                "early_parameter_file": early_label or "",
            }
        )
    return pd.DataFrame(rows)


def structural_calibration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["structural_decile"] = pd.qcut(
        out["structural_intensity"].rank(method="first"), 10, labels=False
    ) + 1
    grouped = out.groupby("structural_decile", observed=False)
    return grouped.agg(
        n=("heldout_count", "size"),
        mean_predicted_intensity=("structural_intensity", "mean"),
        realized_submission_rate=("heldout_count", "mean"),
        total_submissions=("heldout_count", "sum"),
    ).reset_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, source_label, early_label = build_player_hours()
    df["gap_poisson_intensity"] = fit_gap_benchmark(df)
    summary = summarize_scores(df, source_label, early_label)
    calibration = structural_calibration(df)

    df.to_csv(RESULTS_DIR / "effort_intensity_cv_player_hours.csv", index=False)
    summary.to_csv(RESULTS_DIR / "effort_intensity_cv_summary.csv", index=False)
    calibration.to_csv(RESULTS_DIR / "effort_intensity_cv_calibration.csv", index=False)
    diagnostics = [
        f"split: {SPLIT}",
        f"source: {source_label}",
        f"parameter_file: {early_label or 'NOT FOUND'}",
        "required_early_parameter_columns: contest_id, mu_0, c_i, c_j, sigma, lambda, r",
        "warning: full_parameters_internal and full_posterior_fallback are internal diagnostics, not valid cross-validation results.",
    ]
    (RESULTS_DIR / "effort_intensity_cv_diagnostics.txt").write_text(
        "\n".join(diagnostics) + "\n", encoding="utf-8"
    )

    print(f"source: {source_label}")
    if early_label:
        print(f"parameter_file: {early_label}")
    else:
        print("parameter_file: NOT FOUND; used full posterior fallback")
    print(summary.to_string(index=False))
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
