"""Describe agreement between official public and private Kaggle leaderboards.

The analysis is restricted to the 73 contests in the main structural sample.
Statistics are first computed within each contest and then summarized across
contests with equal contest weights. Confidence intervals refer to the
across-contest means and are obtained by resampling contests.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
META_KAGGLE_ZIP = ROOT / "metakaggle" / "__rawdata__" / "meta-kaggle.zip"
SAMPLE_PATH = ROOT / "validation" / "results" / "structural_full_parameters.csv"
RESULTS_DIR = ROOT / "validation" / "results"
CONTEST_OUTPUT = RESULTS_DIR / "leaderboard_agreement_contest_level.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "leaderboard_agreement_summary.csv"

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260913


def main_sample_contest_ids() -> set[int]:
    sample = pd.read_csv(SAMPLE_PATH, usecols=["contest_id"])
    contest_ids = set(sample["contest_id"].astype(int))
    if len(contest_ids) != 73:
        raise ValueError(f"Expected 73 main-sample contests, found {len(contest_ids)}")
    return contest_ids


def load_official_ranks(contest_ids: set[int]) -> pd.DataFrame:
    """Stream the large Teams table and retain usable official ranks."""
    rows: list[dict[str, int]] = []
    with zipfile.ZipFile(META_KAGGLE_ZIP) as archive:
        with archive.open("Teams.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            for row in reader:
                try:
                    contest_id = int(row["CompetitionId"])
                except (TypeError, ValueError):
                    continue
                if contest_id not in contest_ids or row["IsBenchmark"].lower() == "true":
                    continue
                try:
                    public_rank = int(row["PublicLeaderboardRank"])
                    private_rank = int(row["PrivateLeaderboardRank"])
                    team_id = int(row["Id"])
                except (TypeError, ValueError):
                    continue
                if public_rank <= 0 or private_rank <= 0:
                    continue
                rows.append(
                    {
                        "contest_id": contest_id,
                        "team_id": team_id,
                        "public_rank": public_rank,
                        "private_rank": private_rank,
                    }
                )

    frame = pd.DataFrame(rows)
    observed_contests = set(frame["contest_id"].unique())
    missing = contest_ids - observed_contests
    if missing:
        raise ValueError(f"No usable official ranks for contests: {sorted(missing)}")
    return frame


def top_k_overlap(group: pd.DataFrame, k: int) -> float:
    if len(group) < k:
        raise ValueError(
            f"Contest {int(group['contest_id'].iloc[0])} has fewer than {k} ranked teams"
        )
    public_top = set(group.nsmallest(k, ["public_rank", "team_id"])["team_id"])
    private_top = set(group.nsmallest(k, ["private_rank", "team_id"])["team_id"])
    return len(public_top & private_top) / k


def contest_level_statistics(ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contest_id, group in ranks.groupby("contest_id", sort=True):
        correlation = stats.spearmanr(
            group["public_rank"], group["private_rank"]
        ).statistic
        rows.append(
            {
                "contest_id": int(contest_id),
                "n_ranked_teams": int(len(group)),
                "rank_correlation": float(correlation),
                "top_2_overlap_rate": top_k_overlap(group, 2),
                "top_5_overlap_rate": top_k_overlap(group, 5),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    bootstrap_means = np.empty(replications)
    for replication in range(replications):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[replication] = values[indices].mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def summary_statistics(contest_level: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "rank_correlation": "Public-private rank correlation",
        "top_2_overlap_rate": "Top-2 overlap rate",
        "top_5_overlap_rate": "Top-5 overlap rate",
    }
    rows = []
    for column, label in labels.items():
        values = contest_level[column].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(values)
        rows.append(
            {
                "statistic": label,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci_95_low_for_mean": ci_low,
                "ci_95_high_for_mean": ci_high,
                "n_contests": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    contest_ids = main_sample_contest_ids()
    ranks = load_official_ranks(contest_ids)
    contest_level = contest_level_statistics(ranks)
    if len(contest_level) != 73:
        raise ValueError(f"Expected 73 contest-level rows, found {len(contest_level)}")
    summary = summary_statistics(contest_level)

    contest_level.to_csv(CONTEST_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote contest-level results to {CONTEST_OUTPUT}")
    print(f"Wrote summary results to {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
