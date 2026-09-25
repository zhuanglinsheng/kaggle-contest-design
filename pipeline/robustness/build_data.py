#!/usr/bin/env python3
"""Build the second-versus-third-place robustness sample.

The output uses the same 73 contests, hourly score construction, score
transformation, and prize data as the baseline empirical sample.  Contestants
are selected by their official final private-leaderboard ranks in the Meta
Kaggle Teams table.
"""

from __future__ import annotations

import json
import zipfile
import pandas as pd

from metakaggle._data_clean import contest_basic_setting, save_contest_data
from pipeline.config import (
    MAIN_DATA_DIR,
    RAW_DATA_ARCHIVE,
    ROBUSTNESS_DATA_DIR,
    ROOT,
)


RAW_ZIP = RAW_DATA_ARCHIVE
BASELINE_DIR = MAIN_DATA_DIR
OUTPUT_DIR = ROBUSTNESS_DATA_DIR
MANIFEST = OUTPUT_DIR / "pair_manifest.csv"
CHUNK_SIZE = 500_000


def baseline_ids() -> list[int]:
    return sorted(
        int(path.stem.split("_")[-1])
        for path in BASELINE_DIR.glob("contest_*.json")
    )


def read_zip_table(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with archive.open(name) as stream:
        return pd.read_csv(stream, low_memory=False, **kwargs)


def main() -> None:
    contest_ids = baseline_ids()
    contest_set = set(contest_ids)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW_ZIP) as archive:
        contests = read_zip_table(
            archive,
            "Competitions.csv",
            usecols=[
                "Id",
                "DeadlineDate",
                "MaxDailySubmissions",
                "LeaderboardPercentage",
                "RewardQuantity",
            ],
            parse_dates=["DeadlineDate"],
        )
        contests = contests.loc[contests["Id"].isin(contest_set)].copy()

        teams = read_zip_table(
            archive,
            "Teams.csv",
            usecols=["Id", "CompetitionId", "PrivateLeaderboardRank"],
        )
        candidates = teams.loc[
            teams["CompetitionId"].isin(contest_set)
            & teams["PrivateLeaderboardRank"].between(1, 20)
        ].copy()
        duplicate_ranks = candidates.duplicated(
            ["CompetitionId", "PrivateLeaderboardRank"], keep=False
        )
        if duplicate_ranks.any():
            raise ValueError(
                "Duplicate final private ranks: "
                + candidates.loc[duplicate_ranks].to_string(index=False)
            )
        candidate_counts = candidates.groupby("CompetitionId").size()
        missing = [cid for cid in contest_ids if candidate_counts.get(cid, 0) < 3]
        if missing:
            raise ValueError(f"Contests with fewer than three ranked candidates: {missing}")

        selected_team_ids = set(candidates["Id"].astype(int))
        selected_chunks: list[pd.DataFrame] = []
        with archive.open("Submissions.csv") as stream:
            reader = pd.read_csv(
                stream,
                usecols=[
                    "Id",
                    "TeamId",
                    "SubmissionDate",
                    "IsAfterDeadline",
                    "PublicScoreLeaderboardDisplay",
                    "PrivateScoreFullPrecision",
                ],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                keep = chunk["TeamId"].isin(selected_team_ids)
                if keep.any():
                    selected_chunks.append(chunk.loc[keep].copy())

    submissions = pd.concat(selected_chunks, ignore_index=True)
    submissions = submissions.loc[~submissions["IsAfterDeadline"].fillna(False)].copy()
    submissions["SubmissionDate"] = pd.to_datetime(submissions["SubmissionDate"])
    submissions.rename(
        columns={
            "PublicScoreLeaderboardDisplay": "PublicScore",
            "PrivateScoreFullPrecision": "PrivateScore",
        },
        inplace=True,
    )
    submissions.drop(columns="IsAfterDeadline", inplace=True)
    submissions.dropna(subset=["PublicScore", "PrivateScore"], inplace=True)
    submissions.drop_duplicates(subset=["SubmissionDate", "TeamId"], inplace=True)
    submissions = submissions.merge(
        candidates[["Id", "CompetitionId"]],
        how="left",
        left_on="TeamId",
        right_on="Id",
        suffixes=("", "_Team"),
    )
    submissions.drop(columns="Id_Team", inplace=True)

    manifest_rows = []
    for contest_id in contest_ids:
        deadline, prize, max_daily, percentage = contest_basic_setting(
            contests, contest_id
        )
        contest_submissions = submissions.loc[
            submissions["CompetitionId"] == contest_id
        ]
        candidate_table = candidates.loc[
            candidates["CompetitionId"] == contest_id
        ].sort_values("PrivateLeaderboardRank")
        eligible = []
        for team_id in candidate_table["Id"].astype(int):
            dates = contest_submissions.loc[
                contest_submissions["TeamId"] == team_id, "SubmissionDate"
            ].sort_values()
            if len(dates) < 5:
                continue
            if (dates.iloc[-1] - dates.iloc[0]).days < 10:
                continue
            eligible.append(team_id)
        output_path = OUTPUT_DIR / f"contest_{contest_id}.json"
        if output_path.exists():
            output_path.unlink()
        if eligible is None or len(eligible) < 3:
            manifest_rows.append(
                {
                    "contest_id": contest_id,
                    "rank_2_team_id": None,
                    "rank_3_team_id": None,
                    "rank_2_private_rank": None,
                    "rank_3_private_rank": None,
                    "rank_2_submissions": 0,
                    "rank_3_submissions": 0,
                    "output_created": False,
                }
            )
            continue
        selected_pair = None
        for left in range(1, len(eligible) - 1):
            for right in range(left + 1, len(eligible)):
                candidate_i = int(eligible[left])
                candidate_j = int(eligible[right])
                dates_i = contest_submissions.loc[
                    contest_submissions["TeamId"] == candidate_i, "SubmissionDate"
                ].sort_values()
                dates_j = contest_submissions.loc[
                    contest_submissions["TeamId"] == candidate_j, "SubmissionDate"
                ].sort_values()
                overlap_start = max(dates_i.iloc[0], dates_j.iloc[0])
                overlap_end = min(dates_i.iloc[-1], dates_j.iloc[-1])
                if (overlap_end - overlap_start).days < 10:
                    continue
                save_contest_data(
                    submissions,
                    contest_id,
                    candidate_i,
                    candidate_j,
                    deadline,
                    float(prize),
                    int(max_daily),
                    float(percentage),
                    "Percentage_Small",
                    str(OUTPUT_DIR.relative_to(ROOT)),
                )
                if output_path.exists():
                    selected_pair = (candidate_i, candidate_j)
                    break
            if selected_pair is not None:
                break
        if selected_pair is None:
            manifest_rows.append(
                {
                    "contest_id": contest_id,
                    "rank_2_team_id": None,
                    "rank_3_team_id": None,
                    "rank_2_private_rank": None,
                    "rank_3_private_rank": None,
                    "rank_2_submissions": 0,
                    "rank_3_submissions": 0,
                    "output_created": False,
                }
            )
            continue
        team_i, team_j = selected_pair
        contest_candidates = candidates.loc[
            candidates["CompetitionId"] == contest_id
        ].set_index("Id")
        private_rank_i = int(contest_candidates.loc[team_i, "PrivateLeaderboardRank"])
        private_rank_j = int(contest_candidates.loc[team_j, "PrivateLeaderboardRank"])
        count_i = int((contest_submissions["TeamId"] == team_i).sum())
        count_j = int((contest_submissions["TeamId"] == team_j).sum())
        manifest_rows.append(
            {
                "contest_id": contest_id,
                "rank_2_team_id": team_i,
                "rank_3_team_id": team_j,
                "rank_2_private_rank": private_rank_i,
                "rank_3_private_rank": private_rank_j,
                "rank_2_submissions": count_i,
                "rank_3_submissions": count_j,
                "output_created": output_path.exists(),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST, index=False)
    created = int(manifest["output_created"].sum())
    print(f"Created {created} of {len(contest_ids)} contest files in {OUTPUT_DIR}")
    if created != len(contest_ids):
        failed = manifest.loc[~manifest["output_created"], "contest_id"].tolist()
        raise RuntimeError(f"Robustness data construction failed for: {failed}")


if __name__ == "__main__":
    main()
