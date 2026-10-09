"""Recover candidate source records without changing the accepted sample.

The cache is an audit artifact, not a regenerated estimation input. Pair
identities must be verified against retained gaps before reporting clipping
counts as properties of the estimation sample.
"""
from pathlib import Path
import zipfile
import pandas as pd
from pipeline.config import RAW_DATA_ARCHIVE, MAIN_ESTIMATION_DIR

OUT = MAIN_ESTIMATION_DIR.parent / "chapter5_score_audit"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv").contest_id)
    with zipfile.ZipFile(RAW_DATA_ARCHIVE) as z:
        with z.open("Teams.csv") as stream:
            teams = pd.read_csv(stream, usecols=["Id", "CompetitionId", "PrivateLeaderboardRank"])
        teams = teams.loc[teams.CompetitionId.isin(ids)]
        teams.to_pickle(OUT / "teams.pkl")
        team_ids = set(teams.Id)
        parts = []
        with z.open("Submissions.csv") as stream:
            for chunk in pd.read_csv(stream, chunksize=500000, usecols=[
                "Id", "TeamId", "SubmissionDate", "IsAfterDeadline",
                "PublicScoreLeaderboardDisplay", "PrivateScoreFullPrecision"]):
                keep = chunk.TeamId.isin(team_ids) & ~chunk.IsAfterDeadline.fillna(False)
                if keep.any():
                    parts.append(chunk.loc[keep].copy())
        frame = pd.concat(parts, ignore_index=True)
        frame["SubmissionDate"] = pd.to_datetime(frame.SubmissionDate)
        frame = frame.merge(teams, left_on="TeamId", right_on="Id", suffixes=("", "_team"))
        frame.to_pickle(OUT / "submissions.pkl")
        print(f"Cached {len(frame)} source records for {len(ids)} contests", flush=True)

if __name__ == "__main__":
    main()
