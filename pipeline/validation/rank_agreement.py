"""Reproduce Table 4 from official ranks, independent of structural fits."""
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pipeline.config import RAW_DATA_ARCHIVE, MAIN_ESTIMATION_DIR, MAIN_VALIDATION_DIR


def main():
    ids = set(pd.read_csv(MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv").contest_id)
    parts = []
    with zipfile.ZipFile(RAW_DATA_ARCHIVE) as archive:
        with archive.open("Teams.csv") as stream:
            for chunk in pd.read_csv(stream, chunksize=500000, usecols=[
                "Id", "CompetitionId", "IsBenchmark", "PublicLeaderboardRank", "PrivateLeaderboardRank"]):
                parts.append(chunk.loc[chunk.CompetitionId.isin(ids)])
    teams = pd.concat(parts, ignore_index=True)
    teams = teams.loc[teams.IsBenchmark.ne(True)].dropna(subset=["PublicLeaderboardRank", "PrivateLeaderboardRank"])
    teams = teams.loc[(teams.PublicLeaderboardRank > 0) & (teams.PrivateLeaderboardRank > 0)]
    rows = []
    for cid, work in teams.groupby("CompetitionId"):
        if len(work) < 5:
            raise RuntimeError(f"Too few jointly ranked teams: {cid}")
        pub, private = work.sort_values("PublicLeaderboardRank"), work.sort_values("PrivateLeaderboardRank")
        rows.append(dict(contest_id=int(cid), teams=len(work),
            rank_correlation=float(spearmanr(work.PublicLeaderboardRank, work.PrivateLeaderboardRank).statistic),
            top2_overlap=len(set(pub.Id.head(2)) & set(private.Id.head(2))) / 2,
            top5_overlap=len(set(pub.Id.head(5)) & set(private.Id.head(5))) / 5))
    frame = pd.DataFrame(rows)
    if set(frame.contest_id) != ids:
        raise RuntimeError("Rank-agreement sample differs from estimation sample")
    rng = np.random.default_rng(20261008)
    indices = rng.integers(0, len(frame), size=(10000, len(frame)))
    report = dict(seed=20261008, bootstrap_draws=10000, contests=len(frame), statistics={})
    for key in ("rank_correlation", "top2_overlap", "top5_overlap"):
        values = frame[key].to_numpy()
        ci = np.quantile(values[indices].mean(axis=1), [.025, .975])
        report["statistics"][key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                        ci_low=float(ci[0]), ci_high=float(ci[1]))
    frame.to_csv(MAIN_VALIDATION_DIR / "chapter5_rank_agreement.csv", index=False)
    (MAIN_VALIDATION_DIR / "chapter5_rank_agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
