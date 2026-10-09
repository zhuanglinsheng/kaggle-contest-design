"""Rebuild hourly inputs from all submissions, preserving the focal pairs.

Recover pair IDs from joint public/private gap signatures, then verify every
distinct retained gap against source score pairs. Do not select new players.
Keep the legacy window and common prize units. Same-date records are ordered
by submission ID and assigned sorted random second offsets (UTC+8 legacy
timestamp convention). Source submission IDs, not dates, identify duplicates.
"""
import argparse
import json
import zipfile
from pathlib import Path
from shutil import copy2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pipeline.config import MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, RAW_DATA_ARCHIVE

AUDIT = MAIN_ESTIMATION_DIR.parent / "chapter5_score_audit"
FIELDS = ["PrivateScoreFullPrecision", "PublicScoreLeaderboardDisplay"]
PRIZE_CORRECTIONS_USD = {2489: 250, 3064: 150, 3353: 6000, 3366: 200,
                         3774: 250, 4493: 5000, 7115: 10000, 14420: 8000}

def logits(frame, signed=False):
    values = frame[FIELDS].to_numpy(float)
    if signed:
        values = (values + 1) / 2
    values = np.clip(values, .0001, .9999)
    return np.log(values / (1 - values))

def pair_candidates(frame, gaps):
    points = logits(frame)
    tree = cKDTree(points)
    teams = frame.TeamId.to_numpy(int)
    candidates = None
    for gap in gaps:
        pairs = set()
        for i, js in enumerate(tree.query_ball_point(points - gap, 1e-6)):
            pairs.update((int(teams[i]), int(teams[j])) for j in js if teams[i] != teams[j])
        candidates = pairs if candidates is None else candidates & pairs
        if len(candidates) <= 1:
            break
    return candidates

def identify_pair(frame, raw):
    last_day = frame.loc[frame.SubmissionDate == frame.groupby("TeamId").SubmissionDate.transform("max")]
    pairs = pair_candidates(last_day, [np.array([raw["real_y"][-1], raw["hat_y"][-1]])])
    if len(pairs) != 1:
        gaps = sorted(set(zip(raw["real_y"], raw["hat_y"])),
                      key=lambda x: abs(x[0]) + abs(x[1]), reverse=True)
        pairs = pair_candidates(frame, [np.array(gap) for gap in gaps])
    if len(pairs) != 1:
        raise ValueError(f"Ambiguous focal pair: {len(pairs)} candidates")
    pair = next(iter(pairs))
    for tid, key in zip(pair, ("hat_t_i", "hat_t_j")):
        dates = set(pd.to_datetime(raw["start_time"] + np.asarray(raw[key]) * 3600
                                  + 8 * 3600, unit="s").normalize())
        available = set(frame.loc[frame.TeamId == tid, "SubmissionDate"])
        if not dates <= available:
            raise ValueError("Recovered pair fails the legacy submission-date signature")
    first, second = (logits(frame.loc[frame.TeamId == tid]) for tid in pair)
    tree = cKDTree(second)
    for gap in set(zip(raw["real_y"], raw["hat_y"])):
        if not any(tree.query_ball_point(first - np.array(gap), 1e-6)):
            raise ValueError("Recovered pair does not reproduce a retained score-gap signature")
    return pair

def assign_times(frame, seed):
    frame = frame.sort_values(["SubmissionDate", "Id"]).copy()
    times = np.empty(len(frame))
    for _, positions in frame.groupby("SubmissionDate", sort=True).indices.items():
        seconds = np.sort(np.random.default_rng(seed).choice(86400, len(positions), replace=False))
        # The old files were exported using UTC+8 local timestamps.
        base = frame.iloc[positions[0]].SubmissionDate.timestamp() - 8 * 3600
        times[positions] = base + seconds
    frame["imputed_time"] = times
    return frame

def rebuild(frame, raw, pair, seed, metric):
    signed = any(word in metric.lower() for word in ("kappa", "matthews", "spearman", "pearson"))
    timed = assign_times(frame, seed)
    windows = [frame.loc[frame.TeamId == tid, "SubmissionDate"] for tid in pair]
    if any((dates.max() - dates.min()).days < 10 for dates in windows) or (
        min(dates.max() for dates in windows) - max(dates.min() for dates in windows)).days <= 10:
        raise ValueError("Recovered pair fails the source activity-window criterion")
    n = int(raw["N_Delta"])
    endpoints = float(raw["start_time"]) + np.arange(1, n + 1) * 3600
    values = []
    counts = []
    clip = {}
    first_joint = max(timed.loc[timed.TeamId == tid, "imputed_time"].min() for tid in pair)
    start_hour = max(23, int(np.floor((first_joint - raw["start_time"]) / 3600)))
    if start_hour >= n - 2:
        raise ValueError("Focal pair has no usable overlapping observation window")
    for tid in pair:
        records = timed.loc[timed.TeamId == tid].sort_values("imputed_time")
        times = records.imputed_time.to_numpy()
        positions = np.searchsorted(times, endpoints, side="right") - 1
        if (positions[start_hour:] < 0).any():
            raise ValueError("A focal player has no score at the observation-window start")
        source_scores = records[FIELDS].to_numpy(float)
        lower = -1 if signed else 0
        if ((source_scores < lower) | (source_scores > 1)).any():
            raise ValueError(f"Score outside the declared domain of {metric}")
        scores = logits(records, signed)
        values.append(scores[np.maximum(positions, 0)])
        observation_start = raw["start_time"] + start_hour * 3600
        accepted = (times > observation_start) & (times <= endpoints[-1])
        counts.append(((times[accepted] - raw["start_time"]) / 3600).tolist())
        used_positions = set(np.flatnonzero(accepted).tolist())
        initial_position = int(np.searchsorted(times, observation_start, side="right") - 1)
        if initial_position >= 0:
            used_positions.add(initial_position)
        used = records.iloc[sorted(used_positions)]
        if signed:
            used = used.copy()
            used[FIELDS] = (used[FIELDS] + 1) / 2
        clip[str(tid)] = {key: {"n": len(used), "lower": int((used[key] <= .0001).sum()),
                              "upper": int((used[key] >= .9999).sum()),
                              "below_half": int((used[key] < .5).sum())} for key in FIELDS}
    gaps = values[0] - values[1]
    gaps[:start_hour] = gaps[start_hour]
    new = dict(raw, Ni=len(counts[0]), Nj=len(counts[1]), hat_t_i=counts[0], hat_t_j=counts[1],
               real_y=gaps[:, 0].tolist(), hat_y=gaps[:, 1].tolist(),
               team_i=pair[0], team_j=pair[1], time_imputation_seed=seed, count_start_hour=start_hour,
               initialization_hours=start_hour, path_start_index=start_hour,
               score_metric=metric, signed_score_normalization=signed,
               preprocessing_version="all-submissions-endpoint-v1")
    if min(new["Ni"], new["Nj"]) < 5:
        raise ValueError("Restored focal pair fails minimum retained submissions")
    return new, clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=MAIN_DATA_DIR)
    args = parser.parse_args()
    source = pd.read_pickle(AUDIT / "submissions.pkl").drop_duplicates("Id").dropna(subset=FIELDS)
    ids = sorted(pd.read_csv(MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv").contest_id.unique())
    with zipfile.ZipFile(RAW_DATA_ARCHIVE) as archive:
        with archive.open("Competitions.csv") as stream:
            metrics = pd.read_csv(stream, usecols=["Id", "EvaluationAlgorithmName"]).set_index("Id").EvaluationAlgorithmName.to_dict()
    backup = AUDIT / "legacy_inputs"
    backup.mkdir(exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Validate the entire batch before replacing any canonical inputs.
    rebuilt = []
    for cid in ids:
        path = MAIN_DATA_DIR / f"contest_{cid}.json"
        old = backup / path.name
        if not old.exists():
            copy2(path, old)
        raw = json.loads(old.read_text())
        if int(cid) in PRIZE_CORRECTIONS_USD:
            raw["theta"] = PRIZE_CORRECTIONS_USD[int(cid)] / 1000
        frame = source.loc[source.CompetitionId == cid]
        pair = identify_pair(frame, raw)
        new, clip = rebuild(frame, raw, pair, args.seed, metrics[cid])
        rebuilt.append((path.name, new))
        manifest.append(dict(contest_id=int(cid), team_i=pair[0], team_j=pair[1],
                             old_counts=[raw["Ni"], raw["Nj"]], new_counts=[new["Ni"], new["Nj"]],
                             clipping=clip))
    for name, data in rebuilt:
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    (args.output_dir / "rebuild_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified and rebuilt {len(rebuilt)} focal pairs with seed {args.seed}", flush=True)

if __name__ == "__main__":
    main()
