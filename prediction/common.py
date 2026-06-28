from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "prediction" / "results"

# These contests have JSON/PKL files but are not in the current main-text
# 73-contest empirical table.
DEFAULT_EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}


@dataclass(frozen=True)
class ContestRecord:
    contest_id: int
    json_path: Path
    posterior_path: Path
    data: dict
    posterior: dict | None


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing contest_*.json and contest_*.pkl files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where prediction outputs are written.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Use all contests with JSON+PKL files instead of the 73-contest main sample.",
    )
    parser.add_argument(
        "--max-n-delta",
        type=int,
        default=2300,
        help="Maximum hourly grid length used to keep the main estimation sample.",
    )
    return parser


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_records(
    data_dir: Path = DATA_DIR,
    *,
    include_all: bool = False,
    max_n_delta: int = 2300,
    load_posterior: bool = False,
) -> list[ContestRecord]:
    records: list[ContestRecord] = []
    for json_path in sorted(data_dir.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue
        if not include_all and contest_id in DEFAULT_EXCLUDE_CONTESTS:
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data["N_Delta"] > max_n_delta:
            continue
        posterior = None
        if load_posterior:
            with posterior_path.open("rb") as f:
                posterior = pickle.load(f)
        records.append(
            ContestRecord(
                contest_id=contest_id,
                json_path=json_path,
                posterior_path=posterior_path,
                data=data,
                posterior=posterior,
            )
        )
    return records


def normalized_increment_volatility(values: list[float] | np.ndarray, delta: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    increments = np.diff(arr) / np.sqrt(delta)
    return float(np.std(increments, ddof=0))


def late_improvement(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    idx = int(np.floor(0.75 * (arr.size - 1)))
    return float(arr[-1] - arr[idx])


def build_base_frame(records: list[ContestRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        data = rec.data
        hat_y = np.asarray(data["hat_y"], dtype=float)
        real_y = np.asarray(data["real_y"], dtype=float)
        delta = float(data["Delta2f"])
        duration = float(data["N_Delta"]) * delta

        rows.append(
            {
                "contest_id": rec.contest_id,
                "G": float(real_y[-1]),
                "W": int(real_y[-1] > 0),
                "V": normalized_increment_volatility(real_y, delta),
                "final_public_gap": float(hat_y[-1]),
                "public_winner": int(hat_y[-1] > 0),
                "public_volatility": normalized_increment_volatility(hat_y, delta),
                "Ni": int(data["Ni"]),
                "Nj": int(data["Nj"]),
                "submission_imbalance": int(data["Ni"]) - int(data["Nj"]),
                "theta": float(data["theta"]),
                "duration": duration,
                "late_public_improvement": late_improvement(hat_y),
                "N_Delta": int(data["N_Delta"]),
            }
        )
    return pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)


def save_predictions(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_prediction_file(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    return pd.read_csv(path)
