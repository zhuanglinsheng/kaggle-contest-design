from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"
OUTPUT_PATH = RESULTS_DIR / "structural_full_parameters.csv"

EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
MAX_N_DELTA = 2300
PARAMETERS = ["mu_0", "c_i", "c_j", "sigma", "lambda", "r"]


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data["N_Delta"]) > MAX_N_DELTA:
            continue
        with posterior_path.open("rb") as f:
            posterior = pickle.load(f)

        row = {
            "contest_id": contest_id,
            "split": 1.0,
            "full_N_Delta": int(data["N_Delta"]),
            "early_N_Delta": int(data["N_Delta"]),
            "early_Ni": int(data["Ni"]),
            "early_Nj": int(data["Nj"]),
        }
        for parameter in PARAMETERS:
            values = np.asarray(posterior[parameter], dtype=float)
            row[parameter] = float(np.mean(values))
            row[f"{parameter}_sd"] = float(np.std(values, ddof=1))
            row[f"{parameter}_q025"] = float(np.quantile(values, 0.025))
            row[f"{parameter}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(frame)} contests to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
