"""Verify the portable PaperJK7 release and record compact artifact hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from pipeline.config import (
    COUNTERFACTUAL_DIR,
    MAIN_DATA_DIR,
    MAIN_ESTIMATION_DIR,
    MAIN_VALIDATION_DIR,
    PAPER_DIR,
    RESULTS_DIR,
    ROOT,
    resolve_manifest_path,
)


RELEASE_MANIFEST = RESULTS_DIR / "release_manifest.json"
PAPER_GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")

COMPACT_ARTIFACTS = (
    "requirements.txt",
    "requirements-lock.txt",
    "Makefile",
    "README.md",
    "docs/REPRODUCIBILITY.md",
    "pipeline/README.md",
    "results/README.md",
    "paper/PaperJK7.tex",
    "paper/_Literatures.bib",
    "paper/appendix_100k_rows.tex",
    "paper/figure3.png",
    "paper/numerical_regions_bw.pdf",
    "paper/validate_lamb_revised_100k.pdf",
    "paper/validate_sigma_revised_100k.pdf",
    "paper/validate_lambda_pair.pdf",
    "paper/validate_sigma_pair.pdf",
    "results/estimation/main/posterior_summary_final.csv",
    "results/estimation/main/accepted_chain_sources.csv",
    "results/counterfactual/main/counterfactual_summary.csv",
    "results/counterfactual/main/precision_recommendations.csv",
    "results/counterfactual/main/precision_policy_contests.csv",
    "results/counterfactual/main/precision_policy_summary.json",
    "results/validation/main/chapter5_contest_level.csv",
    "results/validation/main/chapter5_regressions.csv",
    "results/validation/main/chapter5_final_audit.json",
    "results/validation/main/chapter5_paper_numbers.json",
    "results/validation/main/chapter5_signal_robust_regressions.csv",
    "results/validation/main/chapter5_signal_cook_regressions.csv",
    "results/robustness/chapter6/reports/pair_L1_L2/report.json",
    "results/robustness/chapter6/reports/pair_L1_L2/regressions.csv",
    "results/robustness/chapter6/reports/pair_L1_L2/scale_summary.csv",
    "results/robustness/chapter6/posterior_summary.csv",
    "results/robustness/chapter6/fit_config.json",
    "results/robustness/chapter6/scale_calibration.json",
    "results/robustness/chapter6/pair_manifest.csv",
    "results/robustness/chapter6/accepted/pair/accepted_chain_sources.csv",
    "results/robustness/chapter6/accepted/L1/accepted_chain_sources.csv",
    "results/robustness/chapter6/accepted/L2/accepted_chain_sources.csv",
    "results/synthetic/optimal_hourly/manifest.json",
    "results/theory/numerical_joint_plane_moderate_dense/three_class_comparison/metadata.json",
)

COPY_PAIRS = (
    ("results/synthetic/optimal_hourly/nonsteady/figure3.png", "paper/figure3.png"),
    ("results/theory/numerical_joint_plane_moderate_dense/three_class_comparison/numerical_regions_bw.pdf", "paper/numerical_regions_bw.pdf"),
    ("results/robustness/chapter6/reports/pair_L1_L2/validate_lambda_pair.pdf", "paper/validate_lambda_pair.pdf"),
    ("results/robustness/chapter6/reports/pair_L1_L2/validate_sigma_pair.pdf", "paper/validate_sigma_pair.pdf"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Missing release artifact: {path.relative_to(ROOT)}")
    return path


def verify_main_estimation() -> dict[str, int | float]:
    manifest_path = require(MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(int(row["contest_id"]), row["mode"]) for row in rows}
    contests = {key[0] for key in keys}
    expected = {(contest, mode) for contest in contests for mode in ("full", "early")}
    if len(contests) != 73 or len(rows) != 146 or keys != expected:
        raise RuntimeError("Main accepted-fit manifest must contain 73 matched full/early pairs")
    if any(Path(row["run_dir"]).is_absolute() or Path(row["raw_input_path"]).is_absolute()
           for row in rows):
        raise RuntimeError("Accepted-fit manifest contains non-portable absolute paths")
    for row in rows:
        contest = int(row["contest_id"])
        run = resolve_manifest_path(row["run_dir"], manifest_path)
        snapshot = resolve_manifest_path(row["raw_input_path"], manifest_path)
        current = require(MAIN_DATA_DIR / f"contest_{contest}.json")
        expected_hash = row["input_sha256"]
        if sha256(require(snapshot)) != expected_hash or sha256(current) != expected_hash:
            raise RuntimeError(f"Input checksum mismatch for contest {contest}")
        require(run / "data.json")
        for chain in range(1, 5):
            require(run / f"chain_{chain}.csv")
    summary = pd.read_csv(require(MAIN_ESTIMATION_DIR / "posterior_summary_final.csv"))
    if len(summary) != 146 or summary.duplicated(["contest_id", "mode"]).any():
        raise RuntimeError("Final posterior summary is not a unique 73-by-2 panel")
    if int(summary["divergences"].sum()) != 0:
        raise RuntimeError("Accepted posterior summary contains divergent transitions")
    rhat = float(summary.filter(regex="_rhat$").max().max())
    if rhat > 1.01:
        raise RuntimeError(f"Accepted posterior summary has R-hat {rhat:.5f} > 1.01")
    return {"contests": len(contests), "accepted_fits": len(rows), "maximum_rhat": rhat}


def verify_derived_results() -> dict[str, int]:
    contests = pd.read_csv(require(MAIN_VALIDATION_DIR / "chapter5_contest_level.csv"))
    regressions = pd.read_csv(require(MAIN_VALIDATION_DIR / "chapter5_regressions.csv"))
    policies = pd.read_csv(require(COUNTERFACTUAL_DIR / "precision_policy_contests.csv"))
    if len(contests) != 73 or contests["contest_id"].nunique() != 73:
        raise RuntimeError("Chapter 5 contest-level output must contain 73 contests")
    if len(regressions) != 6 or regressions["specification"].nunique() != 6:
        raise RuntimeError("Chapter 5 output must contain six declared regressions")
    if len(policies) != 73 or policies["contest_id"].nunique() != 73:
        raise RuntimeError("Precision-policy output must contain 73 contests")
    cook = pd.read_csv(require(MAIN_VALIDATION_DIR / "chapter5_signal_cook_regressions.csv"))
    source_hash = sha256(MAIN_VALIDATION_DIR / "chapter5_contest_level.csv")
    if set(cook["input_sha256"]) != {source_hash}:
        raise RuntimeError("Cook-distance sensitivity output was generated from stale inputs")
    return {"chapter5_contests": len(contests), "chapter5_regressions": len(regressions),
            "precision_policies": len(policies)}


def verify_paper_assets() -> dict[str, int]:
    source = require(PAPER_DIR / "PaperJK7.tex").read_text()
    graphics = PAPER_GRAPHIC_RE.findall(source)
    for name in graphics:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"Paper graphic is outside paper/: {name}")
        require(PAPER_DIR / name)
    for source_name, paper_name in COPY_PAIRS:
        source_path, paper_path = require(ROOT / source_name), require(ROOT / paper_name)
        if sha256(source_path) != sha256(paper_path):
            raise RuntimeError(f"Paper copy is stale: {paper_name}")
    require(PAPER_DIR / "PaperJK7.pdf")
    return {"paper_graphics": len(graphics), "canonical_copies": len(COPY_PAIRS)}


def verify_chapter6() -> dict[str, int | float]:
    root = RESULTS_DIR / "robustness" / "chapter6"
    summary = pd.read_csv(require(root / "posterior_summary.csv"))
    selected = summary.loc[summary["scenario"].isin(("pair", "L1", "L2"))]
    if len(selected) != 219 or selected.duplicated(["scenario", "contest_id"]).any():
        raise RuntimeError("Chapter 6 release must contain 73 fits for each of pair, L1, and L2")
    if int(selected["divergences"].sum()) or int(selected["saturated_transitions"].sum()):
        raise RuntimeError("Chapter 6 release contains rejected transitions")
    rhat = float(selected.filter(regex="_rhat$").max().max())
    if rhat >= 1.01:
        raise RuntimeError(f"Chapter 6 release has R-hat {rhat:.5f} >= 1.01")
    for scenario in ("pair", "L1", "L2"):
        manifest_path = require(root / "accepted" / scenario / "accepted_chain_sources.csv")
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 73 or len({int(row["contest_id"]) for row in rows}) != 73:
            raise RuntimeError(f"Chapter 6 {scenario} manifest must contain 73 unique contests")
        if any(Path(row["run_dir"]).is_absolute() for row in rows):
            raise RuntimeError(f"Chapter 6 {scenario} manifest contains absolute paths")
        for row in rows:
            contest = int(row["contest_id"])
            run = resolve_manifest_path(row["run_dir"], manifest_path)
            input_path = require(root / "inputs" / scenario / f"contest_{contest}.json")
            if sha256(input_path) != row["input_sha256"]:
                raise RuntimeError(f"Chapter 6 input checksum mismatch: {scenario} {contest}")
            require(run / "data.json")
            for chain in range(1, 5):
                require(run / f"chain_{chain}.csv")
    return {"accepted_fits": len(selected), "maximum_rhat": rhat}


def artifact_manifest() -> dict:
    paths = [require(ROOT / name) for name in COMPACT_ARTIFACTS]
    paths.extend(sorted((ROOT / "pipeline").glob("**/*.py")))
    paths.extend(sorted((ROOT / "tests").glob("test_*.py")))
    paths.extend(sorted((ROOT / "metakaggle").glob("*.py")))
    paths.extend(sorted((ROOT / "metakaggle").glob("*.stan")))
    paths.extend(sorted((ROOT / "synthetic_data").glob("*.py")))
    paths.extend(sorted((ROOT / "synthetic_data").glob("*.stan")))
    chapter6 = RESULTS_DIR / "robustness" / "chapter6"
    for scenario in ("pair", "L1", "L2"):
        paths.extend(sorted((chapter6 / "inputs" / scenario).glob("contest_*.json")))
    accepted_manifests = [MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"]
    accepted_manifests.extend(
        chapter6 / "accepted" / scenario / "accepted_chain_sources.csv"
        for scenario in ("pair", "L1", "L2")
    )
    for manifest in accepted_manifests:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if manifest == MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv":
                    paths.append(require(MAIN_DATA_DIR / f"contest_{int(row['contest_id'])}.json"))
                    paths.append(require(resolve_manifest_path(row["raw_input_path"], manifest)))
                run = resolve_manifest_path(row["run_dir"], manifest)
                paths.append(require(run / "data.json"))
                paths.extend(require(run / f"chain_{chain}.csv") for chain in range(1, 5))
    paths.extend(require(PAPER_DIR / name) for name in
                 PAPER_GRAPHIC_RE.findall((PAPER_DIR / "PaperJK7.tex").read_text()))
    paths.extend(require(PAPER_DIR / name) for name in
                 ("informs3.cls", "informs2014.bst", "endnotes.sty"))
    unique = sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)))
    return {
        "schema_version": 1,
        "scope": "PaperJK7 release; accepted CmdStan chains are hashed here but remain external large artifacts",
        "files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in unique
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="replace results/release_manifest.json after successful verification")
    parser.add_argument("--check-manifest", action="store_true",
                        help="require current compact hashes to equal the saved release manifest")
    args = parser.parse_args()
    report = {
        "main_estimation": verify_main_estimation(),
        "derived_results": verify_derived_results(),
        "chapter6": verify_chapter6(),
        "paper": verify_paper_assets(),
    }
    current = artifact_manifest()
    if args.check_manifest:
        saved = json.loads(require(RELEASE_MANIFEST).read_text())
        if saved != current:
            raise RuntimeError("Release manifest is stale; regenerate outputs and run --write-manifest")
    if args.write_manifest:
        RELEASE_MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
