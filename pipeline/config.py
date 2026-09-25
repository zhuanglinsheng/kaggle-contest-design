"""Canonical paths and units for the current PaperJK7 workflow.

All executable modules import paths from this file. The current workflow writes
only below ``results/``; historical outputs are stored under ``archive/``.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PAPER_DIR = ROOT / "paper"
METAKAGGLE_DIR = ROOT / "metakaggle"
SYNTHETIC_DIR = ROOT / "synthetic_data"

MAIN_DATA_DIR = METAKAGGLE_DIR / "__jsondata__"
ROBUSTNESS_DATA_DIR = METAKAGGLE_DIR / "__jsondata_robust23__"
RAW_DATA_ARCHIVE = METAKAGGLE_DIR / "__rawdata__" / "meta-kaggle.zip"

RESULTS_DIR = ROOT / "results"
MAIN_ESTIMATION_DIR = RESULTS_DIR / "estimation" / "main"
ROBUSTNESS_ESTIMATION_DIR = RESULTS_DIR / "estimation" / "robustness"
COUNTERFACTUAL_DIR = RESULTS_DIR / "counterfactual" / "main"
MAIN_VALIDATION_DIR = RESULTS_DIR / "validation" / "main"
ROBUSTNESS_VALIDATION_DIR = RESULTS_DIR / "validation" / "robustness"
SYNTHETIC_RESULTS_DIR = RESULTS_DIR / "synthetic"

CACHE_DIR = ROOT / ".cache"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"

EMPIRICAL_MODEL = METAKAGGLE_DIR / "real_data"
SYNTHETIC_SINGLE_MODEL = SYNTHETIC_DIR / "synthetic_data"
SYNTHETIC_POOLED_MODEL = SYNTHETIC_DIR / "synthetic_data_90_replicated"

# Source JSON prizes are measured in thousands of U.S. dollars.  The current
# paper expresses every prize in units of USD 100,000.
PRIZE_UNIT_USD = 100_000.0
PRIZE_DIVISOR_FROM_THOUSANDS = PRIZE_UNIT_USD / 1_000.0
