#!/usr/bin/env python3
"""Run the current USD-100,000 estimator for the robustness focal pairs."""

from __future__ import annotations

import sys

from pipeline.config import ROBUSTNESS_DATA_DIR, ROBUSTNESS_ESTIMATION_DIR
from pipeline.empirical import estimate as estimation


estimation.DATA_DIR = ROBUSTNESS_DATA_DIR
estimation.OUTPUT_DIR = ROBUSTNESS_ESTIMATION_DIR
estimation.EXCLUDED = set()


if __name__ == "__main__":
    if "--modes" not in sys.argv:
        sys.argv.extend(["--modes", "full"])
    estimation.main()
