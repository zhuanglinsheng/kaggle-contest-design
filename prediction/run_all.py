from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = sys.argv[1:]
    run("predict_fp.py", args)
    run("predict_rf.py", args)
    run("predict_structural.py", args)
    run("summarize_results.py", args)


if __name__ == "__main__":
    main()

