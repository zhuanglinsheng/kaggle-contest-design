"""Compile the three Stan models used by the current pipeline."""

import subprocess
from pathlib import Path

from pipeline.config import METAKAGGLE_DIR, SYNTHETIC_DIR


MODEL_FILES = (
    METAKAGGLE_DIR / "real_data.stan",
    SYNTHETIC_DIR / "synthetic_data.stan",
    SYNTHETIC_DIR / "synthetic_data_90_replicated.stan",
)


def main() -> None:
    try:
        from cmdstanpy import CmdStanModel
    except ModuleNotFoundError:
        installations = sorted((Path.home() / ".cmdstan").glob("cmdstan-*"))
        if not installations:
            raise RuntimeError(
                "cmdstanpy is not installed and no local CmdStan installation was found"
            )
        cmdstan_dir = installations[-1]
        for stan_file in MODEL_FILES:
            print(f"Compiling {stan_file.relative_to(stan_file.parents[1])}")
            subprocess.run(
                ["make", str(stan_file.with_suffix(""))],
                cwd=cmdstan_dir,
                check=True,
            )
    else:
        for stan_file in MODEL_FILES:
            print(f"Compiling {stan_file.relative_to(stan_file.parents[1])}")
            CmdStanModel(stan_file=str(stan_file))


if __name__ == "__main__":
    main()
