"""Stop this task's sampler driver after a specified subset is complete.

Retain every output file. Used only to rebalance the running correction batch;
the exact PID and command must match the requested output directory.
"""
import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path
from pipeline.config import MAIN_ESTIMATION_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--last-id", type=int, required=True)
    args = parser.parse_args()
    with (MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv").open() as handle:
        ids = {int(row["contest_id"]) for row in csv.DictReader(handle)
               if args.first_id <= int(row["contest_id"]) <= args.last_id}
    required = {(cid, mode) for cid in ids for mode in ("full", "early")}
    if not required:
        raise RuntimeError("Empty completion target")
    while True:
        command = subprocess.check_output(["ps", "-p", str(args.pid), "-o", "command="], text=True).strip()
        if "pipeline.empirical.estimate " not in command or f"--output-dir {args.directory} " not in command:
            raise RuntimeError("PID no longer matches this task's exact sampler driver")
        with (args.directory / "posterior_summary.csv").open() as handle:
            completed = {(int(row["contest_id"]), row["mode"]) for row in csv.DictReader(handle)}
        if required <= completed:
            os.kill(args.pid, signal.SIGSTOP)
            processes = subprocess.check_output(["ps", "-axo", "pid,ppid,command"], text=True)
            children = [int(line.split(None, 2)[0]) for line in processes.splitlines()[1:]
                        if len(line.split(None, 2)) == 3 and line.split(None, 2)[1] == str(args.pid)]
            for pid in children:
                os.kill(pid, signal.SIGTERM)
            os.kill(args.pid, signal.SIGTERM)
            os.kill(args.pid, signal.SIGCONT)
            print(f"Completed {len(required)} targeted fits; stopped driver {args.pid}; all files retained", flush=True)
            break
        time.sleep(.5)


if __name__ == "__main__":
    main()
