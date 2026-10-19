"""Fresh, resumable NUTS estimates under the frozen empirical lambda prior.

Original accepted estimates and research experiments are read-only inputs.
Run: python -m pipeline.empirical.unified_prior_reestimate --workers 2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

from pipeline.config import ROOT, MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, manifest_path, resolve_manifest_path
from pipeline.empirical import estimate as e
from pipeline.empirical import nonlinear_reestimate as n

OUT = ROOT / "results/reestimation/lambda100_sd225"
MODEL = ROOT / "results/validation/estimation_details/detail_model"
NAMES = ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prior(data):
    return dict(prior_mean=[0., 0., math.log(2), math.log(100), data["hat_y"][0], math.log(10)],
                prior_sd=[.75, .75, .5, 2.25, 1., .5], sigma_floor=.01)


def baseline(cid, mode):
    manifest = MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"
    row = next(row for row in csv.DictReader(manifest.open())
               if int(row["contest_id"]) == cid and row["mode"] == mode)
    run = resolve_manifest_path(row["run_dir"], manifest)
    raw = MAIN_DATA_DIR / f"contest_{cid}.json"
    if digest(raw) != row["input_sha256"]:
        raise RuntimeError("Original input mismatch")
    return run, raw


def initializations(cid, mode, previous=None):
    if previous is not None:
        paths = previous
    else:
        run, _ = baseline(cid, mode)
        paths = [run / f"chain_{j}.csv" for j in range(1, 5)]
    starts = []
    candidate = ROOT / f"results/validation/sigma_lambda_trials/posterior/empirical/{cid}/lambda_external/contest_{cid}"
    for j, path in enumerate(paths, 1):
        chain = e.read_columns(path)
        init = {name: float(np.median(chain[name])) for name in e.PARAMETERS}
        # Existing fresh, same-prior fits are useful starting points, not reused draws.
        if previous is None and mode == "full" and (candidate / f"chain_{j}.npz").exists():
            z = np.median(np.load(candidate / f"chain_{j}.npz")["z"], axis=0)
            init = dict(zip(e.PARAMETERS, np.r_[np.exp(z[:4]), z[4:]].tolist()))
        elif previous is None:
            init["lambda"] = max(.011, init["lambda"] * (1, 3, 10, 30)[j-1])
        for key in ("c_i", "c_j", "sigma", "lambda"):
            init[key] = max(.0101, init[key])
        init["log_r"] = max(math.log(.01)+.001, init["log_r"])
        starts.append(init)
    return starts


def diagnose(cid, mode, paths, data):
    row = e.summarize(cid, mode, paths)
    chains = [e.read_columns(path) for path in paths]
    arrays = {name: np.asarray([np.exp(c["log_r"]) if name == "r" else c[name] for c in chains]) for name in NAMES}
    arrays["w_i"] = data["theta"] / (arrays["sigma"]**2 * arrays["c_i"])
    arrays["w_j"] = data["theta"] / (arrays["sigma"]**2 * arrays["c_j"])
    arrays["max_w"] = np.maximum(arrays["w_i"], arrays["w_j"])
    diagnostics = {}
    for name, values in arrays.items():
        bulk, tail = n.rank_ess(values)
        ess = n.effective_sample_size(values)
        sd = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        mcse = sd / math.sqrt(max(ess, 1))
        diagnostics[name] = dict(rank_rhat=float(n.rank_rhat(values)), bulk_ess=float(bulk), tail_ess=float(tail),
                                 mean_mcse=mcse, relative_mean_mcse=mcse/abs(mean) if mean else None)
        row[f"{name}_median"] = float(np.median(values))
        if name in NAMES:
            row[f"{name}_rank_rhat"] = diagnostics[name]["rank_rhat"]
            row[f"{name}_bulk_ess"] = float(bulk)
            row[f"{name}_tail_ess"] = float(tail)
            row[f"{name}_mean_mcse"] = mcse
    saturated = sum(int(np.sum(c["treedepth__"] >= 13)) for c in chains)
    energy_bfmi = [float(np.mean(np.diff(c["energy__"])**2)/np.var(c["energy__"])) for c in chains]
    passed = (row["divergences"] == 0 and saturated == 0 and min(energy_bfmi) >= .3
              and all(v["rank_rhat"] <= 1.01 and min(v["bulk_ess"], v["tail_ess"]) >= 400 for v in diagnostics.values())
              and all(diagnostics[p]["relative_mean_mcse"] <= .03 for p in ("sigma", "lambda")))
    return row, dict(passed=bool(passed), quantities=diagnostics, saturated_transitions=saturated, energy_bfmi=energy_bfmi)


def fit_one(task):
    cid, mode, pilot = task
    run, rawpath = baseline(cid, mode)
    data = e.stan_data(json.loads(rawpath.read_text()), mode)
    inputs = dict(data, **prior(data))
    config = dict(contest_id=cid, mode=mode, inputs=inputs, original_input_sha256=digest(rawpath),
                  model_sha256=digest(MODEL.with_suffix(".stan")), executable_sha256=digest(MODEL),
                  runner_sha256=digest(__file__), sampler="CmdStan NUTS", chains=4)
    case = OUT / "estimation" / mode / f"contest_{cid}"
    signature = case / "configuration.json"
    if signature.exists() and json.loads(signature.read_text()) != config:
        raise RuntimeError(f"Frozen configuration changed: {cid} {mode}")
    save(signature, config)
    accepted = case / "accepted.json"
    if accepted.exists():
        result = json.loads(accepted.read_text())
        if all(digest(Path(p)) == h for p,h in zip(result["chain_paths"], result["chain_sha256"])):
            return result
        raise RuntimeError("Accepted chain checksum mismatch")
    schedule = ((400, 600, .97),) if pilot else ((1000, 2000, .98), (2000, 4000, .99), (4000, 8000, .995))
    previous = None
    for attempt, (warmup, samples, delta) in enumerate(schedule, 1):
        folder = case / (("pilot" if pilot else "attempt") + str(attempt))
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / "data.json", inputs)
        starts = initializations(cid, mode, previous)
        processes = []
        paths = [folder / f"chain_{j}.csv" for j in range(1,5)]
        for j, path in enumerate(paths, 1):
            save(folder / f"init_{j}.json", starts[j-1])
            if e.csv_complete(path, samples):
                continue
            log = (folder / f"chain_{j}.log").open("w")
            command = [str(MODEL), "sample", f"num_warmup={warmup}", f"num_samples={samples}",
                       "adapt", f"delta={delta}", "algorithm=hmc", "engine=nuts", "max_depth=13", "metric=dense_e",
                       "random", f"seed={20261015+cid+(100000 if mode=='early' else 0)}", f"id={j}",
                       "data", f"file={folder/'data.json'}", "output", f"file={path}", "refresh=100",
                       "sig_figs=18", f"init={folder/f'init_{j}.json'}"]
            processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")), log))
        failures = []
        for process, log in processes:
            failures.append(process.wait()); log.close()
        if any(failures):
            raise RuntimeError(f"CmdStan failed: {cid} {mode} {failures}")
        row, diagnostics = diagnose(cid, mode, paths, data)
        result = dict(summary=row, diagnostics=diagnostics, chain_paths=[str(p) for p in paths],
                      chain_sha256=[digest(p) for p in paths], attempt=attempt, configuration=config)
        save(folder / "diagnostics.json", result)
        print(f"FIT {cid} {mode} attempt={attempt} passed={diagnostics['passed']} "
              f"rhat={max(v['rank_rhat'] for v in diagnostics['quantities'].values()):.4f} "
              f"minESS={min(min(v['bulk_ess'],v['tail_ess']) for v in diagnostics['quantities'].values()):.0f} "
              f"div={row['divergences']}", flush=True)
        if diagnostics["passed"] and not pilot:
            save(accepted, result)
            return result
        previous = paths
    if pilot:
        return result
    raise RuntimeError(f"Fit did not meet frozen diagnostic gates: {cid} {mode}")


def assemble():
    manifest = MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"
    required = {(int(r['contest_id']), r['mode']) for r in csv.DictReader(manifest.open())}
    rows, sources = [], []
    for cid, mode in sorted(required):
        path = OUT / "estimation" / mode / f"contest_{cid}/accepted.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if not result['diagnostics']['passed']:
            raise RuntimeError("Rejected fit in accepted file")
        for p,h in zip(result['chain_paths'], result['chain_sha256']):
            if digest(p) != h:
                raise RuntimeError("Changed accepted chain")
        row = dict(result['summary'], deadline_preserved=1, input_sha256=result['configuration']['original_input_sha256'])
        rows.append(row)
        raw = MAIN_DATA_DIR / f"contest_{cid}.json"
        sourcefile = OUT / "estimation/accepted_chain_sources.csv"
        sources.append(dict(contest_id=cid,mode=mode,accepted_source="lambda100_sd225_nuts",
                            run_dir=manifest_path(Path(result['chain_paths'][0]).parent, sourcefile),
                            input_sha256=digest(raw),raw_input_path=manifest_path(raw,sourcefile)))
    for filename, values in (("posterior_summary_final.csv",rows),("accepted_chain_sources.csv",sources)):
        if values:
            efields = list(values[0])
            with (OUT/'estimation'/filename).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=efields);writer.writeheader();writer.writerows(values)
    save(OUT/'progress.json',dict(accepted=len(rows),required=len(required),complete=len(rows)==len(required)))
    print(f"ACCEPTED {len(rows)}/{len(required)}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--contests',type=int,nargs='+')
    parser.add_argument('--modes',nargs='+',choices=('full','early'),default=['full','early'])
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--assemble-only',action='store_true')
    args=parser.parse_args()
    if args.assemble_only:
        assemble();return
    ids=sorted({int(r['contest_id']) for r in csv.DictReader((MAIN_ESTIMATION_DIR/'accepted_chain_sources.csv').open())})
    if args.contests:
        ids=[cid for cid in ids if cid in args.contests]
    tasks=[(cid,mode,args.pilot) for cid in ids for mode in args.modes]
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fit_one,task):task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(dict(task=futures[future],error=str(error)))
                print(f"FAILED {futures[future]}: {error}",flush=True)
            if not args.pilot:
                assemble()
    save(OUT/('pilot_failures.json' if args.pilot else 'failures.json'),failures)
    if failures:
        raise RuntimeError(f"{len(failures)} fits failed; see failures.json")


if __name__=='__main__':
    main()
