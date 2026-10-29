"""Verify recovered original estimation targets and report summary drift.

Recovered chains are new draws from the original approximate posterior, not
the byte-identical historical draws. Median drift is descriptive, not a test
that the original posterior is correct or that its mean is precisely known.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical import estimate, unified_prior_reestimate as unified

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    manifest = {int(r['contest_id']): r for r in csv.DictReader(
        (unified.OUT / 'estimation/accepted_chain_sources.csv').open()) if r['mode'] == 'full'}
    summaries = {int(r['contest_id']): r for r in csv.DictReader(
        (unified.OUT / 'estimation/posterior_summary_final.csv').open()) if r['mode'] == 'full'}
    frozen = REVIEW / 'reconstruction_20261025/model_snapshot/detail_model'
    records, missing = [], []
    for cid, entry in sorted(manifest.items()):
        paths = [REVIEW / f'reconstruction_20261025/estimation/full/contest_{cid}/accepted.json',
                 HERE / f'posterior/contest_{cid}/accepted.json']
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            missing.append(cid)
            continue
        accepted = json.loads(path.read_text())
        config = accepted['configuration']
        raw = MAIN_DATA_DIR / f'contest_{cid}.json'
        data = estimate.stan_data(json.loads(raw.read_text()), 'full')
        expected_inputs = dict(data, **unified.prior(data))
        chain_paths = [Path(p) for p in accepted['chain_paths']]
        inputs = json.loads((chain_paths[0].parent / 'data.json').read_text())
        checks = dict(
            contest_and_mode=(config['contest_id'] == cid and config['mode'] == 'full'),
            raw_matches_historical_manifest=(digest(raw) == entry['input_sha256'] == config['raw_input_sha256']),
            frozen_model=(config['source_model_sha256'] == digest(frozen.with_suffix('.stan'))),
            frozen_executable=(config['executable_sha256'] == digest(frozen)),
            original_runner=(config['original_runner_sha256'] == digest(Path(unified.__file__))),
            original_stan_inputs_and_priors=(inputs == expected_inputs and config['prior'] == unified.prior(data)),
            accepted_diagnostics=bool(accepted['diagnostics']['passed']),
            four_chain_hashes=(len(chain_paths) == len(accepted['chain_sha256']) == 4 and all(
                p.is_file() and digest(p) == h for p, h in zip(chain_paths, accepted['chain_sha256']))),
        )
        changes = {}
        for name in ['c_i', 'c_j', 'sigma', 'lambda', 'mu_0', 'r']:
            old = float(summaries[cid][name + '_median'])
            new = accepted['summary'][name + '_median']
            scale = float(summaries[cid][name + '_sd']) if name == 'mu_0' else abs(old)
            changes[name] = dict(original_median=old, recovered_median=new,
                                 change_scaled=(new-old)/scale if scale > 0 else None,
                                 scale='historical posterior sd' if name == 'mu_0' else 'historical median')
        records.append(dict(contest_id=cid, accepted=str(path), checks=checks,
                            passed=all(checks.values()), descriptive_median_changes=changes))
    drift = {}
    for name in ['c_i', 'c_j', 'sigma', 'lambda', 'mu_0', 'r']:
        values = [abs(r['descriptive_median_changes'][name]['change_scaled']) for r in records
                  if r['descriptive_median_changes'][name]['change_scaled'] is not None]
        drift[name] = dict(median_absolute_scaled_change=float(np.median(values)),
                           maximum_absolute_scaled_change=max(values)) if values else None
    result = dict(expected_cases=len(manifest), verified_cases=len(records), missing=missing,
                  all73_original_targets_verified=len(records) == 73 and all(r['passed'] for r in records),
                  failed_cases=[r['contest_id'] for r in records if not r['passed']],
                  descriptive_summary_drift=drift, cases=records,
                  scope='Original one-order Bayesian posterior target and original inputs; '
                  'new chain realizations. No nonlinear Bayesian reestimation.')
    (HERE / 'posterior_target_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    return result


if __name__ == '__main__':
    audit()
