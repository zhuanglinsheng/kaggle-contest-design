"""Fresh JK9 posterior reruns from frozen inputs, without historical chains.

This reproduces the target/protocol, not the historical chain bytes or a table
that depends on those particular finite Monte Carlo draws. Failed sampling
attempts remain visible; no fit is accepted merely because sampling completed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil

from pipeline.publication import ROOT, digest, local, write_json
from pipeline.storage import BUILD, RUNS, compile_stan


def plan(scenario, mode, contest=None):
    index = ROOT / 'results/releases/paperjk7-review-20261016/accepted_fits.json'
    fits = json.loads(index.read_text())['fits']
    rows = [r for r in fits if r['scenario'] == scenario and r['mode'] == mode
            and (contest is None or r['contest_id'] == contest)]
    if not rows:
        raise ValueError('No frozen input matches this scenario/mode/contest')
    for row in rows:
        path = local(ROOT, row['input_snapshot'])
        if digest(path) != row['input_sha256']:
            raise ValueError(f'Changed frozen input: {path}')
    return rows


def training_data(raw, mode, cutoff=None):
    from pipeline.empirical.estimate import stan_data
    data = stan_data(raw, mode if cutoff is None else 'full')
    if cutoff is not None:
        if not raw.get('count_start_hour', 23) < cutoff < raw['N_Delta']:
            raise ValueError('Training cutoff must be after initialization and before deadline')
        data['N_Delta'] = cutoff
        for player in ('i', 'j'):
            data['hat_t_'+player] = [t for t in data['hat_t_'+player] if t <= cutoff]
            data['N'+player] = len(data['hat_t_'+player])
        data['hat_y'] = data['hat_y'][:cutoff]
    return data


def diagnose_free(cid, mode, paths, data):
    import numpy as np
    from pipeline.empirical import estimate as e, nonlinear_reestimate as n
    from pipeline.empirical.unified_prior_medians import diagnose
    summary, diagnostics = diagnose(cid, mode, paths, data)
    columns = [e.read_columns(p) for p in paths]
    arrays = {k: np.asarray([c[k] for c in columns]) for k in ('deadline_power','gap_log_scale')}
    arrays['gap_scale'] = np.exp(arrays['gap_log_scale'])
    for name, values in arrays.items():
        bulk, tail = n.rank_ess(values)
        rhat = float(n.rank_rhat(values))
        diagnostics['quantities'][name] = dict(rank_rhat=rhat, bulk_ess=float(bulk), tail_ess=float(tail))
        diagnostics['passed'] = bool(diagnostics['passed'] and np.isfinite(values).all()
                                      and rhat <= 1.01 and min(bulk,tail) >= 400)
        summary[name+'_median'] = float(np.median(values))
    return summary, diagnostics



COST_SCENARIOS = {
    'common_sd_100': (1., .01),
    'cost_floor_0001': (.75/math.sqrt(2), .001),
}


# Position in the original 19-task direct-fit schedule.
COST_TASK_INDEX = {
    'common_sd_100': {2464: 3, 2551: 4, 5144: 5, 8219: 10},
    'cost_floor_0001': {2464: 14, 2551: 15, 5144: 16, 8219: 18},
}


def cost_prior(scenario):
    sd, floor = COST_SCENARIOS[scenario]
    return dict(common_log_mean=0., common_log_sd=sd, cost_floor=floor,
                relative_log_sd=.75*math.sqrt(2))


def diagnose_cost(cid, mode, paths, data):
    import numpy as np
    from pipeline.empirical import estimate as e, nonlinear_reestimate as n
    from pipeline.empirical.unified_prior_medians import diagnose

    summary, diagnostics = diagnose(cid, mode, paths, data)
    columns = [e.read_columns(p) for p in paths]
    ci, cj = [np.log([chain[key] for chain in columns]) for key in ('c_i','c_j')]
    relative = ci-cj
    for name, values in dict(common_log_cost=(ci+cj)/2, relative_log_cost=relative,
                             cost_ratio=np.exp(relative)).items():
        bulk, tail = n.rank_ess(values)
        rhat = float(n.rank_rhat(values))
        mcse = float(values.std(ddof=1)/math.sqrt(max(n.effective_sample_size(values), 1)))
        mean = float(values.mean())
        diagnostics['quantities'][name] = dict(rank_rhat=rhat, bulk_ess=float(bulk),
            tail_ess=float(tail), mean_mcse=mcse, relative_mean_mcse=mcse/abs(mean) if mean else None)
        diagnostics['passed'] = bool(diagnostics['passed'] and np.isfinite(values).all()
                                    and rhat <= 1.01 and min(bulk,tail) >= 400)
        summary.update({name+'_median':float(np.median(values)), name+'_mean':mean,
                        name+'_mean_mcse':mcse})
    return summary, diagnostics


def fit(row, output, *, model_kind='D', cutoff=None, split=None, cost_scenario=None, baseline=None):
    import numpy as np
    from pipeline.empirical import estimate as estimator
    from pipeline.empirical import unified_prior_reestimate as original
    from pipeline.empirical.unified_prior_medians import diagnose

    case = (output / model_kind / split / str(row['contest_id']) if split else
            output / row['scenario'] / row['mode'] / str(row['contest_id']))
    if cost_scenario:
        if row['scenario']!='main' or row['mode']!='full' or cutoff is not None or split or model_kind!='D':
            raise ValueError('Cost-prior refits require a main, full-window leading target')
        case = output / 'fits' / cost_scenario / str(row['contest_id'])
    case.mkdir(parents=True, exist_ok=True)
    accepted = case / 'accepted.json'
    if accepted.exists():
        raise FileExistsError(f'Preserve existing accepted fit: {accepted}')
    raw = local(ROOT, row['input_snapshot'])
    if digest(raw) != row['input_sha256']:
        raise ValueError(f'Changed frozen input: {raw}')
    data = training_data(json.loads(raw.read_text()), row['mode'], cutoff)
    data.update(original.prior(data))
    source = ROOT / 'pipeline/models' / ('free_kernel.stan' if model_kind == 'C' else 'leading_order.stan')
    if cost_scenario:
        data.update(cost_prior(cost_scenario))
        source = ROOT / 'pipeline/models/common_cost.stan'
    model = compile_stan(source)
    # Deterministic prior-scale starts avoid depending on deleted old chains.
    inits = [dict(c_i=f, c_j=1/f, sigma=2., **{'lambda': 100.},
                  mu_0=float(data['prior_mean'][4]), log_r=math.log(10.))
             for f in (.7, .9, 1.2, 1.5)]
    if model_kind == 'C':
        for j, init in enumerate(inits):
            init.update(deadline_power=(.2,.4,.6,.8)[j],gap_log_scale=(-.2,-.05,.05,.2)[j])
    seed = 20261029 + row['contest_id'] + sum(map(ord, row['scenario'])) * 100000
    configuration = dict(raw_input_sha256=digest(raw), input_snapshot=row['input_snapshot'],
                         source_model_sha256=digest(source), prior=original.prior(data),
                         initialization='Fixed dispersed prior-scale starts; no historical draws',
                         seed=seed, mode=row['mode'], scenario=row['scenario'],
                         model=model_kind, training_cutoff=data['N_Delta'],
                         full_constructed_deadline=data['T_contest'], split=split,
                         scope='Fresh same-target posterior; historical chain identity not claimed')
    if cost_scenario:
        if baseline is None:
            raise ValueError('Supply a matching accepted full-window baseline for initialization')
        starting = json.loads(baseline.read_text())
        cfg = starting['configuration']
        if (not starting['diagnostics']['passed'] or cfg['raw_input_sha256']!=digest(raw)
                or cfg['training_cutoff']!=data['N_Delta'] or cfg['model']!='D'
                or cfg['source_model_sha256']!=digest(ROOT/'pipeline/models/leading_order.stan')
                or cfg['prior']!=original.prior(data)):
            raise ValueError('Baseline fit does not match the retained full-window target')
        center = starting['summary']
        inits = []
        for j, factor in enumerate((.8,1.,1.2,1.5)):
            init = {k:max(1.01*(data['cost_floor'] if k.startswith('c_') else .01),
                          center[k+'_median']*factor) for k in ('c_i','c_j','sigma','lambda')}
            init.update(mu_0=center['mu_0_median']+(j-1.5)*.05,
                        log_r=max(math.log(.01)+.001,math.log(center['r_median'])+(j-1.5)*.1))
            inits.append(init)
        seed = 20261025 + 700000 + row['contest_id'] + 1000000*COST_TASK_INDEX[cost_scenario][row['contest_id']]
        configuration.update(seed=seed, initialization='Dispersed full-window baseline medians',
            inputs=data, input_sha256=digest(raw), model_sha256=digest(source),
            executable_sha256=digest(model.exe_file), cost_scenario=cost_scenario,
            baseline_accepted_sha256=digest(baseline))
    write_json(case / 'configuration.json', configuration)
    schedule = [(1000,2000,.98),(2000,4000,.99),(4000,8000,.995)]
    if split or cost_scenario:
        schedule += [(2000,32000,.995),(4000,64000,.9975)]
    for attempt, (warmup, samples, delta) in enumerate(schedule, 1):
        folder = case / f'attempt{attempt}'
        if folder.exists():
            raise FileExistsError(f'Preserve previous attempt: {folder}')
        folder.mkdir()
        write_json(folder / 'data.json', data)
        result = model.sample(data=data, chains=4, parallel_chains=2, seed=seed,
                              iter_warmup=warmup, iter_sampling=samples, inits=inits,
                              adapt_delta=delta, max_treedepth=13, metric='dense_e',
                              output_dir=str(folder), sig_figs=18, show_progress=False)
        paths = [Path(p).resolve() for p in result.runset.csv_files]
        diagnostic = diagnose_cost if cost_scenario else diagnose_free if model_kind == 'C' else diagnose
        summary, diagnostics = diagnostic(row['contest_id'], row['mode'], paths, data)
        record = dict(summary=summary, diagnostics=diagnostics, configuration=configuration,
                      chain_paths=[os.path.relpath(p, accepted.parent) for p in paths],
                      chain_path_base='accepted_parent',
                      chain_sha256=[digest(p) for p in paths], attempt=attempt)
        write_json(folder / 'diagnostics.json', record)
        if diagnostics['passed']:
            write_json(accepted, record)
            return accepted
        inits = []
        for path in paths:
            values = estimator.read_columns(path)
            names = (*estimator.PARAMETERS,'deadline_power','gap_log_scale') if model_kind == 'C' else estimator.PARAMETERS
            inits.append({name: float(np.median(values[name])) for name in names})
    raise RuntimeError(f'No accepted posterior for {row["contest_id"]}; retain failures for review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='main')
    parser.add_argument('--mode', choices=['full', 'early'], default='full')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--contest', type=int)
    selection.add_argument('--all', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', default='tmp/runs')
    args = parser.parse_args()
    rows = plan(args.scenario, args.mode, args.contest)
    output = local(ROOT, args.output)
    for row in rows:
        if args.dry_run:
            print(row['scenario'], row['mode'], row['contest_id'], row['input_snapshot'])
        else:
            print(fit(row, output), flush=True)


if __name__ == '__main__':
    main()
