"""Same fresh Bayesian posterior; primary estimates are posterior medians.

This matches the point convention of the prior-selection recovery experiments.
Raw posterior means and their Monte Carlo errors remain sensitivity diagnostics.
"""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import numpy as np
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical import unified_prior_auxiliary as aux

BASE_OUT=u.OUT
RAW_DIAGNOSE=u.diagnose


def diagnose(cid,mode,paths,data):
    row,d=RAW_DIAGNOSE(cid,mode,paths,data)
    d['passed']=bool(row['divergences']==0 and d['saturated_transitions']==0 and min(d['energy_bfmi'])>=.3
        and all(v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400 for v in d['quantities'].values()))
    d['primary_point_summary']='posterior median; same convention as the nonlinear-generated recovery experiments'
    d['posterior_mean_stability']={p:dict(relative_mcse=d['quantities'][p]['relative_mean_mcse'],below_three_percent=d['quantities'][p]['relative_mean_mcse']<=.03) for p in ('sigma','lambda')}
    d['median_protocol_sha256']=u.digest(__file__)
    medians={}
    for p in ('sigma','lambda'):
        values=np.asarray([u.e.read_columns(path)[p] for path in paths]);logged=np.log(values)
        ess=u.n.effective_sample_size((values<=np.median(values)).astype(float))
        width=np.diff(np.quantile(logged,[.45,.55]))[0]
        medians[p]=dict(approximate_log_median_mcse=float(5*width/np.sqrt(max(1,ess))),median_indicator_ess=float(ess))
    d['median_monte_carlo_diagnostics']=medians
    return row,d


def fit(task):
    scenario,cid,mode=task
    if scenario=='main':u.OUT=BASE_OUT
    else:
        aux.CURRENT=scenario;u.OUT=BASE_OUT/'auxiliary'/scenario;u.baseline=aux.baseline;u.initializations=aux.starts
    u.diagnose=diagnose
    case=u.OUT/'estimation'/mode/f'contest_{cid}'
    accepted=case/'accepted.json'
    if accepted.exists():
        result=json.loads(accepted.read_text())
        if result['diagnostics'].get('median_protocol_sha256')!=u.digest(__file__):
            paths=[Path(p) for p in result['chain_paths']]
            data=json.loads((paths[0].parent/'data.json').read_text())
            row,d=diagnose(cid,mode,paths,data)
            if not d['passed']:raise RuntimeError('Previously accepted fit fails median-protocol MCMC gates')
            result['summary']=row;result['diagnostics']=d;u.save(accepted,result)
    # Preserve partial files from interrupted controllers before resampling them.
    for attempt,samples in ((1,2000),(2,4000),(3,8000)):
        folder=case/f'attempt{attempt}'
        for j in range(1,5):
            path=folder/f'chain_{j}.csv'
            if path.exists() and not u.e.csv_complete(path,samples):
                k=1
                while (folder/f'chain_{j}.interrupted{k}.csv').exists():k+=1
                path.rename(folder/f'chain_{j}.interrupted{k}.csv')
    return u.fit_one((cid,mode,False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--auxiliary',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    ids=sorted({int(r['contest_id']) for r in csv.DictReader((u.MAIN_ESTIMATION_DIR/'accepted_chain_sources.csv').open())})
    if args.auxiliary:
        tasks=[(scenario,cid,'full') for scenario in ('pair','L1','L2') for cid in ids]
        tasks.extend((scenario,cid,mode) for scenario in ('seed_20261008','seed_20261009') for cid in aux.TIME_IDS for mode in ('full','early'))
    else:tasks=[('main',cid,mode) for cid in ids for mode in ('full','early')]
    name='auxiliary_median_protocol.json' if args.auxiliary else 'median_protocol.json'
    protocol=dict(tasks=tasks,source_sha256=u.digest(__file__),target_source_sha256=u.digest(u.__file__),auxiliary_source_sha256=u.digest(aux.__file__),
        primary_summary='Posterior medians, matching the existing recovery experiments. Full joint posterior unchanged.',
        gates='All primitive and w rank Rhat<=1.01 and bulk/tail ESS>=400; no HMC divergences/depth saturations; all energy BFMI>=.3.',
        means='Raw means retained with explicit MCSE; long-tailed means are not substituted for validated median point summaries. Earlier conditional-integration experiments are preserved.',
        prior='log(lambda)~N(log100,2.25^2); other priors, supports, data, leading policy and USD100000 units unchanged.')
    path=BASE_OUT/name
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):raise RuntimeError('Median protocol changed')
    u.save(path,protocol);failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fit,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                failures.append(dict(task=futures[future],error=str(error)));print('FAILED',futures[future],error,flush=True)
            if args.auxiliary:aux.assemble()
            else:u.assemble()
            u.save(BASE_OUT/('auxiliary_median_failures.json' if args.auxiliary else 'median_failures.json'),failures)
    if failures:raise RuntimeError('Failed fits remain; their completed chains are preserved')


if __name__=='__main__':main()
