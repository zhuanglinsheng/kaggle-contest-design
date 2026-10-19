"""Fresh score-scale NUTS fits initialized from the new baseline posterior.

Use a short run only when every unchanged median-protocol convergence gate
passes. Failed short runs are preserved; the ordinary full runner refines them.
Never replace a longer attempted fit with a shorter one.
"""
import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path
import numpy as np
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical import unified_prior_medians as med
from pipeline.empirical import unified_prior_auxiliary as aux

BASE_OUT=u.OUT
SCENARIO=None
CURRENT_BASE=None


def baseline(cid,mode):
    return Path(CURRENT_BASE['chain_paths'][0]).parent,aux.input_directory(SCENARIO)/f'contest_{cid}.json'


def starts(cid,mode,previous=None):
    if previous is not None:raise RuntimeError('Short run has no continuation stage')
    raw=json.loads((aux.input_directory(SCENARIO)/f'contest_{cid}.json').read_text());a=float(raw['score_scale_a'])
    result=[]
    for path in CURRENT_BASE['chain_paths']:
        c=u.e.read_columns(Path(path));v={p:float(np.median(c[p])) for p in u.e.PARAMETERS}
        for p in ('c_i','c_j','lambda'):v[p]=max(.0101,v[p]/a**2)
        v['sigma']=max(.0101,v['sigma']*a);v['mu_0']*=a;v['log_r']=max(math.log(.01)+.001,v['log_r']-math.log(a))
        result.append(v)
    return result


def fit(task):
    global SCENARIO,CURRENT_BASE
    scenario,cid=task;SCENARIO=scenario
    case=BASE_OUT/'auxiliary'/scenario/'estimation/full'/f'contest_{cid}'
    # A full controller may already be working on this case; do not race it.
    if (case/'configuration.json').exists():return
    base=BASE_OUT/'estimation/full'/f'contest_{cid}/accepted.json'
    if not base.exists():return
    CURRENT_BASE=json.loads(base.read_text())
    if not CURRENT_BASE['diagnostics']['passed']:raise RuntimeError('Rejected baseline warm start')
    u.OUT=BASE_OUT/'auxiliary'/scenario;u.baseline=baseline;u.initializations=starts;u.diagnose=med.diagnose
    result=u.fit_one((cid,'full',True))
    result['diagnostics']['short_run_protocol_sha256']=u.digest(__file__)
    result['diagnostics']['warmstart_baseline_chain_sha256']=CURRENT_BASE['chain_sha256']
    u.save(case/'pilot1/diagnostics.json',result)
    if result['diagnostics']['passed']:
        u.save(case/'accepted.json',result)
        print('ACCEPTED SHORT SCALE',scenario,cid,flush=True)
    else:print('SHORT SCALE REQUIRES ORDINARY REFINEMENT',scenario,cid,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    seen=set()
    protocol=BASE_OUT/'scale_warmstart_protocol.json'
    design=dict(source_sha256=u.digest(__file__),primary_gate_source_sha256=u.digest(med.__file__),warmup=400,samples_per_chain=600,chains=4,
                acceptance='Exactly the primary-median HMC/rank/ESS gates; no mean-stability substitution; fresh target-specific NUTS draws.',
                starts='Transformed accepted new-baseline joint chains; priors remain unchanged in transformed score units; no draw reuse or importance-only estimate.')
    if protocol.exists() and json.loads(protocol.read_text())!=design:raise RuntimeError('Changed scale warm-start protocol')
    u.save(protocol,design)
    while True:
        ids=sorted(int(p.parent.name.split('_')[1]) for p in (BASE_OUT/'estimation/full').glob('contest_*/accepted.json'))
        tasks=[]
        for scenario in ('L2','L1'):
            for cid in reversed(ids):
                if (scenario,cid) not in seen:
                    seen.add((scenario,cid));tasks.append((scenario,cid))
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(fit,tasks))
        progress=BASE_OUT/'progress.json'
        if progress.exists() and json.loads(progress.read_text())['complete']:break
        time.sleep(30)


if __name__=='__main__':main()
