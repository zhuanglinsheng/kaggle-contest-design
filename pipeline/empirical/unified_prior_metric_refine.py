"""Fresh NUTS refinement initialized from a validated same-target global check.

Preconditioning and starting points change; likelihood, priors, observations,
NUTS target, and primary-median acceptance gates remain unchanged.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from pipeline.empirical import unified_prior_reestimate as u,unified_prior_medians as med


def run(scenario,cid,mode):
    base=u.OUT if scenario=='main' else u.OUT/'auxiliary'/scenario
    case=base/'estimation'/mode/f'contest_{cid}'
    reference=u.OUT/'global_crosschecks'/f'{scenario}_{cid}_{mode}'
    summary=json.loads((reference/'summary.json').read_text())
    if not summary['passed']:raise RuntimeError('Global reference is not converged')
    config=json.loads((case/'configuration.json').read_text());inputs=config['inputs']
    original_paths=[reference/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
    if [u.digest(p) for p in original_paths]!=summary['chain_sha256']:raise RuntimeError('Changed global reference')
    z=np.array([np.load(p)['z'] for p in original_paths]);flat=z.reshape(-1,6)
    unconstrained=flat.copy()
    unconstrained[:,:4]=np.log(np.maximum(np.exp(flat[:,:4])-.01,1e-15))
    unconstrained[:,5]=np.log(np.maximum(flat[:,5]-math.log(.01),1e-15))
    cov=np.cov(unconstrained,rowvar=False);eigen,vec=np.linalg.eigh(cov)
    covariance=(vec*np.maximum(eigen,max(1e-10,float(eigen.max())*1e-10)))@vec.T
    folder=case/'global_metric_refinement';folder.mkdir(parents=True,exist_ok=True)
    settings=dict(source_sha256=u.digest(__file__),global_summary_sha256=u.digest(reference/'summary.json'),
                  global_chain_sha256=[u.digest(p) for p in original_paths],inverse_metric=covariance.tolist(),
                  warmup=1000,samples_per_chain=2000,adapt_delta=.99,max_depth=13,
                  interpretation='Stan inverse metric is covariance in its exact unconstrained coordinates; based on the same-target global posterior, used only for initialization.')
    signature=folder/'settings.json'
    if signature.exists() and json.loads(signature.read_text())!=settings:raise RuntimeError('Changed metric-refinement configuration')
    u.save(signature,settings);u.save(folder/'data.json',inputs);u.save(folder/'metric.json',dict(inv_metric=covariance.tolist()))
    rng=np.random.default_rng(314150+cid);starts=[]
    for j in range(4):
        point=z[j,int(rng.integers(len(z[j])))];values=np.r_[np.exp(point[:4]),point[4:]]
        values[:4]=np.maximum(values[:4],.0100000001);values[5]=max(values[5],math.log(.01)+1e-8)
        starts.append(dict(zip(u.e.PARAMETERS,values.tolist())))
    paths=[folder/f'chain_{j}.csv' for j in range(1,5)];processes=[]
    for j,path in enumerate(paths,1):
        u.save(folder/f'init_{j}.json',starts[j-1])
        if u.e.csv_complete(path,2000):continue
        log=(folder/f'chain_{j}.log').open('w')
        command=[str(u.MODEL),'sample','num_warmup=1000','num_samples=2000','adapt','delta=.99',
                 'algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e',f'metric_file={folder/"metric.json"}',
                 'random',f'seed={20271300+cid+(100000 if mode=="early" else 0)}',f'id={j}',
                 'data',f'file={folder/"data.json"}','output',f'file={path}','refresh=100','sig_figs=18',f'init={folder/f"init_{j}.json"}']
        processes.append((subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')),log))
    codes=[]
    for process,handle in processes:codes.append(process.wait());handle.close()
    if any(codes):raise RuntimeError('Metric-initialized NUTS failed')
    row,diagnostics=med.diagnose(cid,mode,paths,inputs)
    diagnostics['global_metric_refinement_source_sha256']=u.digest(__file__)
    diagnostics['global_reference_summary_sha256']=u.digest(reference/'summary.json')
    result=dict(summary=row,diagnostics=diagnostics,configuration=config,attempt='global_metric_refinement',
                chain_paths=[str(p) for p in paths],chain_sha256=[u.digest(p) for p in paths])
    u.save(folder/'diagnostics.json',result)
    if not diagnostics['passed']:raise RuntimeError('Fresh preconditioned NUTS does not meet unchanged gates')
    u.save(case/'metric_candidate.json',result)
    print('METRIC NUTS CANDIDATE PASSED',scenario,cid,mode,flush=True)
    # The ordinary controller may still be writing attempt3. Keep its diagnostic
    # history intact and avoid a late overwrite of the accepted result.
    while not (case/'attempt3/diagnostics.json').exists():time.sleep(10)
    old=json.loads((case/'attempt3/diagnostics.json').read_text())
    if old['diagnostics']['passed']:
        print('Ordinary third attempt also passed; retain it and keep independent metric check',flush=True)
        return
    time.sleep(1)
    u.save(case/'accepted.json',result)
    print('ACCEPTED METRIC-INITIALIZED NUTS',scenario,cid,mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');args=parser.parse_args();run(args.scenario,args.contest_id,args.mode)
