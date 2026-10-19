"""Export validated global MH draws without inventing HMC diagnostics.

Selected multimodal cases use the same posterior and primary-median criteria;
HMC-only fields are explicitly unavailable. Ordinary NUTS attempts are retained.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
from pipeline.empirical import unified_prior_reestimate as u,unified_prior_medians as med,unified_prior_global_check as g


def run(scenario,cid,mode):
    base=u.OUT if scenario=='main' else u.OUT/'auxiliary'/scenario
    case=base/'estimation'/mode/f'contest_{cid}'
    reference=u.OUT/'global_crosschecks'/f'{scenario}_{cid}_{mode}'
    check=json.loads((reference/'summary.json').read_text());parity=json.loads((reference/'target_parity.json').read_text())
    if not check['passed'] or not parity['passed']:raise RuntimeError('Global posterior or density parity not verified')
    if check['source_sha256']!=u.digest(g.__file__):raise RuntimeError('Global sampler source changed')
    cfg=json.loads((case/'configuration.json').read_text())
    data=json.loads((reference/'data.json').read_text())
    if data!=cfg['inputs']:raise RuntimeError('Global target differs from the intended empirical target')
    paths=[reference/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
    if [u.digest(p) for p in paths]!=check['chain_sha256']:raise RuntimeError('Changed global chains')
    z=np.array([np.load(p)['z'] for p in paths]);v=u.n.physical_draws(z.reshape(-1,6),data['theta']).reshape(4,z.shape[1],9)
    folder=case/'global_mh_refinement';folder.mkdir(parents=True,exist_ok=True)
    exported=[]
    for j,path in enumerate(paths,1):
        destination=folder/f'chain_{j}.csv';raw=np.load(path)
        if not destination.exists():
            with destination.open('w',newline='') as handle:
                handle.write('# sampler = fixed-mixture Metropolis-Hastings, exact leading posterior\n')
                handle.write('# lp__ is density in sampler coordinates; HMC diagnostics are not applicable\n')
                writer=csv.writer(handle);writer.writerow(('lp__',)+u.e.PARAMETERS)
                for point,lp in zip(raw['z'],raw['logp']):writer.writerow(np.r_[lp,np.exp(point[:4]),point[4:]].tolist())
        exported.append(destination)
    u.save(folder/'data.json',data)
    row=dict(json.loads((case/'attempt2/diagnostics.json').read_text())['summary'])
    diagnostics={};names=u.NAMES+('w_i','w_j','max_w')
    for index,name in enumerate(names):
        values=v[:,:,index];mean=float(values.mean());sd=float(values.std(ddof=1));ess=u.n.effective_sample_size(values)
        mcse=sd/math.sqrt(max(ess,1));bulk,tail=u.n.rank_ess(values)
        diagnostics[name]=dict(rank_rhat=float(u.n.rank_rhat(values)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,
                               relative_mean_mcse=mcse/abs(mean) if mean else None)
        row[name+'_median']=float(np.median(values))
        if name in u.NAMES:
            row.update({name+'_mean':mean,name+'_sd':sd,name+'_q025':float(np.quantile(values,.025)),name+'_q975':float(np.quantile(values,.975)),
                        name+'_rhat':u.e.split_rhat(list(z[:,:,5] if name=='r' else values)),name+'_rank_rhat':diagnostics[name]['rank_rhat'],
                        name+'_bulk_ess':float(bulk),name+'_tail_ess':float(tail),name+'_mean_mcse':mcse})
    row['divergences']=None;row['max_treedepth']=None
    median_diagnostics={}
    for index,name in ((2,'sigma'),(3,'lambda')):
        values=v[:,:,index];indicator_ess=u.n.effective_sample_size((values<=np.median(values)).astype(float))
        width=float(np.diff(np.quantile(np.log(values),[.45,.55]))[0])
        median_diagnostics[name]=dict(approximate_log_median_mcse=5*width/math.sqrt(max(1,indicator_ess)),median_indicator_ess=float(indicator_ess))
    passed=all(d['rank_rhat']<=1.01 and min(d['bulk_ess'],d['tail_ess'])>=400 for d in diagnostics.values())
    if not passed:raise RuntimeError('Global export does not meet common rank/ESS gates')
    d=dict(passed=True,quantities=diagnostics,sampler_family='fixed_mixture_metropolis_hastings',hmc_applicable=False,
           saturated_transitions=None,energy_bfmi=[],primary_point_summary='posterior medians',median_protocol_sha256=u.digest(med.__file__),
           mh_export_source_sha256=u.digest(__file__),global_reference=str(reference),global_summary_sha256=u.digest(reference/'summary.json'),
           global_target_parity_sha256=u.digest(reference/'target_parity.json'),median_monte_carlo_diagnostics=median_diagnostics,
           posterior_mean_stability={p:dict(relative_mcse=diagnostics[p]['relative_mean_mcse'],below_three_percent=diagnostics[p]['relative_mean_mcse']<=.03) for p in ('sigma','lambda')})
    result=dict(summary=row,diagnostics=d,configuration=dict(cfg,sampler='fixed_mixture_metropolis_hastings'),attempt='global_mh_refinement',
                chain_paths=[str(p) for p in exported],chain_sha256=[u.digest(p) for p in exported])
    u.save(folder/'diagnostics.json',result);u.save(case/'global_mh_candidate.json',result)
    print('VALIDATED GLOBAL MH CANDIDATE',scenario,cid,mode,flush=True)
    # An in-flight ordinary attempt can still write an accepted file. Wait for
    # its final diagnostic, retaining the entire ordinary diagnostic history.
    while not (case/'attempt3/diagnostics.json').exists():time.sleep(10)
    time.sleep(1)
    u.save(case/'accepted.json',result)
    print('ACCEPTED SAME-POSTERIOR GLOBAL MH',scenario,cid,mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');args=parser.parse_args();run(args.scenario,args.contest_id,args.mode)
