"""Independent fixed-mixture MH check for a slow, potentially multimodal fit.

This does not replace an accepted empirical estimate. The density is exactly
the same leading-policy posterior, checked against Stan before sampling.
"""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel
from pipeline.empirical import unified_prior_reestimate as u,sigma_lambda_trials as s,nonlinear_reestimate as n


class Target(s.Target):
    def __init__(self,data,*args,**kwargs):super().__init__(data,'lambda_external')


def folder():return u.OUT/'global_crosschecks'/os.environ['UNIFIED_GLOBAL_CASE']


def load(cid):
    root=folder();design=json.loads((root/'design.json').read_text());data=json.loads((root/'data.json').read_text())
    chains=[]
    for path,h in zip(design['source_chain_paths'],design['source_chain_sha256']):
        if u.digest(path)!=h:raise RuntimeError('Changed fixed proposal-training chains')
        columns=u.e.read_columns(Path(path));chains.append(np.array([columns[p] for p in u.e.PARAMETERS]).T)
    z=np.concatenate(chains);z[:,:4]=np.log(z[:,:4])
    return data,chains,z,dict(global_design_sha256=u.digest(root/'design.json'),global_script_sha256=u.digest(__file__),target='Same leading likelihood and lambda100_sd225 priors')


def configure(case):
    os.environ['UNIFIED_GLOBAL_CASE']=case
    s.kernel();n.SOURCE=s.OUT/'weighted_leading_kernel.cpp';n.Likelihood=Target;n.kernel=s.kernel;n.load_baseline=load;n.OUT=folder()


def prepare(scenario,cid,mode):
    case=f'{scenario}_{cid}_{mode}';configure(case);root=folder();root.mkdir(parents=True,exist_ok=True)
    base=u.OUT if scenario=='main' else u.OUT/'auxiliary'/scenario
    source=base/'estimation'/mode/f'contest_{cid}/attempt2'
    if not all(u.e.csv_complete(source/f'chain_{j}.csv',4000) for j in range(1,5)):raise RuntimeError('Four completed second-attempt chains are needed')
    data=json.loads((source/'data.json').read_text())
    if any(data[k]!=v for k,v in u.prior(data).items()):raise RuntimeError('Prior mismatch')
    paths=[source/f'chain_{j}.csv' for j in range(1,5)]
    design=dict(scenario=scenario,contest_id=cid,mode=mode,source_chain_paths=[str(p) for p in paths],source_chain_sha256=[u.digest(p) for p in paths],
                source_script_sha256=u.digest(__file__),target_model_sha256=u.digest(u.MODEL.with_suffix('.stan')),kernel_sha256=u.digest(n.SOURCE),
                interpretation='Fixed proposal training only. Exact density corrections, no likelihood/priors/model change. Not a nonlinear-equilibrium estimation.')
    signature=root/'design.json'
    if signature.exists() and json.loads(signature.read_text())!=design:raise RuntimeError('Global-check design changed')
    u.save(signature,design);u.save(root/'data.json',data)
    data,_,z,_=load(cid)
    model=CmdStanModel(stan_file=str(u.MODEL.with_suffix('.stan')),exe_file=str(u.MODEL));target=Target(data)
    differences=[]
    for index in (0,len(z)//3,2*len(z)//3,len(z)-1):
        point=z[index];params=dict(zip(u.e.PARAMETERS,np.r_[np.exp(point[:4]),point[4:]].tolist()))
        stan=float(model.log_prob(params,data=data,jacobian=False,sig_figs=18).iloc[0,0])+sum(point[:4])
        differences.append(target.posterior(point)-stan)
    if np.ptp(differences)>1e-6:raise RuntimeError('Global sampler target differs from Stan')
    u.save(root/'target_parity.json',dict(passed=True,relative_target_differences=differences,max_after_constant=float(np.ptp(differences))))
    if not (root/f'proposal_{cid}.json').exists():
        components=[];cut=float(np.median(z[:,2]));count=len(z)//4
        for j in range(4):
            a=z[j*count:(j+1)*count]
            for group in (a[a[:,2]<=cut],a[a[:,2]>cut]):
                if len(group)<100:continue
                cov=np.cov(group,rowvar=False)+np.eye(6)*1e-7
                components.append(dict(weight=1.,mean=group.mean(0).tolist(),covariance=cov.tolist()))
        for c in components:c['weight']=1/len(components)
        u.save(root/f'proposal_{cid}.json',dict(contest_id=cid,kernel_sha256=u.digest(n.SOURCE),mode_z=np.median(z,axis=0).tolist(),
            covariance=(np.cov(z,rowvar=False)+np.eye(6)*1e-7).tolist(),mixture_components=components))
    return case


def chain(task):
    case,cid,j,samples=task;configure(case)
    n.sample_mixture_gaussian(cid,j,1500,samples,.025,64,202610150+cid,pilot=True)


def summarize(case,cid):
    configure(case);data,_,_,_=load(cid);paths=[folder()/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
    z=np.array([np.load(p)['z'] for p in paths]);v=n.physical_draws(z.reshape(-1,6),data['theta']).reshape(4,z.shape[1],9)
    names=u.NAMES+('w_i','w_j','max_w');diagnostics={};summaries={}
    for index,name in enumerate(names):
        a=v[:,:,index];bulk,tail=n.rank_ess(a);diagnostics[name]=dict(rank_rhat=float(n.rank_rhat(a)),bulk_ess=float(bulk),tail_ess=float(tail))
        summaries[name]=dict(mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std(ddof=1)),q025=float(np.quantile(a,.025)),q975=float(np.quantile(a,.975)))
    passed=all(d['rank_rhat']<=1.01 and min(d['bulk_ess'],d['tail_ess'])>=400 for d in diagnostics.values())
    result=dict(passed=passed,diagnostics=diagnostics,summaries=summaries,chain_sha256=[u.digest(p) for p in paths],
                interpretation='Independent same-posterior computation. No empirical truth or nonlinear-approximation-error claim.',source_sha256=u.digest(__file__))
    u.save(folder()/'summary.json',result)
    print('GLOBAL CHECK',case,'passed',passed,'rhat',max(d['rank_rhat'] for d in diagnostics.values()),'minESS',min(min(d['bulk_ess'],d['tail_ess']) for d in diagnostics.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('contest_id',type=int);parser.add_argument('--mode',choices=('full','early'),default='full');parser.add_argument('--samples',type=int,default=30000);args=parser.parse_args()
    case=prepare(args.scenario,args.contest_id,args.mode)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,[(case,args.contest_id,j,args.samples) for j in range(1,5)]))
    summarize(case,args.contest_id)
