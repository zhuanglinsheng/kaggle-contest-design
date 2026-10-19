"""Broad activity-scale sensitivity, applied identically to both estimators.

Reject low-overlap importance estimates rather than extrapolating posterior
tails. No existing result or publication model is modified.
"""
import json
import math
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_prior_report as report


def target(data):
    cfg=a.settings(data,'scale_weak');cfg['prior_mean'][5]=math.log(80);cfg['prior_sd'][5]=1.5
    return cfg


def reweight(z,baseweights,data,old):
    new=target(data)
    ratio=-.5*np.sum(((z-np.array(new['prior_mean']))/new['prior_sd'])**2-
                     ((z-np.array(old['prior_mean']))/old['prior_sd'])**2,axis=1)
    weights=softmax(np.log(baseweights)+ratio)
    values=report.quantities(z,data);means,sds,q=n.weighted_summary(values,weights)
    ess=[]
    for k in range(values.shape[1]):
        influence=(len(z)*weights*(values[:,k]-means[k])).reshape(4,-1)
        ess.append(n.effective_sample_size(influence)*sds[k]**2/max(influence.var(ddof=1),1e-30))
    diagnostics=dict(weight_ess=float(1/(weights@weights)),minimum_mean_ess=float(min(ess)),maximum_weight=float(weights.max()))
    passed=diagnostics['weight_ess']>=400 and diagnostics['minimum_mean_ess']>=400 and diagnostics['maximum_weight']<.02
    return weights,dict(validated=passed,**diagnostics),report.summary(z,weights,data)


def run():
    rows=[];comparisons=[]
    for cid in a.IDS:
        data,_,_,_=n.load_baseline(cid);sources=[]
        folder=report.leading_source(cid,'scale_weak')
        if folder is not None:
            _,gate,config=report.leading_result(folder,data)
            if gate['passed']:
                x=np.load(folder/'posterior.npz')['values'].transpose(1,0,2)
                z=x.reshape(-1,6).copy();z[:,:4]=np.log(z[:,:4])
                sources.append(('leading',z,np.ones(len(z))/len(z),config['inputs'],folder/'posterior.npz'))
        root=p.OUT/'scale_weak/nonlinear'
        gate=root/f'posterior_grid_{cid}.json'
        if gate.exists() and json.loads(gate.read_text())['status']=='validated':
            grid=np.load(root/f'posterior_grid_{cid}.npz')
            sources.append(('nonlinear',grid['z'],softmax(grid['fine']-grid['base']),a.settings(data,'scale_weak'),root/f'posterior_grid_{cid}.npz'))
        current={}
        for model,z,bw,old,path in sources:
            weights,diag,summaries=reweight(z,bw,data,old)
            item=dict(contest_id=cid,model=model,source_sha256=n.digest(path),**diag)
            rows.append(item)
            if diag['validated']:
                current[model]=summaries
                destination=p.OUT/'activity_scale'/model;destination.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination/f'posterior_{cid}.npz',z=z,weights=weights)
                n.save_json(destination/f'summary_{cid}.json',dict(**item,prior=target(data),summaries=summaries))
        if len(current)==2:
            for key in report.NAMES:
                lhs,rhs=current['leading'][key],current['nonlinear'][key]
                comparisons.append(dict(contest_id=cid,parameter=key,leading_median=lhs['median'],nonlinear_median=rhs['median'],
                   leading_q025=lhs['q025'],leading_q975=lhs['q975'],nonlinear_q025=rhs['q025'],nonlinear_q975=rhs['q975'],
                   gap_in_pooled_sd=abs(lhs['mean']-rhs['mean'])/math.sqrt((lhs['sd']**2+rhs['sd']**2)/2)))
    # Existing known-parameter leading control: its source already passed NUTS checks.
    folder=a.OUT/'feedback_timing/recovery_4657/leading_control/interval_start_r_weak'
    config=json.loads((folder/'run.json').read_text());data=config['inputs']
    x=np.load(folder/'posterior.npz')['values'].transpose(1,0,2)
    z=x.reshape(-1,6).copy();z[:,:4]=np.log(z[:,:4])
    weights,diag,summaries=reweight(z,np.ones(len(z))/len(z),data,data)
    truth=json.loads((folder.parent/'truth.json').read_text())['truth']
    recovery=dict(source_sha256=n.digest(folder/'posterior.npz'),prior=target(data),diagnostics=diag,
                  note='One preserved realization, not a repeated-sample coverage study.')
    if diag['validated']:
        recovery['summaries']=summaries
        recovery['truth']=truth
        recovery['truth_in_95_interval']={key:summaries[key]['q025']<=truth[key]<=summaries[key]['q975'] for key in n.NAMES}
        recovery['median_relative_error']={key:abs(summaries[key]['median']/truth[key]-1) for key in ('c_i','c_j','sigma','lambda','r')}
    n.save_json(p.OUT/'activity_scale_recovery_control.json',recovery)
    n.save_json(p.OUT/'activity_scale_status.json',rows)
    if comparisons:n.write_csv(p.OUT/'activity_scale_comparison.csv',comparisons)
    print(json.dumps(dict(statuses=rows,control=recovery),indent=2),flush=True)


if __name__=='__main__':run()
