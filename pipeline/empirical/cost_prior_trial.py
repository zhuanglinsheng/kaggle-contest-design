"""Separate common cost scale and relative cost without changing likelihoods.

This is a controlled prior sensitivity, not an automatic prior selector.
Validated source draws are retained; failed importance overlap yields no usable
target summaries. Empirical and recovery targets remain paired within dataset.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as audit
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_prior_report as report
from pipeline.empirical import paired_prior_recovery as recovery

OUT=p.OUT/'cost_structure_trial'
# Keep original relative-cost uncertainty; vary only common scale, then noise.
CANDIDATES={
    'relative_cost_preserved':dict(common_sd=1.25/math.sqrt(2),relative_sd=.75*math.sqrt(2),sigma_sd=.5,lambda_sd=.75),
    'common_scale_wide':dict(common_sd=1.25,relative_sd=.75*math.sqrt(2),sigma_sd=.5,lambda_sd=.75),
    'common_scale_noise_wide':dict(common_sd=1.25,relative_sd=.75*math.sqrt(2),sigma_sd=.75,lambda_sd=1.25),
}


def log_ratio(z,candidate):
    """New minus source log prior; only changed coordinates enter.

    Source independent log costs each SD1.25; log sigma/precision SD .5/.75.
    Transformation (log ci, log cj)->(u,d) has absolute determinant one.
    All parameter supports and remaining priors stay identical.
    """
    u=(z[:,0]+z[:,1])/2;d=z[:,0]-z[:,1]
    logp=-.5*((u/candidate['common_sd'])**2+(d/candidate['relative_sd'])**2)
    old=-.5*((z[:,0]/1.25)**2+(z[:,1]/1.25)**2)
    ratio=logp-old-math.log(candidate['common_sd']*candidate['relative_sd']/1.25**2)
    for k,mean,previous,key in [(2,math.log(2),.5,'sigma_sd'),(3,0.,.75,'lambda_sd')]:
        sd=candidate[key]
        ratio+=-.5*(z[:,k]-mean)**2*(1/sd**2-1/previous**2)-math.log(sd/previous)
    return ratio


def quantities(z,data):
    return np.column_stack((report.quantities(z,data),np.exp(z[:,0]-z[:,1])))


def evaluate(z,baseweights,data,candidate):
    weights=softmax(np.log(baseweights)+log_ratio(z,candidate))
    values=quantities(z,data);means,sds,q=n.weighted_summary(values,weights)
    ess=[]
    for k in range(values.shape[1]):
        influence=(len(z)*weights*(values[:,k]-means[k])).reshape(4,-1)
        ess.append(float(min(len(z),n.effective_sample_size(influence)*sds[k]**2/max(influence.var(ddof=1),1e-30))))
    diagnostic=dict(weight_ess=float(1/(weights@weights)),minimum_mean_ess=min(ess),maximum_weight=float(weights.max()))
    passed=diagnostic['weight_ess']>=400 and diagnostic['minimum_mean_ess']>=400 and diagnostic['maximum_weight']<.02
    diagnostic['passed']=passed
    names=report.NAMES+('cost_ratio',)
    summary={key:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,key in enumerate(names)}
    return weights,diagnostic,summary


def grid_check(dataset,weights,data):
    if dataset.startswith('recovery_'):
        root=p.OUT/'recovery_4657'/dataset.removeprefix('recovery_')/'activity_scale/nonlinear';cid=4657
    else:root=p.OUT/'scale_weak/nonlinear';cid=int(dataset)
    grid=np.load(root/f'posterior_grid_{cid}.npz');z=grid['z']
    representative=np.unique(np.linspace(0,len(z)-1,256,dtype=int))
    checked=np.flatnonzero(np.isfinite(grid['reference']))
    delta=grid['reference']-grid['fine']
    a=weights[representative];a=a/a.sum()
    b=softmax(np.log(a)+delta[representative])
    values=quantities(z,data)
    ma,sa,_=n.weighted_summary(values[representative],a)
    mb,_,_=n.weighted_summary(values[representative],b)
    shift=float(np.max(np.abs(mb-ma)/np.maximum(sa,1e-12)))
    center=float(a@delta[representative]);error=np.abs(delta[representative]-center)
    order=np.argsort(error);q95=float(np.interp(.95,np.cumsum(a[order]),error[order]))
    maximum=float(np.max(np.abs(delta[checked]-center)))
    return dict(maximum_mean_shift_in_sd=shift,centered_logdensity_q95=q95,
        centered_logdensity_maximum=maximum,passed=shift<.05 and q95<.1 and maximum<.25)


def source(dataset,model):
    if dataset.startswith('recovery_'):
        policy=dataset.removeprefix('recovery_');cid=4657
        root=p.OUT/'recovery_4657'/policy/'activity_scale'/model
        data=json.loads((recovery.FIXTURE/policy/'interval_start/run.json').read_text())['inputs']
        truth=json.loads((recovery.FIXTURE/policy/'truth.json').read_text())['truth']
    else:
        cid=int(dataset);data,_,_,_=n.load_baseline(cid);truth=None
        root=p.OUT/'scale_weak'/model
    if model=='nonlinear':
        path=root/f'posterior_grid_{cid}.npz'
        gate=json.loads(path.with_suffix('.json').read_text())
        if gate['status']!='validated':raise RuntimeError('Unvalidated source grid')
        grid=np.load(path);z=grid['z'];bw=softmax(grid['fine']-grid['base'])
        paths=[path,path.with_suffix('.json')]
    else:
        if dataset=='4699':
            root=audit.OUT/'contest_4699/scale_weak_delta99'
            path=root/'posterior.npz';data=json.loads((root/'run.json').read_text())['inputs']
            _,diag,_=report.leading_result(root,data)
            if not diag['passed']:raise RuntimeError('Unvalidated leading source')
            z=np.load(path)['values'].transpose(1,0,2).reshape(-1,6).copy();z[:,:4]=np.log(z[:,:4]);paths=[path]
        else:
            paths=[root/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
            z=np.concatenate([np.load(path)['z'] for path in paths])
            chain=quantities(z,data).reshape(4,-1,len(report.NAMES)+1)
            if max(n.rank_rhat(chain[:,:,k]) for k in range(chain.shape[2]))>1.01 or min(min(n.rank_ess(chain[:,:,k])) for k in range(chain.shape[2]))<400:
                raise RuntimeError(f'Unvalidated source: {dataset}/{model}')
        bw=np.ones(len(z))/len(z)
    return z,bw,data,truth,{str(path.relative_to(n.ROOT)):n.digest(path) for path in paths}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(candidates=CANDIDATES,definition='u=(log ci+log cj)/2, d=log ci-log cj; independent centered Gaussian priors on u,d.',
        basis='Original independent cost log-SD .75 implies relative log-cost SD sqrt(2)*.75. Preserve it while separating common-scale uncertainty.',
        acceptance='Weight ESS and dependence-adjusted mean ESS >=400; max normalized weight <.02, all reported quantities including cost ratio.',
        selection='No automatic selection by w, agreement or single-control truth. First-stage sensitivity only.')
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=design:
        previous=json.loads(path.read_text())
        if {k:v for k,v in previous.items() if k!='acceptance'}!={k:v for k,v in design.items() if k!='acceptance'}:
            raise RuntimeError('Existing trial design differs')
    n.save_json(path,design)
    results=[];rows=[]
    for dataset,models in [('recovery_leading_control',['leading_fast']),('recovery_nonlinear',['leading_fast','nonlinear']),('4699',['leading_fast','nonlinear']),('4657',['leading_fast','nonlinear']),('5390',['leading_fast','nonlinear'])]:
        for model in models:
            z,bw,data,truth,hashes=source(dataset,model)
            _,_,base=evaluate(z,bw,data,dict(common_sd=1.25/math.sqrt(2),relative_sd=1.25*math.sqrt(2),sigma_sd=.5,lambda_sd=.75))
            for name,candidate in CANDIDATES.items():
                weights,diag,summaries=evaluate(z,bw,data,candidate)
                if model=='nonlinear':
                    diag['grid_check']=grid_check(dataset,weights,data)
                    diag['passed']=diag['passed'] and diag['grid_check']['passed']
                item=dict(dataset=dataset,model=model,candidate=name,diagnostics=diag,source_sha256=hashes)
                if diag['passed']:
                    item['summaries']=summaries;item['baseline_summaries']=base
                    if truth:
                        target=dict(truth,cost_ratio=truth['c_i']/truth['c_j'])
                        item['truth']=target
                        item['truth_in_95_interval']={key:summaries[key]['q025']<=target[key]<=summaries[key]['q975'] for key in n.NAMES+('cost_ratio',)}
                    for key in report.NAMES+('cost_ratio',):
                        row=dict(dataset=dataset,model=model,candidate=name,parameter=key,baseline_median=base[key]['median'],
                            trial_median=summaries[key]['median'],trial_q025=summaries[key]['q025'],trial_q975=summaries[key]['q975'])
                        if truth and key in item['truth']:row['truth']=item['truth'][key]
                        else:row['truth']=''
                        rows.append(row)
                results.append(item)
                print(dataset,model,name,json.dumps(diag),flush=True)
    n.save_json(OUT/'results.json',results);n.write_csv(OUT/'comparison.csv',rows)
    lines=['共同成本与相对成本先验：第一阶段试验','',
        '三个候选事先固定：保留原相对成本不确定性；额外放宽共同成本尺度；最后额外放宽sigma/lambda的对数标准差。所有候选均在两种估计中相同。',
        '这里只改变先验，通过有效后验重加权比较。未通过有效样本门槛的候选不报告参数结论。原有估计和完成清单保持不变。','']
    for item in results:
        diag=item['diagnostics'];lines.append(f"{item['dataset']} / {item['model']} / {item['candidate']}: {'通过' if diag['passed'] else '未通过'}；权重ESS={diag['weight_ess']:.0f}，均值最低ESS={diag['minimum_mean_ess']:.0f}")
        if diag['passed'] and 'truth' in item:
            for key in ('c_i','c_j','r','cost_ratio','sigma','lambda'):
                s=item['summaries'][key];b=item['baseline_summaries'][key]
                lines.append(f"  {key}: 真实={item['truth'][key]:.5g}；原中位数={b['median']:.5g}；新中位数={s['median']:.5g}；95%区间=[{s['q025']:.5g},{s['q975']:.5g}]")
    lines+=['','这是先验敏感性试验。区间收缩或单次恢复变好均不足以确定应替换主先验；多次模拟的覆盖与预测表现尚待检验。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    n.save_json(OUT/'completion.json',dict(status='screen_complete',script_sha256=n.digest(__file__),
        artifacts={name:n.digest(OUT/name) for name in ('design.json','results.json','comparison.csv','report.txt')}))


if __name__=='__main__':run()
