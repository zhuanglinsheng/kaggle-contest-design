"""Report common-prior estimation separately from numerical/approximation errors."""
from __future__ import annotations
import json
import math
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import paired_prior_improvement as p


def quantities(z,data):
    v=n.physical_draws(z,data['theta'])
    ci,cj,sigma,lam,r=v[:,0],v[:,1],v[:,2],v[:,3],v[:,5]
    # Identified combinations may be more stable than individual scale parameters.
    return np.column_stack((v,data['theta']/ci,data['theta']/cj,r/ci,r/cj,
          sigma*np.sqrt(lam),1/(sigma*data['T_contest']*np.sqrt(lam))))


NAMES=n.NAMES+('w_i','w_j','max_w','theta_over_ci','theta_over_cj',
             'r_over_ci','r_over_cj','learning_rate','normalized_uncertainty')


def summary(z,weights,data):
    values=quantities(z,data)
    means,sds,q=n.weighted_summary(values,weights)
    return {key:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,key in enumerate(NAMES)}


def leading_source(cid,profile):
    folder=p.OUT/profile/'leading'/f'contest_{cid}'
    if (folder/'posterior.npz').exists():return folder
    if cid==4699 and profile=='scale_weak':return a.OUT/f'contest_{cid}/scale_weak_delta99'
    if cid==4657 and profile=='r_weak':return a.OUT/f'contest_{cid}/r_weak'
    return None


def fast_leading_result(cid,profile,data):
    folder=p.OUT/profile/'leading_fast'/f'contest_{cid}'
    path=folder/'summary.json'
    if not path.exists():return None
    saved=json.loads(path.read_text())
    paths=[folder/f'chain_{j}.npz' for j in range(1,5)]
    if saved['chain_sha256']!=[n.digest(path) for path in paths]:raise RuntimeError('Fast posterior changed after summary')
    z=np.array([np.load(path)['z'] for path in paths]);flat=z.reshape(-1,6)
    values=quantities(flat,data).reshape(4,z.shape[1],-1)
    rh=max(n.rank_rhat(values[:,:,k]) for k in range(values.shape[2]))
    ess=min(min(n.rank_ess(values[:,:,k])) for k in range(values.shape[2]))
    diag=dict(max_rank_rhat=rh,min_ess=ess,passed=rh<=1.01 and ess>=400,
              sampler='Four-chain exact-target Metropolis; HMC energy diagnostics do not apply.')
    return summary(flat,np.ones(len(flat))/len(flat),data),diag,folder


def leading_result(folder,data):
    x=np.load(folder/'posterior.npz')['values'].transpose(1,0,2).copy()
    x[:,:,:4]=np.log(x[:,:,:4]);flat=x.reshape(-1,6)
    v=quantities(flat,data).reshape(4,x.shape[1],-1)
    rh=max(n.rank_rhat(v[:,:,k]) for k in range(v.shape[2]))
    ess=min(min(n.rank_ess(v[:,:,k])) for k in range(v.shape[2]))
    run=json.loads((folder/'run.json').read_text())
    if (folder/'sampler.json').exists():sampler=json.loads((folder/'sampler.json').read_text())
    else:
        from cmdstanpy import from_csv
        fit=from_csv([str(path) for path in sorted(folder.glob('detail_model-*.csv'))])
        method=fit.method_variables();energy=method['energy__']
        sampler=dict(divergences=int(method['divergent__'].sum()),treedepth_hits=int((method['treedepth__']>=12).sum()),
          minimum_bfmi=float(np.min(np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0))))
    passed=rh<=1.01 and ess>=400 and sampler['divergences']==0 and sampler['treedepth_hits']==0 and sampler['minimum_bfmi']>=.3
    return summary(flat,np.ones(len(flat))/len(flat),data),dict(max_rank_rhat=rh,min_ess=ess,passed=bool(passed),**sampler),run


def report():
    comparisons=[];statuses=[]
    for profile in p.PROFILES:
        for cid in a.IDS:
            data,_,_,_=n.load_baseline(cid)
            folder=leading_source(cid,profile)
            lhs=None
            if folder is not None:
                lhs,diag,run=leading_result(folder,data)
                statuses.append(dict(contest_id=cid,profile=profile,model='leading',source=str(folder),**diag))
            fast=fast_leading_result(cid,profile,data)
            if fast is not None:
                fast_summary,fast_diag,fast_folder=fast
                statuses.append(dict(contest_id=cid,profile=profile,model='leading_fast',source=str(fast_folder),**fast_diag))
                if fast_diag['passed'] and (lhs is None or not diag['passed']):
                    lhs,diag=fast_summary,fast_diag
            root=p.OUT/profile/'nonlinear'
            check=root/f'posterior_grid_{cid}.json'
            if not check.exists():
                statuses.append(dict(contest_id=cid,profile=profile,model='nonlinear',status='not_validated'));continue
            gate=json.loads(check.read_text())
            statuses.append(dict(contest_id=cid,profile=profile,model='nonlinear',status=gate['status']))
            if gate['status']!='validated' or lhs is None or not diag['passed']:continue
            grid=np.load(root/f'posterior_grid_{cid}.npz');weights=softmax(grid['fine']-grid['base'])
            rhs=summary(grid['z'],weights,data)
            for key in NAMES:
                l,r=lhs[key],rhs[key]
                comparisons.append(dict(contest_id=cid,profile=profile,parameter=key,
                   leading_mean=l['mean'],nonlinear_mean=r['mean'],
                   leading_median=l['median'],nonlinear_median=r['median'],
                   leading_q025=l['q025'],leading_q975=l['q975'],nonlinear_q025=r['q025'],nonlinear_q975=r['q975'],
                   absolute_mean_gap=abs(l['mean']-r['mean']),
                   gap_in_pooled_sd=abs(l['mean']-r['mean'])/math.sqrt((l['sd']**2+r['sd']**2)/2),
                   relative_median_gap=abs(l['median']-r['median'])/max(abs(r['median']),1e-12)))
    n.save_json(p.OUT/'status.json',statuses)
    if comparisons:n.write_csv(p.OUT/'paired_comparison.csv',comparisons)
    lines=['共同先验改进：一阶与非线性估计','',
       '共同候选：r的对数标准差0.5→1；第二个候选同时将两人的成本对数标准差0.75→1.25。所有先验中心、创新风险先验、数据和奖金单位保持不变。',
       '先验候选依据此前已知参数恢复与提交分配问题确定；不以w下降或两模型接近为选择标准。',
       '参数比较只有在双方抽样诊断及非线性后验网格检查通过后才进入表格。',
       '95%区间相交、w较小、单次模拟恢复均不能单独证明一阶近似可用于设计。Table 7仍需与估计一致的非线性反事实检验。','']
    for s in statuses:
        lines.append(f"{s['contest_id']} / {s['profile']} / {s['model']}: "+
                     (f"诊断{'通过' if s['passed'] else '未通过'}，Rhat={s['max_rank_rhat']:.4f}, 最低ESS={s['min_ess']:.0f}" if 'passed' in s else s['status']))
    for profile in p.PROFILES:
        for cid in a.IDS:
            rows=[row for row in comparisons if row['contest_id']==cid and row['profile']==profile]
            if not rows:continue
            lines+=['',f'{cid} / {profile}：参数 | 一阶中位数 | 非线性中位数 | 均值差/合并后验SD']
            lines.extend(f"{r['parameter']} | {r['leading_median']:.5g} | {r['nonlinear_median']:.5g} | {r['gap_in_pooled_sd']:.3f}" for r in rows)
    (p.OUT/'comparison_report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':report()
