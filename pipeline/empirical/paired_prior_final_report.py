"""Consolidate validated fits, retained controls, and excluded experiments."""
import json
import math
import numpy as np
from scipy.special import softmax
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_prior_recovery as recovery
from pipeline.empirical import paired_prior_report as rep


def control(policy,model):
    root=p.OUT/'recovery_4657'/policy/'activity_scale'/model
    paths=[root/f'contest_4657/chain_{j}.npz' for j in range(1,5)]
    z=np.array([np.load(path)['z'] for path in paths])
    data=json.loads((recovery.FIXTURE/policy/'interval_start/run.json').read_text())['inputs']
    truth=json.loads((recovery.FIXTURE/policy/'truth.json').read_text())['truth']
    values=rep.quantities(z.reshape(-1,6),data).reshape(4,z.shape[1],-1)
    values=np.concatenate((values,np.exp(z[:,:,0]-z[:,:,1])[:,:,None]),axis=2)
    diagnostics=dict(maximum_rank_rhat=max(n.rank_rhat(values[:,:,k]) for k in range(values.shape[2])),
        minimum_ess=min(min(n.rank_ess(values[:,:,k])) for k in range(values.shape[2])))
    if diagnostics['maximum_rank_rhat']>1.01 or diagnostics['minimum_ess']<400:
        raise RuntimeError(f'Control diagnostics failed: {policy}/{model}')
    if model=='nonlinear':
        gate=json.loads((root/'posterior_grid_4657.json').read_text())
        if gate['status']!='validated':raise RuntimeError('Nonlinear control grid failed')
        grid=np.load(root/'posterior_grid_4657.npz');flat=grid['z'];weights=softmax(grid['fine']-grid['base'])
        diagnostics['numerical_validation']=gate
    else:
        flat=z.reshape(-1,6);weights=np.ones(len(flat))/len(flat)
    summaries=rep.summary(flat,weights,data)
    means,sds,q=n.weighted_summary(np.exp(flat[:,0]-flat[:,1])[:,None],weights)
    summaries['cost_ratio']=dict(mean=float(means[0]),sd=float(sds[0]),q025=float(q[0,0]),median=float(q[0,1]),q975=float(q[0,2]))
    truth['cost_ratio']=truth['c_i']/truth['c_j']
    return dict(policy_generating_data=policy,estimator=model,diagnostics=diagnostics,summaries=summaries,truth=truth,
        coverage={k:summaries[k]['q025']<=truth[k]<=summaries[k]['q975'] for k in n.NAMES+('cost_ratio',)},
        source_sha256={str(path.relative_to(n.ROOT)):n.digest(path) for path in paths},
        scope='One preserved realization; not a repeated-simulation accuracy or coverage study.')


def run():
    rep.report()
    empirical=[]
    for cid in (4699,4657,5390):
        root=p.OUT/'scale_weak/nonlinear'
        paths=[root/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
        z=np.array([np.load(path)['z'] for path in paths])
        data,_,_,_=n.load_baseline(cid)
        values=rep.quantities(z.reshape(-1,6),data).reshape(4,z.shape[1],-1)
        values=np.concatenate((values,np.exp(z[:,:,0]-z[:,:,1])[:,:,None]),axis=2)
        diag=dict(maximum_rank_rhat=max(n.rank_rhat(values[:,:,k]) for k in range(values.shape[2])),
            minimum_ess=min(min(n.rank_ess(values[:,:,k])) for k in range(values.shape[2])))
        if diag['maximum_rank_rhat']>1.01 or diag['minimum_ess']<400:
            raise RuntimeError(f'Extended nonlinear diagnostics failed: {cid}')
        gate=json.loads((root/f'posterior_grid_{cid}.json').read_text())
        if gate['status']!='validated':raise RuntimeError('Empirical posterior grid failed')
        empirical.append(dict(contest_id=cid,extended_diagnostics=diag,numerical_validation=gate,
            source_sha256={str(path.relative_to(n.ROOT)):n.digest(path) for path in paths},
            posterior_sha256=n.digest(root/f'posterior_grid_{cid}.npz')))
    n.save_json(p.OUT/'validated_empirical_fits.json',empirical)
    controls=[control('leading_control','leading_fast'),control('nonlinear','leading_fast'),control('nonlinear','nonlinear')]
    n.save_json(p.OUT/'validated_recovery_comparison.json',controls)
    lines=['先验与估计细节改进：最终核验报告','',
        '结论：两种估计均已实施先验敏感性改进；非线性估计已有三场真实比赛的可用共同先验结果和一个已知参数对照。尚不能认为一阶估计与非线性估计的差距普遍可接受。',
        '', '1. 改动范围',
        'scale_weak：r的对数标准差由0.5放宽至1，两人成本的对数标准差由0.75放宽至1.25，先验中心不变。两种估计使用完全相同的先验。',
        'activity_scale：作为另一个共同先验敏感性方案，r先验中位数80、对数标准差1.5，成本对数标准差1.25。80取自排除原最高w十场后的公共质量增量/提交量描述性尺度（中位数约82），不是r的直接估计；该方案尚未替换论文主先验。',
        '保留奖金单位、数据、似然和努力公式。模拟估计对齐区间起点反馈；修复新增并行计算中先验未传入子进程的问题。采用固定混合提议改善后验尾部探索；此操作不改变先验或似然。',
        '', '2. 已知参数对照（同一activity_scale先验）',
        '生成机制 | 估计方法 | ci | cj | sigma | lambda | r | ci/cj']
    truth=controls[0]['truth']
    lines.append('真实参数 | — | '+' | '.join(f'{truth[k]:.5g}' for k in ('c_i','c_j','sigma','lambda','r','cost_ratio')))
    for c in controls:
        lines.append(c['policy_generating_data']+' | '+c['estimator']+' | '+' | '.join(f"{c['summaries'][k]['median']:.5g}" for k in ('c_i','c_j','sigma','lambda','r','cost_ratio')))
    lines+=['',
        '自身一阶生成数据：旧先验下ci、cj、r的中位数约0.434、0.347、38.8；新先验下约1.432、1.052、122.0，真实值约1.452、1.286、129.7。六个原始参数的真实值均在新95%区间内，但部分尺度区间仍较宽。',
        '非线性生成数据：一阶成本比的95%区间约[3.93,9.52]，非线性约[0.96,4.26]，真实比值约1.13。非线性估计六个原始参数中五个覆盖真实值，mu_0未覆盖；r中位数误差约40%，不能宣称所有参数都精确恢复。',
        '原因：一阶式两人的努力共享同一个状态函数，因此mi/mj=cj/ci恒定。在这组非线性数据中，提交量22比137；一阶估计容易将这种不平衡解释为成本差异，而非线性策略的努力比例可以随状态改变。尺度先验不能消除这一结构限制。',
        '两种生成机制的实现路径不同；只在同一非线性数据上的两种估计之间进行直接比较，不将两组路径的误差相减作为偏差分解。',
        '', '3. 真实数据共同先验比较',
        '详见paired_comparison.csv及comparison_report.txt。4699的部分原始参数差距减小，但4657仍有多项超过两个合并后验标准差；5390的sigma相差约5.51个合并后验标准差。5390采用改进固定提议后通过诊断的一阶结果。不能据此认可一阶近似。',
        'activity_scale从scale_weak非线性后验重加权的三场结果均未通过有效样本门槛，故不用于结论。新activity_scale一阶结果只作为先验敏感性结果。',
        '', '4. 核验与使用建议',
        '共同先验下三场非线性抽样及后验数值网格检查通过；更细网格造成的最大均值变化小于0.01个后验标准差。已知参数非线性对照亦通过。具体诊断、哈希和区间见JSON文件。',
        '可以满意的是：先验造成的尺度压缩已有明确改善，并取得了数值核验通过的非线性比较基准。尚未解决的是：一阶近似在高w样本中的统计偏差以及设计结果对该偏差的敏感性。',
        '建议将共同先验的非线性估计作为后续扩展的基准；不要继续以压低w或迫使两种后验接近来选择先验。activity_scale尚属敏感性方案，一次模拟不足以证明其在整个样本中的优越性。',
        '本轮未覆盖全部73场，也未更新Table 7、论文或计算非线性设计反事实。原最高w十场和原主估计结果保留。']
    (p.OUT/'final_report.txt').write_text('\n'.join(lines)+'\n')
    artifacts=['final_report.txt','validated_recovery_comparison.json','validated_empirical_fits.json','paired_comparison.csv','comparison_report.txt','status.json','activity_prior_design.json','activity_scale_status.json']
    sources=['paired_prior_improvement.py','paired_prior_report.py','paired_prior_recovery.py','paired_prior_mixture.py','activity_nonlinear_recovery.py','activity_recovery_mixture.py','fast_leading_prior.py','leading_on_nonlinear_control.py','leading_prior_mixture.py','paired_prior_final_report.py']
    n.save_json(p.OUT/'completion.json',dict(status='completed_scoped_improvement',
        scope='Three empirical scale_weak nonlinear fits and retained common activity_scale recovery controls; no full-sample or design replacement.',
        artifact_sha256={name:n.digest(p.OUT/name) for name in artifacts},
        script_sha256={name:n.digest(n.ROOT/'pipeline/empirical'/name) for name in sources},
        controls=[dict(policy=c['policy_generating_data'],estimator=c['estimator'],diagnostics=c['diagnostics']) for c in controls],
        empirical=empirical))
    print('FINAL REPORT',p.OUT/'final_report.txt',flush=True)


if __name__=='__main__':run()
