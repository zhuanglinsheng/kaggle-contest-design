"""Report the completed cost-prior attempt without selecting by agreement."""
import json
import math
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import cost_prior_trial as trial


def run():
    results=json.loads((trial.OUT/'results.json').read_text())
    freshroot=trial.OUT/'fresh_relative_cost/leading_fast'
    fresh=json.loads((freshroot/'summary.json').read_text())
    if not fresh['diagnostics']['passed']:raise RuntimeError('Fresh 5390 fit failed')
    paths=[freshroot/f'contest_5390/chain_{j}.npz' for j in range(1,5)]
    if fresh['chain_sha256']!=[n.digest(path) for path in paths]:raise RuntimeError('Fresh chains changed')
    rows=[]
    for cid in ('4699','4657','5390'):
        pair={x['model']:x for x in results if x['dataset']==cid and x['candidate']=='relative_cost_preserved'}
        lhs=fresh['summaries'] if cid=='5390' else pair['leading_fast']['summaries']
        rhs=pair['nonlinear']['summaries']
        for key in trial.report.NAMES+('cost_ratio',):
            l,r=lhs[key],rhs[key]
            rows.append(dict(contest_id=cid,parameter=key,leading_median=l['median'],nonlinear_median=r['median'],
                leading_q025=l['q025'],leading_q975=l['q975'],nonlinear_q025=r['q025'],nonlinear_q975=r['q975'],
                mean_gap_in_pooled_sd=abs(l['mean']-r['mean'])/math.sqrt((l['sd']**2+r['sd']**2)/2)))
    n.write_csv(trial.OUT/'validated_paired_comparison.csv',rows)
    lines=['成本先验进一步优化：已完成试验','',
        '结论：分开设定共同成本与相对成本有小幅改善，但没有消除一阶与非线性估计的实质差距。额外放宽共同尺度或同时放宽sigma/lambda没有一致改善；本轮不替换主先验。',
        '', '试验设计',
        'u=(log ci+log cj)/2表示共同成本水平，d=log ci-log cj表示相对成本。两者独立服从均值0的正态先验。',
        '候选A：u的标准差1.25/sqrt(2)，d的标准差0.75*sqrt(2)。共同成本不确定性与此前scale_weak一致，相对成本不确定性恢复到原论文先验的水平。成本比的先验95%区间约[0.125,8.00]，并未要求两人成本相同。',
        '候选B：u的标准差进一步放宽至1.25，d保持同A。候选C：在B基础上，sigma/lambda对数先验标准差由0.5/0.75放宽至0.75/1.25，中心不变。',
        '每个候选在两种估计中完全相同。真实数据保持r先验中位数10、对数标准差1；已知参数对照保持上一轮activity_scale的r先验80、1.5。不跨这两种不同的r先验进行直接比较。',
        '', '已知参数对照：候选A的效果',
        '非线性生成数据，非线性估计：',
        '  ci：真实1.452，原中位数1.903，新中位数1.739；相对误差约31%→20%。',
        '  r：真实129.7，原181.6，新173.0；相对误差约40%→33%。',
        '  ci/cj：真实1.128，原1.636，新1.538；相对误差约45%→36%。',
        '  cj：真实1.286，原1.174，新1.139；此项反而略远离真实值。因此是部分参数改善，并非全面提高准确性。',
        '  六个原始参数仍有五个真实值在95%区间内，mu_0仍未覆盖。',
        '同一非线性数据，一阶估计：成本比5.942→5.663，95%区间约[3.78,8.93]，仍不包含真实1.128。先验调整不能纠正固定努力比例的结构限制。',
        '自身一阶生成数据：候选A的估计基本保持原先恢复表现；不是通过牺牲该对照来缩小另一组差距。',
        '', '扩大先验不一定更好',
        '候选B在非线性对照中使ci/r中位数约1.945/184.6，较上一轮没有改善。候选C使sigma与lambda更接近真实值，却使ci/r约2.314/211.5，尺度恢复变差；不能只挑改善的参数。',
        '', '真实比赛：候选A两种估计的中位数',
        '比赛 | sigma（一阶/非线性） | r（一阶/非线性） | ci/cj（一阶/非线性）']
    for cid in ('4699','4657','5390'):
        selected={r['parameter']:r for r in rows if r['contest_id']==cid}
        lines.append(cid+' | '+' | '.join(f"{selected[k]['leading_median']:.5g}/{selected[k]['nonlinear_median']:.5g}" for k in ('sigma','r','cost_ratio')))
    lines+=['',
        '4699与4657两种估计的主要差距仍在；5390的sigma约0.120与0.754，亦未解决。不能据此认定一阶近似偏差可接受。',
        '', '核验与边界',
        f"候选A的5390一阶重加权有效样本不足，已改为四链直接采样：最大Rhat={fresh['diagnostics']['maximum_rank_rhat']:.5f}，最低ESS={fresh['diagnostics']['minimum_ess']:.0f}。仅采用通过诊断的新结果。",
        '所有报告的非线性重加权结果同时检查权重集中、链内依赖和新权重下更细数值网格误差。未通过门槛的B/C真实数据结果只记录诊断，不报告参数结论。',
        '先验变换与相关二维正态密度逐点核对，并验证交换参赛者标签不改变先验。两项单元检验通过。',
        '本轮只完成已有模拟实现与三场真实比赛的先验敏感性试验，没有新增多次随机模拟；不能根据单次恢复选择出版用主先验。',
        '建议保留候选A用于后续重复模拟与预测检验。B/C没有提供一致改善，不建议仅为调整估计数值继续放宽。原论文、主估计及Table 7未改动。']
    (trial.OUT/'final_report.txt').write_text('\n'.join(lines)+'\n')
    sources=['cost_prior_trial.py','cost_prior_fresh_leading.py','cost_prior_trial_report.py']
    artifacts=['design.json','results.json','comparison.csv','report.txt','validated_paired_comparison.csv','final_report.txt','fresh_relative_cost/leading_fast/summary.json','fresh_relative_cost/leading_fast/target.json']
    n.save_json(trial.OUT/'final_completion.json',dict(status='completed_prior_sensitivity_attempt',
        source_sha256={name:n.digest(n.ROOT/'pipeline/empirical'/name) for name in sources},
        artifact_sha256={name:n.digest(trial.OUT/name) for name in artifacts},
        fresh_chain_sha256={str(path.relative_to(n.ROOT)):n.digest(path) for path in paths},
        scope='Existing retained realizations plus three empirical contests. No repeated-simulation coverage claim or publication prior selection.'))
    print(trial.OUT/'final_report.txt')


if __name__=='__main__':run()
