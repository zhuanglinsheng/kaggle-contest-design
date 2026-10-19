"""Report successful checks and rejected lambda refinements without cherry picking."""
import json
import math
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import lambda_prior_refinement as r
from pipeline.empirical.sigma_interval_calibration import calibrated_interval


def load(rep,cid,method):
 path=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/method/'summary.json';value=json.loads(path.read_text())
 if not value['passed']:raise RuntimeError(f'Unvalidated posterior {path}')
 chains=[path.parent/f'contest_{cid}/chain_{j}.npz' for j in range(1,5)]
 if value['chain_sha256']!=[n.digest(p) for p in chains]:raise RuntimeError('Posterior changed')
 return value,path


def row(rep,cid,method,fit,source):
 t=fit['truth']['lambda'];v=fit['summaries']['lambda']
 return dict(replicate=rep,contest_id=cid,method=method,truth=t,mean=v['mean'],median=v['median'],q025=v['q025'],q975=v['q975'],
  median_relative_error=abs(v['median']/t-1),mean_relative_error=abs(v['mean']/t-1),
  original_interval_covers=v['q025']<=t<=v['q975'],sigma_relative_error=abs(fit['summaries']['sigma']['median']/fit['truth']['sigma']-1),source=source)


def metrics(rows):
 return dict(cases=len(rows),mean_relative_error_median=float(np.mean([v['median_relative_error'] for v in rows])),
  mean_relative_error_mean=float(np.mean([v['mean_relative_error'] for v in rows])),
  maximum_relative_error_median=float(max(v['median_relative_error'] for v in rows)),
  maximum_relative_error_mean=float(max(v['mean_relative_error'] for v in rows)),
  covered=sum(v['original_interval_covers'] for v in rows),
  sigma_mean_relative_error=float(np.mean([v['sigma_relative_error'] for v in rows])))


def run():
 chosen=json.loads((r.OUT/'selection.json').read_text())['selected']['variant']
 historical=json.loads((r.OUT/'retrospective.json').read_text());rows=[];sources={}
 for rep in (2,3,4,5):
  for cid in s.IDS:
   value,path=load(rep,cid,'lambda_external');rows.append(row(rep,cid,'lambda_external',value,'validated posterior'))
   sources[str(path.relative_to(n.ROOT))]=n.digest(path)
   path=s.OUT/'posterior/recovery'/f'{cid}_{rep}'/chosen/'summary.json'
   if path.exists():
    value,path=load(rep,cid,chosen);sources[str(path.relative_to(n.ROOT))]=n.digest(path)
    rows.append(row(rep,cid,chosen,value,'fresh posterior'))
   else:
    value=next(v for v in historical if v['replicate']==rep and v['contest_id']==cid)
    if not value['passed']:raise RuntimeError('Failed importance update requires fresh posterior')
    rows.append(row(rep,cid,chosen,value,'validated importance update'))
 n.write_csv(r.OUT/'validated_lambda_comparison.csv',rows)
 allmetrics={method:{str(rep):metrics([x for x in rows if x['method']==method and x['replicate']==rep]) for rep in (2,3,4,5)} for method in ('lambda_external',chosen)}
 for method in allmetrics:allmetrics[method]['combined']=metrics([x for x in rows if x['method']==method])
 n.save_json(r.OUT/'validation_metrics.json',allmetrics)
 # Arithmetic means under the rejected broad prior need a separate precision
 # audit; rank ESS alone does not bound the MC error of a long-tailed mean.
 moment_audit=[]
 for method in ('lambda_external',chosen):
  for cid in s.IDS:
   value,path=load(5,cid,method)
   lambdas=np.array([np.exp(np.load(path.parent/f'contest_{cid}/chain_{j}.npz')['z'][:,3]) for j in range(1,5)])
   ess=n.effective_sample_size(lambdas);relative_mcse=float(lambdas.std(ddof=1)/math.sqrt(ess)/lambdas.mean())
   moment_audit.append(dict(method=method,contest_id=cid,mean_ess=ess,relative_mcse=relative_mcse,mean_stable_at_5pct=relative_mcse<=.05))
 n.save_json(r.OUT/'posterior_mean_audit.json',moment_audit)
 intervals=json.loads((r.OUT/'lambda_interval_metrics.json').read_text())
 factor=json.loads((r.OUT/'lambda_interval_frozen.json').read_text())['multiplier']
 empirical=[]
 for cid in s.IDS:
  path=s.OUT/'posterior/empirical'/str(cid)/'lambda_external/summary.json';value=json.loads(path.read_text())
  if not value['passed']:raise RuntimeError('Invalid empirical fit')
  v=value['summaries']['lambda'];lo,hi=calibrated_interval(v['median'],v['q025'],v['q975'],factor)
  empirical.append(dict(contest_id=cid,posterior_mean=v['mean'],posterior_median=v['median'],posterior_q025=v['q025'],posterior_q975=v['q975'],
   calibrated_lower=lo,calibrated_upper=hi,mean_over_median=v['mean']/v['median'],truth='unknown',coverage='not evaluated'))
  sources[str(path.relative_to(n.ROOT))]=n.digest(path)
 n.write_csv(r.OUT/'empirical_lambda_summaries.csv',empirical)
 base=allmetrics['lambda_external'];candidate=allmetrics[chosen]
 n.save_json(r.OUT/'decision.json',dict(prior_candidate=chosen,decision='reject_broader_prior_after_prospective_test',
  retained_prior=dict(logmean=math.log(100.),logsd=2.25),
  prospective_current=base['5'],prospective_candidate=candidate['5'],
  point_summary='Keep previous experimental median convention. Mean-vs-median sensitivity is explicit; the paper currently reports means and is not silently changed.',
  interval='Additional frozen simulation-calibrated bounds; not original95% posterior credible intervals.'))
 lines=['lambda进一步完善：独立验证与右尾稳定性','',
  '结论：不采用进一步放宽/提高中心的lambda先验。它在训练中略好，却在新模拟中明显变差。保留此前log(lambda)~N(log100,2.25^2)的候选方案，并明确后验均值/中位数的差异，补充单独标识的区间校准。',
  '', '本轮设计',
  '12个先验组合：中心30/100/300/1000，对数标准差1.5/2.25/3。只用探索实现0/1的20个后验筛选，保持sigma、成本、r先验和一阶努力函数不变。',
  f'训练选出{chosen}，但训练覆盖仅18/20，未达到预期19/20；这一点保留在selection.json中。冻结后生成新实现5，每个配置一场，不更换不理想的随机种子。',
  '既有实现2/3/4仅作事后检查。三个重加权结果未通过有效样本门槛，已改为四链直接采样，通过后才进入比较。',
  '', '各组lambda平均绝对相对误差（统一的后验中位数口径）',
  '实现 | 当前候选先验 | 更宽先验 | 当前区间覆盖 | 更宽先验区间覆盖']
 for rep in ('2','3','4','5'):
  a,b=base[rep],candidate[rep];lines.append(f"{rep} | {a['mean_relative_error_median']:.2%} | {b['mean_relative_error_median']:.2%} | {a['covered']}/10 | {b['covered']}/10")
 lines+=['', '后验均值的右尾问题',
  '核对原文后确认：论文使用后验均值，而此前试验表格统一使用后验中位数。这里两者并列报告，不将汇总口径变化冒充参数后验精度提高。',
  f"新实现5中，当前先验的均值误差为{base['5']['mean_relative_error_mean']:.2%}，中位数误差为{base['5']['mean_relative_error_median']:.2%}。更宽先验还出现个别后验均值相对Monte Carlo误差超过5%，故其均值不作精确性能结论。",
  f"四组验证共40个配置实现：当前先验中位数平均误差{base['combined']['mean_relative_error_median']:.2%}，均值平均误差{base['combined']['mean_relative_error_mean']:.2%}；中位数最差单场误差{base['combined']['maximum_relative_error_median']:.2%}。平均达标不表示每场误差均小于20%。",
  '如果论文的验收标准采用此前的中位数口径，应在参数汇总表中明确该口径，并保留后验区间和均值敏感性；这不改变联合后验或反事实积分。正文尚未改动。',
  '', 'lambda区间校准',
  f'沿用sigma的规则，只用实现0/1选取一个共同系数k={factor:.5f}，在检验新实现5之前冻结。原lambda后验和点估计不变，对数区间半宽同乘k。新边界称模拟校准区间，不能称为原95%后验可信区间。',
  '实现 | 原后验区间覆盖 | 模拟校准区间覆盖 | 原相对宽度中位数 | 校准后相对宽度中位数']
 for rep in ('2','3','4','5'):
  v=intervals[rep];lines.append(f"{rep} | {v['original_covered']}/10 | {v['calibrated_covered']}/10 | {v['original_median_relative_width']:.1%} | {v['calibrated_median_relative_width']:.1%}")
 covered=sum(intervals[str(rep)]['original_covered'] for rep in (2,3,4,5));calibrated=sum(intervals[str(rep)]['calibrated_covered'] for rep in (2,3,4,5))
 lines += ['',f'四组原区间合计{covered}/40；模拟校准区间{calibrated}/40。覆盖提升伴随区间变宽，不代表点估计误差下降，也不保证总体95%覆盖。',
  '', '使用边界与交付',
  '同一十种参数配置的反复模拟，不是73场总体精度验证。真实比赛没有已知lambda真值，不报告真实数据覆盖率；empirical_lambda_summaries.csv仅并列给出均值、中位数、原后验边界和应用相同规则的校准边界。',
  '原论文、主估计、Table7、既有完成清单和sigma校准系数均未替换。新报告保留失败候选和新模拟结果；不因新模拟不理想再选种子。']
 (r.OUT/'report.txt').write_text('\n'.join(lines)+'\n')
 artifacts=['design.json','selection.json','training.json','retrospective.json','point_summary_design.json','validated_lambda_comparison.csv','validation_metrics.json','posterior_mean_audit.json','decision.json','lambda_interval_frozen.json','lambda_interval_comparison.csv','lambda_interval_metrics.json','empirical_lambda_summaries.csv','report.txt']
 scripts=['lambda_prior_refinement.py','lambda_retrospective_refits.py','lambda_interval_check.py','lambda_refinement_report.py']
 n.save_json(r.OUT/'completion.json',dict(status='completed_lambda_refinement_with_rejected_broad_prior',
  artifact_sha256={name:n.digest(r.OUT/name) for name in artifacts},source_sha256={name:n.digest(n.ROOT/'pipeline/empirical'/name) for name in scripts},source_summary_sha256=sources))
 print(r.OUT/'report.txt',flush=True)


if __name__=='__main__':run()
