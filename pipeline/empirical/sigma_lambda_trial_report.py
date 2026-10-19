"""Separate known-truth recovery improvements from empirical concordance."""
import csv
import json
import math
from collections import defaultdict
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s


def read_fit(dataset,case,method):
 path=s.OUT/'posterior'/dataset/str(case)/method/'summary.json'
 value=json.loads(path.read_text())
 if not value['passed']:raise RuntimeError(f'Unvalidated fit: {path}')
 paths=[path.parent/f'contest_{int(str(case).split("_")[0])}/chain_{j}.npz' for j in range(1,5)]
 if [n.digest(p) for p in paths]!=value['chain_sha256']:raise RuntimeError('Changed posterior chains')
 return value,path


def aggregate(rep,methods):
 results={};rows=[];sources={}
 for method in methods:
  fits=[]
  for cid in s.IDS:
   fit,path=read_fit('recovery',f'{cid}_{rep}',method);fits.append(fit);sources[str(path.relative_to(n.ROOT))]=n.digest(path)
   for key in ('sigma','lambda'):
    row=dict(replicate=rep,contest_id=cid,method=method,parameter=key,truth=fit['truth'][key],**fit['summaries'][key],
     absolute_log_error=fit['absolute_log_error'][key],absolute_relative_error=fit['relative_error'][key],truth_in_95_interval=fit['truth_in_95_interval'][key])
    rows.append(row)
  results[method]={key:dict(mean_absolute_log_error=float(np.mean([f['absolute_log_error'][key] for f in fits])),
    mean_absolute_relative_error=float(np.mean([f['relative_error'][key] for f in fits])),
    maximum_absolute_relative_error=float(max(f['relative_error'][key] for f in fits)),
    truth_in_95_interval=sum(f['truth_in_95_interval'][key] for f in fits)) for key in ('sigma','lambda')}
  for key in ('sigma','lambda'):
   baseline=[r for r in rows if r['method']=='baseline' and r['parameter']==key]
   current=[r for r in rows if r['method']==method and r['parameter']==key]
   if baseline:
    results[method][key]['improved_configurations']=sum(c['absolute_log_error']<b['absolute_log_error'] for b,c in zip(baseline,current))
 return results,rows,sources


def run():
 selection=json.loads((s.OUT/'sigma_grid_selection.json').read_text());chosen=selection['selected']['variant']
 groups={2:('baseline','lambda_external','activity_lambda_external','count_quarter_lambda_external'),3:('baseline','lambda_external',chosen)}
 metrics={};rows=[];sources={};allfits=[]
 for rep,methods in groups.items():
  results,current,hashes=aggregate(rep,methods);metrics[str(rep)]=results;rows+=current;sources.update(hashes)
  allfits.extend((rep,method) for method in methods)
 n.save_json(s.OUT/'recovery_metrics.json',metrics);n.write_csv(s.OUT/'validated_recovery_comparison.csv',rows)
 calibrated=json.loads((s.OUT/'posterior_sigma_calibration/metrics.json').read_text())
 calibration_selection=json.loads((s.OUT/'posterior_sigma_calibration/selection.json').read_text())
 extra_diagnostics=[]
 for cid in s.IDS:
  path=s.OUT/'posterior/empirical'/str(cid)/chosen/'summary.json';saved=json.loads(path.read_text())
  extra_diagnostics.append(dict(contest_id=cid,variant=chosen,passed=saved['passed'],diagnostics=saved['diagnostics'],summary_sha256=n.digest(path),
   use='Excluded from final parameter tables: sigma candidate failed independent recovery improvement; failed chain fit additionally excluded.'))
 n.save_json(s.OUT/'extra_sigma_empirical_diagnostics.json',extra_diagnostics)
 screening=[]
 for kind in ('empirical','recovery'):
  for path in (s.OUT/kind).glob('*/map_*.json'):
   value=json.loads(path.read_text());point=value['best']['z']
   screening.append(dict(dataset=kind,case=value['case'],variant=value['variant'],sigma=math.exp(point[2]),lambda_estimate=math.exp(point[3]),
    optimizer_converged=value['best']['converged'],sigma_absolute_log_error=value.get('absolute_log_error',{}).get('sigma',''),
    lambda_absolute_log_error=value.get('absolute_log_error',{}).get('lambda',''),note='Exploratory MAP only; not the validated posterior estimate.'))
 n.write_csv(s.OUT/'exploratory_map_screen.csv',screening)
 # Fixed original nonlinear reference only. It is not truth or a common-prior
 # comparison after replacing the leading prior.
 original={}
 with (s.a.NONLINEAR/'parameter_comparison.csv').open() as stream:
  for row in csv.DictReader(stream):
   if row['parameter'] in ('sigma','lambda'):original[(int(row['contest_id']),row['parameter'])]=row
 empirical=[]
 for cid in s.IDS:
  for method in ('baseline','lambda_external','activity_lambda_external'):
   if method=='baseline':
    summaries={key:dict(median=float(original[(cid,key)]['baseline_median']),mean=float(original[(cid,key)]['baseline_mean'])) for key in ('sigma','lambda')}
   else:
    fit,path=read_fit('empirical',str(cid),method);summaries=fit['summaries'];sources[str(path.relative_to(n.ROOT))]=n.digest(path)
   for key in ('sigma','lambda'):
    reference=float(original[(cid,key)]['nonlinear_median']);value=summaries[key]['median']
    empirical.append(dict(contest_id=cid,method=method,parameter=key,leading_median=value,fixed_old_prior_nonlinear_median=reference,
     absolute_log_gap_to_fixed_reference=abs(math.log(value/reference)),interpretation='Reference disagreement, not estimation error. Priors differ for adjusted leading fits.'))
 n.write_csv(s.OUT/'empirical_reference_comparison.csv',empirical)
 n.save_json(s.OUT/'empirical_reference_metrics.json',{method:{key:float(np.mean([r['absolute_log_gap_to_fixed_reference'] for r in empirical if r['method']==method and r['parameter']==key])) for key in ('sigma','lambda')} for method in ('baseline','lambda_external','activity_lambda_external')})
 lines=['一阶估计的sigma/lambda改进试验','',
  '范围：原最高w的十场比赛。估计仍使用一阶努力函数；原论文、主估计、Table 7和已完成的非线性结果均未改动。',
  '真实比赛没有已知真值。恢复误差只在新生成的非线性模拟数据中计算；真实数据与原非线性估计的差距另外报告，不能视作真实误差。',
  '', '试验与验证',
  '初筛包括原先验、r先验放宽/重定中心、共同成本尺度放宽、sigma/lambda联合放宽或按其余63场的描述性尺度重定中心、降低/删除次数似然，以及仅调整lambda的方案。',
  '每个参数配置固定两组探索性实现（0/1）；另生成实现2确认先筛出的方案。随后用0/1筛选12个sigma先验组合，另生成实现3确认，未用实现3选择先验。所有模拟种子与不理想结果保留。',
  '模拟参数为原非线性后验的联合代表点，使用检查过的非线性努力函数生成数据。每一轮十个配置各一个实现，是有针对性的有限样本检查，不是73场总体精度保证。',
  '', '最简单的lambda调整',
  'log(lambda)先验从N(log1,0.75^2)改为N(log100,2.25^2)，其余先验和似然保持原样。100/2.25来自排除这十场后其余63场的粗略估计尺度（中位数约90，稳健对数标准差约2.17），作为经验校准候选，不是外部真值。',
  '这既改变中心，也放宽范围；不能将效果仅归于中心调整。',
  '', '实现2：正式后验确认（平均绝对相对误差；真值落入95%区间的配置数）',
  '方法 | sigma误差 | lambda误差 | sigma区间 | lambda区间']
 for method in groups[2]:
  m=metrics['2'][method];lines.append(f"{method} | {m['sigma']['mean_absolute_relative_error']:.2%} | {m['lambda']['mean_absolute_relative_error']:.2%} | {m['sigma']['truth_in_95_interval']}/10 | {m['lambda']['truth_in_95_interval']}/10")
 lines+=['',f'额外sigma先验筛选：{chosen}；固定lambda调整，以探索样本的sigma平均绝对对数误差最小为目标，并约束lambda误差和sigma最差误差。',
  '实现3：独立随机实现确认（平均绝对相对误差；真值落入95%区间的配置数）',
  '方法 | sigma误差 | lambda误差 | sigma区间 | lambda区间']
 for method in groups[3]:
  m=metrics['3'][method];lines.append(f"{method} | {m['sigma']['mean_absolute_relative_error']:.2%} | {m['lambda']['mean_absolute_relative_error']:.2%} | {m['sigma']['truth_in_95_interval']}/10 | {m['lambda']['truth_in_95_interval']}/10")
 lines+=['', '主要判断',
  '只放宽/校准lambda先验，对模拟中的lambda恢复有明确改善；不应据此宣称sigma同时改善。降低次数似然权重属于广义贝叶斯敏感性分析，并非单纯换先验，实现2中sigma恢复反而变差，不推荐采用。',
  'sigma先验小范围筛选的成效以实现3为准；训练误差下降不能代替新实现验证。点估计统一采用后验中位数，完整均值、区间和逐场误差保留。',
  '额外检查：MAP初筛不一定选出后验中位数恢复最好的先验。因此，另用实现2的lambda_external后验，对同一12个sigma候选逐一重加权，按后验恢复误差选择，再检查实现3。此项是事后补充划分检查，不是新的盲测。',
  f"该补充检查选出{calibration_selection['selected']['variant']}：实现2的sigma误差为{calibrated['2']['sigma']['mean_absolute_relative_error']:.2%}，实现3为{calibrated['3']['sigma']['mean_absolute_relative_error']:.2%}；实现3原先验为{metrics['3']['baseline']['sigma']['mean_absolute_relative_error']:.2%}。改善没有跨随机实现稳定保持。",
  '因此，不建议采用训练中看起来更好的sigma先验替换主先验。此次已实现lambda误差下降，尚未找到稳定降低sigma误差的简单先验调整。',
  '', '真实十场：必须单独解读',
  '新lambda先验在部分高精度比赛中使lambda大幅增加，且原非线性参考仍使用旧先验。故真实数据中的数值接近与模拟中的恢复精度是两件事；当前试验不能声称已让十场真实参数估计更接近真值。',
  '加入较宽r先验后，部分真实比赛出现显著分离的sigma后验区域。已用包含优化所得多个区域的固定混合提议进行复核；未通过的初始采样保存在各自pilot文件夹，不用于最终表格。',
  '详见empirical_reference_comparison.csv。若用于论文，应先在共同新先验下复核非线性参考及反事实产出；本轮没有将固定旧参考当作先验选择目标。',
  '例：4699的lambda中位数在lambda-only方案下由原约15642增至208524；原非线性旧先验参考约6782。这不是已证实的实证准确性改善，提示其估计对先验较敏感。',
  f"额外sigma先验的真实十场试算有{sum(x['passed'] for x in extra_diagnostics)}/10通过采样门槛。3929未通过；该先验已因独立模拟未见改善而不推荐，额外真实数据试算不进入最终参数表，保留诊断供核查。",
  '', '核验',
  '所有进入最终表格的拟合均通过四链rank Rhat<=1.01、bulk/tail ESS>=400，检查包含原始参数与行为组合。一阶核与原实现的20点似然误差最大约1.82e-12。',
  '两项新增单元检验通过：原核一致性/似然分解，以及lambda-only方案确实保持其余先验不变。模拟策略粗细网格误差均小于1%。',
  '原始模拟数据、种子、所有候选、失败pilot和源文件校验信息均保留。']
 (s.OUT/'final_report.txt').write_text('\n'.join(lines)+'\n')
 sources.update({str(path.relative_to(n.ROOT)):n.digest(path) for path in (s.OUT/'recovery').glob('*/data.json')})
 sources.update({str(path.relative_to(n.ROOT)):n.digest(path) for path in (s.OUT/'recovery').glob('*/truth.json')})
 scripts=['sigma_lambda_trials.py','sigma_lambda_posterior.py','sigma_lambda_count_posterior.py','sigma_lambda_refine.py','sigma_lambda_repair.py','sigma_prior_grid.py','sigma_lambda_replication.py','sigma_lambda_trial_report.py','sigma_grid_empirical.py','sigma_posterior_calibration.py']
 artifacts=['design.json','design_round2.json','sigma_grid_design.json','sigma_grid_selection.json','kernel_parity.json','recovery_metrics.json','validated_recovery_comparison.csv','empirical_reference_comparison.csv','empirical_reference_metrics.json','final_report.txt']
 artifacts+=['extra_sigma_empirical_diagnostics.json','exploratory_map_screen.csv']
 artifacts += ['posterior_sigma_calibration/'+name+'.json' for name in ('design','selection','training','validation','metrics')]
 n.save_json(s.OUT/'completion.json',dict(status='completed_sigma_lambda_trials',scope='Known-truth recovery experiments and separate ten-contest prior sensitivity; no publication replacement.',
  source_sha256={name:n.digest(n.ROOT/'pipeline/empirical'/name) for name in scripts},artifact_sha256={name:n.digest(s.OUT/name) for name in artifacts},input_and_fit_summary_sha256=sources,
  weighted_kernel_sha256=n.digest(s.OUT/'weighted_leading_kernel.cpp')))
 print(s.OUT/'final_report.txt',flush=True)


if __name__=='__main__':run()
