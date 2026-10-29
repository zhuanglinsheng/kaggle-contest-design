# Task 3：代表性设计的完全披露非线性核算批次

日期：2026-10-25。6 个重建案例均已通过最终原混合门槛及本次额外均值精度检查。每例从全部联合抽样中，选择在 log 正参数、原 mu0 与 log_r 坐标下、按 IQR 标准化后距离边际中位数向量最近的实际抽样。保留抽样索引和来源哈希，避免拼接边际中位数当作联合代表点。

每例评价：原奖金期限下的完全披露、所选奖金期限下的完全披露、原一阶设计表中重优化的完全披露比较设计。原选择无反馈的案例，其同奖金期限完全披露仅为比较，不能替代实际所选精度。零奖金另用解析无努力基准。

非线性完全披露使用报告中解析策略和线性边值输出方程，容差、域、初始先验求积一起加密；一阶输出在同一个代表点、同一个固定初始分布下计算。此批次还不等于原有限精度设计的非线性输出，也不能认证后验平均保持质量。

```python
import csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from pipeline.empirical import estimate as estimator
from pipeline.empirical.fixed_initial_design import Output,beta
report=Path('docs/review/PaperJK9');rebuild=report/'reconstruction_20261025'
method=(report/'task3_full_disclosure_asymmetric_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0]
exec(compile(method,'<reported full asymmetry method>','exec'))
selection=list(csv.DictReader((report/'task3_representative_input_inventory_20261025.csv').open()))
comparison={(int(r['contest_id']),float(r['delta'])):r for r in csv.DictReader(Path('results/reestimation/lambda100_sd225/fixed_initial_design/full_disclosure_comparison.csv').open())}
out=report/'task3_full_disclosure_cases_20261025.json'
if out.exists():raise RuntimeError('Existing output: inspect rather than overwrite')
state=dict(date='2026-10-25',status='running',scope='Nonlinear full-disclosure evaluation at actual joint-posterior representative draws; not posterior-average quality certification',method_sha256=hashlib.sha256(method.encode()).hexdigest(),method_snapshot=method,records=[],nonlinear_quality_preservation_verified=False)
points=[]
def save():
 temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(out)
save()
for selected in selection:
 cid=int(selected['contest_id']);accepted=rebuild/'estimation/full'/f'contest_{cid}/accepted.json';a=json.loads(accepted.read_text());assert a['diagnostics']['passed']
 for f,h in zip(a['chain_paths'],a['chain_sha256']):assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 columns=[estimator.read_columns(Path(f)) for f in a['chain_paths']]
 arrays={k:np.concatenate([c[k] for c in columns]) for k in estimator.PARAMETERS}
 z=np.column_stack([np.log(arrays[k]) if k not in ('mu_0','log_r') else arrays[k] for k in estimator.PARAMETERS])
 center=np.median(z,axis=0);spread=np.maximum(np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0),1e-8)
 index=int(np.argmin(np.sum(((z-center)/spread)**2,axis=1)))
 point={k:float(arrays[k][index]) for k in estimator.PARAMETERS}
 data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());theta0=data['theta'];T0=data['T_contest'];sigma=point['sigma'];S0=sigma/math.sqrt(point['lambda']);mu=point['mu_0']
 point_record=dict(contest_id=cid,selection_reason=selected['selection_reason'],joint_draw_index=index,chain_index=index//len(columns[0]['c_i'])+1,within_chain_index=index%len(columns[0]['c_i']),accepted_sha256=hashlib.sha256(accepted.read_bytes()).hexdigest(),chain_sha256=a['chain_sha256'],point=point,observed_prize=theta0,observed_duration=T0,fixed_initial_variance=S0)
 points.append(point_record)
 benchmark=comparison[(cid,float(selected['delta']))];full_multiple=float(benchmark['full_time_multiplier']);full_ratio=max(0.,float(benchmark['full_cost'])-float(selected['delta'])*full_multiple**2)
 designs=[('full_at_observed_prize_duration',theta0,T0),('full_at_selected_prize_duration',theta0*float(selected['prize_ratio']),T0*float(selected['time_multiplier'])),('reoptimized_first_order_full_benchmark',theta0*full_ratio,T0*full_multiple)]
 for label,theta,T in designs:
  r=dict(contest_id=cid,label=label,theta=theta,T=T,S0=S0,mu=mu,ci=point['c_i'],cj=point['c_j'],sigma=sigma,trials=[],selected_original_kind=selected['choice'],represents_selected_precision=selected['choice'] in ('full_disclosure','zero_prize') and label=='full_at_selected_prize_duration')
  draws={k:np.array([point[k]]) for k in ('c_i','c_j','sigma','lambda','mu_0')};linear=Output(draws,T,96);r['first_order_full_output_at_same_point']=linear.no_effort+theta*linear.value(math.inf)
  if theta==0:
   r['analytic_zero_prize_output']=float(beta(mu,S0+sigma*sigma*T))
  else:
   for tol,quad,base_extent in [(1e-8,80,12.),(1e-10,160,14.)]:
    hermite,_=roots_hermitenorm(quad);extent=max(base_extent,float(np.max(np.abs((mu+math.sqrt(S0)*hermite)/(sigma*math.sqrt(T)))))+1.)
    try:r['trials'].append(full_asymmetric_output(theta,T,sigma,point['c_i'],point['c_j'],S0,mu,extent=extent,tolerance=tol,quadrature=quad))
    except Exception as exc:r['trials'].append(dict(error=str(exc),extent=extent,tolerance=tol,quadrature=quad))
  if len(r['trials'])==2 and all('B' in t for t in r['trials']):
   r['refinement_output_difference']=abs(r['trials'][1]['B']-r['trials'][0]['B'])
   r['nonlinear_minus_first_order']=r['trials'][1]['B']-r['first_order_full_output_at_same_point']
  state['records'].append(r);save();print(cid,label,'done',flush=True)
state['status']='completed';state['representative_points']=points;save()
(report/'task3_representative_joint_points_20261025.json').write_text(json.dumps(dict(scope='Actual joint draws closest to marginal median vector after IQR scaling; not coordinatewise-median parameter combinations',points=points),indent=2,allow_nan=False))
print('Full-disclosure point evaluations completed.',flush=True)
```
