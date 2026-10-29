# Task 4：尾部积分重要区域的目标核对

除训练抽样点外，再在两个罕见小时的分子模式、最大分子权重点和最大分母权重点，与原Stan训练密度及独立Python预测函数逐点核对。两小时虽然使用不同数值提议，其训练分母必须相同，另以估计MCSE核对该不变量。通过后才写入尾部概率索引；该索引仅修正联合分数的数值积分，不改变模型、训练参数后验或边际校准概率。

```python
import json,csv,math,hashlib,subprocess
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9')
def code(path):return Path(path).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
backend=code(p/'task4_predictive_density_v2_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0];exec(compile(backend,'verified-density-v2','exec'));py=code(p/'task4_calibrated_posterior_prediction_method_20261025.md');import ast
from scipy.special import gammaln,gammaincc,logsumexp
env=dict(e=e,np=np,math=math,gammaln=gammaln,gammaincc=gammaincc,logsumexp=logsumexp)
for name in ['counts_for','predictions']:
 fn=next(n for n in ast.parse(py).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(py,fn),'reported-python-predictor','exec'),env)
raw=json.loads(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob('contest_3288_*.json')).read_text());data=e.stan_data(raw,'early');cut=data['N_Delta'];data.update(u.prior(data));folder=p/'predictive_integration_20261025/rare_mode_parity';assert not folder.exists();folder.mkdir();dp=folder/'data.json';dp.write_text(json.dumps(data));reference=json.loads((p/'predictive_integration_20261025/parity_v2.json').read_text());constant=next(r['constant_density_difference'] for r in reference['records'] if r['model']=='D' and r['contest_id']==3288);records=[];integrals=[];exe=(p/'reconstruction_20261025/model_snapshot/detail_model').resolve()
for hour in [711,760]:
 root=p/'predictive_integration_20261025'/f'D_3288_hour_{hour}';s=json.loads((root/'status.json').read_text());assert s['passed'];last=s['records'][-1];integrals.append(last);prod=np.load(root/f"production_{last['samples']}.npz");prop=json.loads((root/'proposal.json').read_text());mode=min(prop['modes'],key=lambda m:m['negative_log_numerator'])['point'];choices=[np.array(mode),prod['z'][np.argmax(prod['log_training']+prod['log_predictive']-prod['log_proposal'])],prod['z'][np.argmax(prod['log_training']-prod['log_proposal'])]];target=ReviewTarget(raw,cut,hour-1,'D')
 for j,z in enumerate(choices):
  params=dict(zip(e.PARAMETERS,np.r_[np.exp(z[:4]),z[4:]].tolist()));pp=folder/f'params_{hour}_{j}.json';pp.write_text(json.dumps(params));out=folder/f'lp_{hour}_{j}.csv';run=subprocess.run([str(exe),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'],capture_output=True,text=True);(folder/f'log_{hour}_{j}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;stan=float(next(csv.DictReader(line for line in out.open() if not line.startswith('#')))['lp__'])+float(z[:4].sum());lp,pr=target.posterior_and_prediction(z);training_error=float(lp[0]-stan-constant);python_prediction=env['predictions'](raw,{k:np.array([v]) for k,v in params.items()},hour-1,hour,free=False)[0][0,0];prediction_error=float(pr[0]-python_prediction);assert abs(training_error)<1e-7 and abs(prediction_error)<1e-7;records.append(dict(hour=hour,point_type=['numerator_mode','largest_numerator_weight','largest_denominator_weight'][j],training_log_density_error_after_constant=training_error,prediction_log_density_error=prediction_error))
se0=[math.sqrt(r['samples']/(r['samples']-1)*(1/r['denominator_ess']-1/r['samples'])) for r in integrals];delta=abs(integrals[0]['log_training_normalizer']-integrals[1]['log_training_normalizer']);limit=5*math.hypot(*se0)+.001;assert delta<=limit
result=dict(status='completed',passed=True,records=records,shared_training_normalizer_difference=delta,shared_normalizer_consistency_limit=limit,scope='Rare numerator modes and largest production weights reproduce original target and forecast function; both hour integrals share the same training normalizer within Monte Carlo precision');out=p/'predictive_integration_20261025/rare_mode_parity.json';out.write_text(json.dumps(result,indent=2));index=p/'task4_tail_probability_index_20261025.json';assert not index.exists();entries={}
for hour,r in zip([711,760],integrals):
 root=p/'predictive_integration_20261025'/f'D_3288_hour_{hour}';entries[f'D:3288:primary_hour:{hour}']=dict(model='D',contest_id=3288,split='primary_hour',hour_index=hour,joint_logscore=r['joint_logscore'],logscore_mcse=r['logscore_mcse'],integral_status=str((root/'status.json').resolve()),integral_status_sha256=hashlib.sha256((root/'status.json').read_bytes()).hexdigest(),rare_mode_parity=str(out.resolve()),rare_mode_parity_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob('contest_3288_*.json')).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(exe.with_suffix('.stan').read_bytes()).hexdigest(),training_cutoff=cut)
index.write_text(json.dumps(dict(scope='Validated same-target importance correction of joint count probabilities only; no parameter/posterior/model update',entries=entries),indent=2));print(json.dumps(result,indent=2))

```
