# Task 4：独立整段概率积分的原目标核验与登记

生产积分通过后，读取最高训练重要性权重点及原链参考点；在原Stan训练目标上比较密度（允许参数无关常数），并与独立Python函数比较这些点的完整预测路径。输入/先验/窗口/模型、提议、生产数据和接受链哈希逐项核对。通过后才登记用于统一评分；旧参数链和边际校准输入不覆盖。

```python
import ast,csv,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.special import gammaln,gammaincc,logsumexp
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9')
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
exec(compile(code(p/'task4_full_path_parity_method_20261025.md').split('# Verify complete held-out paths')[0],'full-path-target','exec'))
stage=Path(globals()['ESTIMATE_STAGE']);st=json.loads((stage/'status.json').read_text());assert st['passed'];prop=json.loads((stage/'proposal.json').read_text());assert digest(stage/'proposal.json')==st['proposal_sha256'];ap=Path(prop['training_accepted']);assert digest(ap)==st['training_accepted_sha256'];a=json.loads(ap.read_text());kind=st['model'];cid=st['contest_id'];split=st['split'];cut=st['training_cutoff'];rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));assert digest(rawpath)==st['input_sha256'];raw=json.loads(rawpath.read_text());target=FullPathTarget(raw,cut,kind);last=st['records'][-1];production=stage/f"production_{last['samples']}.npz";prod=np.load(production);indices=np.argsort(prod['log_training_importance'])[-3:];points=prod['z'][indices];names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if kind=='C' else e.PARAMETERS;c=e.read_columns(Path(a['chain_paths'][0]));reference=np.array([c[k][0] for k in names]);reference[:4]=np.log(reference[:4]);points=np.vstack([reference,points]);folder=stage/'original_target_parity';assert not folder.exists();folder.mkdir()
if split=='primary_hour':data=e.stan_data(raw,'early');data.update(u.prior(data))
else:data=json.loads((p/f'prediction_reconstruction_20261025/day_inputs/contest_{cid}.json').read_text())
assert data['N_Delta']==cut and u.prior(data)==st['training_prior'];exe=(p/('reconstruction_20261025/model_snapshot/detail_model' if kind=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();assert digest(exe.with_suffix('.stan'))==st['model_sha256'];dp=folder/'data.json';dp.write_text(json.dumps(data));txt=code(p/'task4_calibrated_posterior_prediction_method_20261025.md');tree=ast.parse(txt);env=dict(e=e,np=np,math=math,gammaln=gammaln,gammaincc=gammaincc,logsumexp=logsumexp)
for name in ['counts_for','predictions']:
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(txt,fn),'independent-python-prediction','exec'),env)
lp,path=target.posterior_and_path(points);columns={k:(np.exp(points[:,j]) if j<4 else points[:,j]) for j,k in enumerate(names)};independent=env['predictions'](raw,columns,cut,free=(kind=='C'))[0];patherror=float(np.max(abs(path-independent)));assert patherror<1e-7;constant=None;records=[]
for j,z in enumerate(points):
 params={k:float(columns[k][j]) for k in names};pp=folder/f'params_{j}.json';pp.write_text(json.dumps(params));out=folder/f'lp_{j}.csv';run=subprocess.run([str(exe),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'],capture_output=True,text=True);(folder/f'log_{j}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;stan=float(next(csv.DictReader(line for line in out.open() if not line.startswith('#')))['lp__'])+float(z[:4].sum())
 if constant is None:constant=float(lp[j]-stan)
 err=float(lp[j]-stan-constant);assert abs(err)<1e-7;records.append(dict(point_type='training_reference' if j==0 else 'largest_training_importance_weight',density_error_after_constant=err))
parity=dict(passed=True,records=records,maximum_complete_path_error=patherror,training_density_constant=constant,production_sha256=digest(production));pf=stage/'original_target_parity.json';pf.write_text(json.dumps(parity,indent=2));index=p/'task4_global_path_probability_index_20261025.json';ix=json.loads(index.read_text()) if index.exists() else dict(scope='Validated independent joint count probability integration under unchanged training targets; marginal calibration retains accepted chain estimates',entries={});key=f'{kind}:{cid}:{split}';assert key not in ix['entries'];ix['entries'][key]=dict(model=kind,contest_id=cid,split=split,training_cutoff=cut,input_sha256=st['input_sha256'],model_sha256=st['model_sha256'],training_prior=st['training_prior'],estimate_status=str((stage/'status.json').resolve()),estimate_status_sha256=digest(stage/'status.json'),production=str(production.resolve()),production_sha256=digest(production),original_target_parity=str(pf.resolve()),original_target_parity_sha256=digest(pf));tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index);print('GLOBAL PATH REGISTERED',key,last['mean_joint_logscore'],last['mean_logscore_mcse'],flush=True)

```
