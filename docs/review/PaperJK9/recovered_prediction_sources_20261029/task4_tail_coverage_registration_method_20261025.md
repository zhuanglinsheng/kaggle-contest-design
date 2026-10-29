# Task 4：覆盖修复后的原目标核验与登记

只替换已重新标记为待核验的条目。新旧目标、输入、窗口、模型和先验必须相同；新提议三个独立预算通过及原目标/重要区域核验后，旧记录进入history，当前条目重新赋值而不继承挑战标记。旧新数值均可从各自状态追溯；不要求低估的旧MCSE作为新误差上限。

```python
import ast,csv,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.special import gammaln,gammaincc,logsumexp
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9')
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
exec(compile(code(p/'task4_predictive_density_v2_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0],'density-v2','exec'))
KIND=globals().get('KIND','D');CID=globals().get('CID',3288);HOUR=globals().get('HOUR',760);SPLIT=globals().get('SPLIT','primary_hour');root=p/'predictive_integration_20261025'/(f'{KIND}_{CID}_hour_{HOUR}' if SPLIT=='primary_hour' else f'{KIND}_{CID}_complete_day_hour_{HOUR}');root=Path(str(root)+globals().get('RUN_LABEL',''));st=json.loads((root/'status.json').read_text());assert st['passed'];last=st['records'][-1];ap=Path(st['training_accepted']);assert digest(ap)==st['training_accepted_sha256'];a=json.loads(ap.read_text());rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{CID}_*.json'));assert digest(rawpath)==st['input_sha256'];raw=json.loads(rawpath.read_text());cut=st['training_cutoff'];target=ReviewTarget(raw,cut,HOUR-1,KIND)
if SPLIT=='primary_hour':data=e.stan_data(raw,'early');data.update(u.prior(data))
else:data=json.loads((p/f'prediction_reconstruction_20261025/day_inputs/contest_{CID}.json').read_text())
assert data['N_Delta']==cut and u.prior(data)==st['training_prior'];exe=(p/('reconstruction_20261025/model_snapshot/detail_model' if KIND=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();assert digest(exe.with_suffix('.stan'))==st['model_sha256'];check=root/'original_target_parity';assert not check.exists();check.mkdir();dp=check/'data.json';dp.write_text(json.dumps(data));prod=np.load(root/f"production_{last['samples']}.npz");prop=json.loads((root/'proposal.json').read_text());mode=min(prop['modes'],key=lambda m:m['negative_log_numerator'])['point'];names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if KIND=='C' else e.PARAMETERS;c=e.read_columns(Path(a['chain_paths'][0]));reference=np.array([c[k][0] for k in names]);reference[:4]=np.log(reference[:4]);points=[reference,np.array(mode),prod['z'][np.argmax(prod['log_training']+prod['log_predictive']-prod['log_proposal'])],prod['z'][np.argmax(prod['log_training']-prod['log_proposal'])]]
txt=code(p/'task4_calibrated_posterior_prediction_method_20261025.md');tree=ast.parse(txt);env=dict(e=e,np=np,math=math,gammaln=gammaln,gammaincc=gammaincc,logsumexp=logsumexp)
for name in ['counts_for','predictions']:
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.get_source_segment(txt,fn),'independent-python','exec'),env)
records=[];constant=None
for j,z in enumerate(points):
 params=dict(zip(names,np.r_[np.exp(z[:4]),z[4:]].tolist()));pp=check/f'params_{j}.json';pp.write_text(json.dumps(params));out=check/f'lp_{j}.csv';run=subprocess.run([str(exe),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'],capture_output=True,text=True);(check/f'log_{j}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;stan=float(next(csv.DictReader(line for line in out.open() if not line.startswith('#')))['lp__'])+float(z[:4].sum());lp,pr=target.posterior_and_prediction(z)
 if constant is None:constant=float(lp[0]-stan)
 training_error=float(lp[0]-stan-constant);prediction=float(env['predictions'](raw,{k:np.array([v]) for k,v in params.items()},HOUR-1,HOUR,free=(KIND=='C'))[0][0,0]);error=float(pr[0]-prediction);assert abs(training_error)<1e-7 and abs(error)<1e-7;records.append(dict(point_type=['training_reference','proposal_reference_mode','largest_numerator_weight','largest_denominator_weight'][j],training_log_density_error_after_constant=training_error,prediction_log_density_error=error))
result=dict(passed=True,model=KIND,contest_id=CID,split=SPLIT,hour=HOUR,constant_density_difference=constant,records=records,training_accepted_sha256=digest(ap),source_integration_status_sha256=digest(root/'status.json'));pf=root/'original_target_parity.json';pf.write_text(json.dumps(result,indent=2))
with (p/'task4_tail_probability_index_20261025.json.lock').open('a') as registry_lock:
 import fcntl
 fcntl.flock(registry_lock,fcntl.LOCK_EX)
 index=p/'task4_tail_probability_index_20261025.json';ix=json.loads(index.read_text());key=f'{KIND}:{CID}:{SPLIT}:{HOUR}';old=ix['entries'][key];assert old['model_sha256']==st['model_sha256'] and old['input_sha256']==digest(rawpath) and old['training_cutoff']==cut and old['training_prior']==st['training_prior'];assert old.get('precision_challenged',False);assert len(st['records'])>=3 and all(r['numerical_screen_passed'] and r['consistent_with_previous'] for r in st['records'][-3:]);ix.setdefault('history',{}).setdefault(key,[]).append(old);ix['entries'][key]=dict(model=KIND,contest_id=CID,split=SPLIT,hour_index=HOUR,joint_logscore=last['joint_logscore'],logscore_mcse=last['logscore_mcse'],integral_status=str((root/'status.json').resolve()),integral_status_sha256=digest(root/'status.json'),rare_mode_parity=str(pf.resolve()),rare_mode_parity_sha256=digest(pf),input_sha256=digest(rawpath),model_sha256=st['model_sha256'],training_cutoff=cut,training_prior=st['training_prior']);tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index);print('COVERAGE REPAIR REGISTERED',key,last['joint_logscore'],flush=True)

```
