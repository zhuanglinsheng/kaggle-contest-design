# Task 4：D主切分残余积分误差加密

全73例评分与22个新稀有小时积分后，D4104/4488仍有残余平均分数MCSE>.001。仅这两例按原方案延长同目标链，所有已核验尾部积分继续适用；保留原链和未达精度结果。平均分数与半样本、预算间一致性和参数诊断门槛不变。任务表在本批运行前固定。

```python
import ast,types,json,math,hashlib,os,time,subprocess
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nonlinear
p=Path('docs/review/PaperJK9');status=p/'task4_D_residual_precision_status_20261025.json';assert not status.exists(),'Inspect existing precision driver before resume';mp=Path('pipeline/empirical/unified_prior_medians.py');mt=mp.read_text();fn=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');ns={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(mp)};exec(compile(ast.get_source_segment(mt,fn),str(mp),'exec'),ns);med=types.SimpleNamespace(diagnose=ns['diagnose'],__file__=str(mp))
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
training=code(p/'task4_final_protocol_training_method_20261025.md');dfn=next(n for n in ast.parse(training).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');dfunc=ast.get_source_segment(training,dfn);score_template=code(p/'task4_calibrated_posterior_prediction_method_20261025.md');assert 'precision_index' in score_template
index=p/'task4_predictive_chain_index_20261025.json';tasks=[('D',4104,8000),('D',4488,8000)]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,x):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(f)
state=dict(status='running',driver_pid=os.getpid(),scope='Same-target residual predictive integration refinement after all flagged rare hours were integrated; no model or prior selection',tasks=[dict(model=m,contest_id=c,first_samples=n) for m,c,n in tasks],finished=[],failed=[],current=None,all_requested_numerically_checked=False,mean_score_mcse_target=.001,budget_consistency_rule='Difference <= 5*sqrt(previous_MCSE^2+new_MCSE^2)+1e-4, plus own mean-score/half-sample screen',production_files_overwritten=False);save(status,state);clock=time.monotonic()
for kind,cid,first_samples in tasks:
 entry=next(r for r in json.loads((p/'task4_primary_training_index_20261025.json').read_text())['models'][kind]['entries'] if r['contest_id']==cid);base=json.loads(Path(entry['accepted']).read_text());assert base['diagnostics']['passed'];initial=next(r for r in json.loads((p/f'task4_{kind}_calibrated_forecast_20261025.json').read_text())['records'] if r['contest_id']==cid)
 if kind=='C' and cid==2454:
  initial=next(r for r in json.loads((p/'task4_C_cached_precision_forecast_20261025.json').read_text())['records'] if r['contest_id']==cid)
 previous_score=initial;rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));data=e.stan_data(json.loads(rawpath.read_text()),'early');inputs=dict(data,**u.prior(data));model=(p/('reconstruction_20261025/model_snapshot/detail_model' if kind=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();assert digest(model.with_suffix('.stan'))==base['configuration']['model_sha256'];centerchains=[e.read_columns(Path(f)) for f in base['chain_paths']];accepted=False
 for samples in [n for n in [8000,32000,64000] if n>=first_samples]:
  folder=p/'prediction_precision_20261025'/kind/f'contest_{cid}'/f'draws_{samples}';assert not folder.exists(),'Preserve existing precision-stage files';folder.mkdir(parents=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];jobs=[];warmup=2000 if samples==8000 else 4000;delta=.995 if samples<64000 else .9975
  for j,f in enumerate(paths,1):
   names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if kind=='C' else e.PARAMETERS;init={k:float(np.median(centerchains[j-1][k])) for k in names};ip=folder/f'init_{j}.json';save(ip,init);log=(folder/f'chain_{j}.log').open('w');cmd=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261025+900000+cid+(100000 if kind=="C" else 0)+samples}',f'id={j}','data',f'file={folder/"data.json"}','output',f'file={f}','refresh=500','sig_figs=18',f'init={ip}'];job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((job,log,cmd))
  state['current']=dict(model=kind,contest_id=cid,samples=samples,pids=[j[0].pid for j in jobs],commands=[j[2] for j in jobs]);save(status,state);print('PREDICTIVE PRECISION',kind,cid,samples,flush=True)
  while any(j[0].poll() is None for j in jobs):
   state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[j[0].poll() for j in jobs];save(status,state);time.sleep(2)
  codes=[j[0].returncode for j in jobs]
  for _,log,_ in jobs:log.close()
  if any(c!=0 for c in codes):state['failed'].append(dict(model=kind,contest_id=cid,samples=samples,reason='Nonzero sampler exit',codes=codes));save(status,state);break
  assert all(e.csv_complete(f,samples) for f in paths);env=dict(e=e,u=u,np=np,math=math,nonlinear=nonlinear,med=med,MODEL_KIND=kind);exec(compile(dfunc,'final-protocol-diagnose','exec'),env);row,d=env['diagnose'](cid,paths,data);cfg=dict(base['configuration'],predictive_refinement=True,samples_per_chain=samples);a=dict(summary=row,diagnostics=d,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],configuration=cfg,attempt=f'predictive_draws_{samples}',scope='Same-target longer chains to improve posterior count-probability integration');save(folder/'diagnostics.json',a);centerchains=[e.read_columns(f) for f in paths]
  if not d['passed']:print('PREDICTIVE PARAMETER CHECK FAILED',kind,cid,samples,flush=True);continue
  ap=folder/'accepted_parameters.json';save(ap,a);ix=json.loads(index.read_text());key=kind+':'+str(cid);ix['entries'][key]=dict(accepted=str(ap.resolve()),accepted_sha256=digest(ap),draws_per_chain=samples,prediction_precision_not_yet_established=True);save(index,ix)
  execution=score_template.replace("for entry in state['finished']:",f"for entry in [r for r in state['finished'] if r['contest_id']=={cid}]:");execution=execution.replace("(p/f'task4_{MODEL_KIND}_calibrated_forecast_20261025.json')",repr(folder/'prediction.json')).replace("(p/f'task4_{MODEL_KIND}_calibrated_forecast_hours_20261025.csv')",repr(folder/'prediction_hours.csv'))
  # pathlib repr needs a Path constructor in the isolated execution namespace.
  execution=execution.replace('PosixPath(', 'Path(');(folder/'score_execution.txt').write_text(execution);exec(compile(execution,str(folder/'score_execution.txt'),'exec'),{'MODEL_KIND':kind});new=json.loads((folder/'prediction.json').read_text())['records'][0];delta_score=abs(new['mean_joint_logscore']-previous_score['mean_joint_logscore']);limit=5*math.hypot(new['mean_logscore_batch_influence_mcse'],previous_score['mean_logscore_batch_influence_mcse'])+1e-4;consistent=delta_score<=limit;check=dict(previous_samples_per_chain=previous_score['draws_per_chain'],new_samples_per_chain=samples,previous_mean=previous_score['mean_joint_logscore'],new_mean=new['mean_joint_logscore'],mean_change=delta_score,combined_mcse_consistency_limit=limit,consistent=consistent,mean_score_screen_passed=new['mean_score_numerical_screen_passed'],new_mcse=new['mean_logscore_batch_influence_mcse']);save(folder/'precision_check.json',check);print('PREDICTIVE SCORE CHECK',kind,cid,samples,new['mean_logscore_batch_influence_mcse'],consistent,flush=True)
  if new['mean_score_numerical_screen_passed'] and consistent:
   ix=json.loads(index.read_text());ix['entries'][key].update(prediction_precision_not_yet_established=False,score_mcse=new['mean_logscore_batch_influence_mcse'],precision_check=str((folder/'precision_check.json').resolve()));save(index,ix);state['finished'].append(dict(model=kind,contest_id=cid,samples_per_chain=samples,accepted=str(ap.resolve()),prediction=str((folder/'prediction.json').resolve()),check=str((folder/'precision_check.json').resolve())));save(status,state);accepted=True;break
  previous_score=new
 if not accepted and not any(r['model']==kind and r['contest_id']==cid for r in state['failed']):state['failed'].append(dict(model=kind,contest_id=cid,reason='Predictive precision schedule exhausted; retain explicit integration uncertainty'))
 state['current']=None;save(status,state)
state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';state['all_requested_numerically_checked']=len(state['finished'])==len(tasks);save(status,state);print('PREDICTIVE PRECISION ENDED',len(state['finished']),len(state['failed']),flush=True)

```
