# Task 4：接续D模型余下19个主切分训练

前53例已接受；8219通过同目标独立MH后验核验（八点目标一致，四链连续量最小ESS约27505，最大Rhat约1.00020），其旧NUTS驱动和未完成子进程已停止并保留数据。补充接受索引单列该早期后验，不使用full或测试后验。

此批只接续数值ID大于8219的原19例，原目标、先验、初值、种子和诊断不变；不重复已完成54例。新状态与旧53例、MH补充1例需合并计数，单一批次状态不代表全部73。

[实时状态](task4_D_primary_resume_status_20261025.json)。

```python
import csv,json,hashlib,math,os,subprocess,time,re
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nonlinear
import ast,types
median_path=Path('pipeline/empirical/unified_prior_medians.py');median_text=median_path.read_text();median_tree=ast.parse(median_text);median_function=next(node for node in median_tree.body if isinstance(node,ast.FunctionDef) and node.name=='diagnose');median_namespace={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(median_path)};exec(compile(ast.get_source_segment(median_text,median_function),str(median_path),'exec'),median_namespace);med=types.SimpleNamespace(diagnose=median_namespace['diagnose'],__file__=str(median_path))
MODEL_KIND=globals().get('MODEL_KIND','D');assert MODEL_KIND in ('C','D')
p=Path('docs/review/PaperJK9');base=p/'prediction_reconstruction_20261025'/MODEL_KIND/'early';model=(p/('reconstruction_20261025/model_snapshot/detail_model' if MODEL_KIND=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();status=p/f'task4_{MODEL_KIND}_primary_resume_status_20261025.json'
assert not status.exists(),'Inspect existing median-protocol driver before any resume'
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,x):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(f)
def diagnose(cid,paths,data):
 row,d=med.diagnose(cid,'early',paths,data)
 if MODEL_KIND=='C':
  cc=[e.read_columns(f) for f in paths];arrays={k:np.asarray([c[k] for c in cc]) for k in ('deadline_power','gap_log_scale')};arrays['gap_scale']=np.exp(arrays['gap_log_scale'])
  for k,v in arrays.items():
   bulk,tail=nonlinear.rank_ess(v);ess=nonlinear.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));q=dict(rank_rhat=float(nonlinear.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);d['quantities'][k]=q;row[k+'_median']=float(np.median(v));row[k+'_mean']=mean;row[k+'_mean_mcse']=mcse;d['passed']=bool(d['passed'] and np.isfinite(v).all() and q['rank_rhat']<=1.01 and min(bulk,tail)>=400)
 return row,d
def completed(f):
 if not f.exists():return False
 with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4096));tail=h.read()
 return b'#  Elapsed Time:' in tail
source_root=Path('results/reestimation/lambda100_sd225');summary={int(r['contest_id']):r for r in csv.DictReader((source_root/'estimation/posterior_summary_final.csv').open()) if r['mode']=='early'};budgets={r['contest_id']:r['draws_per_chain'] for r in json.loads((source_root/'verified_fits.json').read_text())['checks'] if r['scenario']=='main' and r['mode']=='early'}
assert MODEL_KIND=='D'
summary={cid:r for cid,r in summary.items() if cid>8219};assert len(summary)==19
state=dict(status='running',driver_pid=os.getpid(),model=MODEL_KIND,protocol='Final frozen primary-median protocol, not earlier raw-mean gate',median_protocol_sha256=digest(Path(med.__file__)),raw_means_retained_with_mcse=True,cases=sorted(summary),finished=[],failed=[],current=None,all_remaining_19_accepted=False,production_results_overwritten=False);save(status,state);clock=time.monotonic()
for cid in sorted(summary):
 case=base/f'contest_{cid}';rawpath=next((source_root/'input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());data=e.stan_data(raw,'early');inputs=dict(data,**u.prior(data));old=summary[cid];center={k:float(old[k+'_median']) for k in ('c_i','c_j','sigma','lambda','mu_0','r')};cfg=dict(contest_id=cid,mode='early',model=MODEL_KIND,input_sha256=digest(rawpath),model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),median_protocol_sha256=digest(Path(med.__file__)),training_cutoff=data['N_Delta'],full_constructed_deadline=data['T_contest'],prior=u.prior(data),extra_priors=dict(deadline_power='Normal(.5,.5)',gap_log_scale='Normal(0,.5)') if MODEL_KIND=='C' else {},initialization='Dispersed saved early medians and preset C parameters; no full or holdout posterior use')
 save(case/'configuration_median_protocol.json',cfg);accepted=False;max_completed=0
 def process(paths,label,reused):
  row,d=diagnose(cid,paths,data);result=dict(summary=row,diagnostics=d,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],configuration=cfg,attempt=label,reused_existing_complete_chains=reused);save(paths[0].parent/'diagnostics_median_protocol.json',result)
  print(MODEL_KIND,'MEDIAN CHECK',cid,label,'passed',d['passed'],flush=True)
  if d['passed']:
   target=case/'accepted_median_protocol.json';save(target,result);state['finished'].append(dict(contest_id=cid,accepted=str(target.resolve()),attempt=label));save(status,state)
  return d['passed']
 for folder in sorted(case.glob('attempt*'),key=lambda f:int(re.search(r'(\d+)$',f.name)[1])):
  paths=[folder/f'chain_{j}.csv' for j in range(1,5)]
  if not all(completed(f) for f in paths):continue
  assert json.loads((folder/'data.json').read_text())==inputs
  with paths[0].open() as h:
   comments=''.join(next(h) for _ in range(20))
  n=int(re.search(r'#\s+num_samples\s*=\s*(\d+)',comments)[1]);assert all(e.csv_complete(f,n) for f in paths);max_completed=max(max_completed,n)
  if process(paths,folder.name,True):accepted=True;break
 if accepted:continue
 schedule=[(1,1000,2000,.98),(2,2000,4000,.99),(3,4000,8000,.995),(4,2000,32000,.995),(5,4000,64000,.9975)];schedule=[s for s in schedule if s[2]>=max(budgets[cid],max_completed)];previous=None
 for attempt,warmup,samples,delta in schedule:
  folder=case/f'median_attempt{attempt}';assert not folder.exists(),'Do not overwrite interrupted chains';folder.mkdir(parents=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];jobs=[]
  for j,f in enumerate(paths,1):
   if previous is None:
    factor=(.8,1.,1.2,1.5)[j-1];init={k:max(.0101,center[k]*factor) for k in ('c_i','c_j','sigma','lambda')};init['mu_0']=center['mu_0']+(j-2.5)*.05;init['log_r']=max(math.log(.01)+.001,math.log(center['r'])+(j-2.5)*.1)
    if MODEL_KIND=='C':init.update(deadline_power=(.2,.4,.6,.8)[j-1],gap_log_scale=(-.2,-.05,.05,.2)[j-1])
   else:
    cc=e.read_columns(previous[j-1]);names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if MODEL_KIND=='C' else e.PARAMETERS;init={k:float(np.median(cc[k])) for k in names}
   ip=folder/f'init_{j}.json';save(ip,init);log=(folder/f'chain_{j}.log').open('w');command=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261026+(300000 if MODEL_KIND=="D" else 400000)+cid}',f'id={j}','data',f'file={folder/"data.json"}','output',f'file={f}','refresh=500','sig_figs=18',f'init={ip}'];job=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((job,log,command))
  state['current']=dict(contest_id=cid,attempt=folder.name,pids=[j[0].pid for j in jobs],commands=[j[2] for j in jobs],process_return_codes=[None]*4);save(status,state);print(MODEL_KIND,'MEDIAN TRAIN',cid,folder.name,flush=True)
  while any(j[0].poll() is None for j in jobs):
   state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[j[0].poll() for j in jobs];save(status,state);time.sleep(2)
  codes=[j[0].returncode for j in jobs]
  for _,log,_ in jobs:log.close()
  if any(c!=0 for c in codes):state['failed'].append(dict(contest_id=cid,attempt=folder.name,reason='Nonzero sampler exit',codes=codes));save(status,state);break
  assert all(completed(f) and e.csv_complete(f,samples) for f in paths)
  if process(paths,folder.name,False):accepted=True;break
  previous=paths
 if not accepted and not any(r['contest_id']==cid for r in state['failed']):state['failed'].append(dict(contest_id=cid,reason='Final-median protocol schedule exhausted'))
 state['current']=None;save(status,state)
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_remaining_19_accepted']=len(state['finished'])==19;save(status,state);print(MODEL_KIND,'FINAL MEDIAN BATCH',len(state['finished']),len(state['failed']),flush=True)


```
