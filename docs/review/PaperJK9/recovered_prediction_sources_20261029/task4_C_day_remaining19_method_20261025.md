# Task 4：C日期切分剩余19例接续方法（已启动）

8219同窗口补充已通过，原驱动在该例安全停止，现已启动。只处理ID>8219的19例；原53例和补充接受记录分别保留。输入、完整期限、冷启动、复用门槛和诊断不变。原驱动若已经推进，则不启动这个接续批次。

```python
import ast,types,json,math,hashlib,subprocess,os,time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nonlinear
MODEL_KIND=globals().get('MODEL_KIND','D');assert MODEL_KIND in ['C','D'];p=Path('docs/review/PaperJK9');status=p/f'task4_{MODEL_KIND}_complete_day_resume_status_20261025.json';assert not status.exists(),'Inspect existing day-fold driver before resume';prepared=json.loads((p/'task4_complete_day_inputs_20261025.json').read_text());assert prepared['contests']==73;assert MODEL_KIND=='C';prepared['records']=[r for r in prepared['records'] if r['contest_id']>8219];assert len(prepared['records'])==19;model=(p/('reconstruction_20261025/model_snapshot/detail_model' if MODEL_KIND=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();base=p/'prediction_reconstruction_20261025'/MODEL_KIND/'complete_day'
mp=Path('pipeline/empirical/unified_prior_medians.py');mt=mp.read_text();fn=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');ns={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(mp)};exec(compile(ast.get_source_segment(mt,fn),str(mp),'exec'),ns);med=types.SimpleNamespace(diagnose=ns['diagnose'],__file__=str(mp))
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,value):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(f)
def diagnose(cid,paths,data):
 row,d=med.diagnose(cid,'complete_day',paths,data)
 if MODEL_KIND=='C':
  cc=[e.read_columns(f) for f in paths];arrays={k:np.array([c[k] for c in cc]) for k in ['deadline_power','gap_log_scale']};arrays['gap_scale']=np.exp(arrays['gap_log_scale'])
  for name,v in arrays.items():
   bulk,tail=nonlinear.rank_ess(v);ess=nonlinear.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));d['quantities'][name]=dict(rank_rhat=float(nonlinear.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);row[name+'_median']=float(np.median(v));row[name+'_mean']=mean;row[name+'_mean_mcse']=mcse;d['passed']=bool(d['passed'] and np.isfinite(v).all() and d['quantities'][name]['rank_rhat']<=1.01 and min(bulk,tail)>=400)
 return row,d
def primary_entries():
 entries={}
 for name in [f'task4_{MODEL_KIND}_median_training_status_20261025.json',f'task4_{MODEL_KIND}_primary_resume_status_20261025.json']:
  f=p/name
  if f.exists():
   for r in json.loads(f.read_text())['finished']:entries[r['contest_id']]=r
 f=p/'task4_primary_supplemental_acceptances_20261025.json'
 if f.exists():
  for r in json.loads(f.read_text())['entries']:
   if r['model']==MODEL_KIND:assert digest(r['accepted'])==r['accepted_sha256'];entries[r['contest_id']]=r
 return entries
state=dict(status='running',driver_pid=os.getpid(),model=MODEL_KIND,split='complete_day',scope='Remaining19 complete-source-day training windows after verified same-target 8219 supplement, same full constructed horizon; cold prior-based starts, or identical-input primary reuse only',cases=[r['contest_id'] for r in prepared['records']],finished=[],failed=[],current=None,all_remaining_19_accepted=False,maximum_simultaneous_sampler_processes=2);save(status,state);clock=time.monotonic()
for record in prepared['records']:
 cid=record['contest_id'];path=Path(record['prepared_input']);assert digest(path)==record['prepared_input_sha256'];inputs=json.loads(path.read_text());data={k:v for k,v in inputs.items() if k not in ['prior_mean','prior_sd','sigma_floor']};prior=u.prior(data);assert dict(data,**prior)==inputs;case=base/f'contest_{cid}';cfg=dict(contest_id=cid,mode='complete_day',model=MODEL_KIND,input_sha256=record['source_input_sha256'],prepared_input_sha256=digest(path),model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),median_protocol_sha256=digest(mp),training_cutoff=data['N_Delta'],full_constructed_deadline=data['T_contest'],prior=prior,extra_priors=dict(deadline_power='Normal(.5,.5)',gap_log_scale='Normal(0,.5)') if MODEL_KIND=='C' else {},initialization='Prior-centered dispersed starts from this training window only',concurrent_chains=2,total_chains=4);save(case/'configuration.json',cfg)
 prim=primary_entries().get(cid)
 if record['cutoff_hour']==record['primary_cutoff_hour'] and prim is not None:
  ap=Path(prim['accepted']);a=json.loads(ap.read_text());sourcefolder=Path(a['chain_paths'][0]).parent;dp=sourcefolder/'data.json'
  if not dp.exists():dp=sourcefolder/'parity/data.json'
  assert json.loads(dp.read_text())==inputs;assert a['configuration']['model_sha256']==cfg['model_sha256'] and a['configuration']['prior']==prior and a['diagnostics']['passed'];assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));a=dict(a,configuration=cfg,reused_identical_training_target=str(ap.resolve()),primary_accepted_sha256=digest(ap));save(case/'accepted.json',a);state['finished'].append(dict(contest_id=cid,accepted=str((case/'accepted.json').resolve()),reused_identical_primary_target=True));save(status,state);print(MODEL_KIND,'DAY REUSE',cid,flush=True);continue
 previous=None;accepted=False
 for attempt,(warmup,samples,delta) in enumerate([(1000,2000,.98),(2000,4000,.99),(4000,8000,.995),(2000,32000,.995),(4000,64000,.9975)],1):
  folder=case/f'attempt{attempt}';assert not folder.exists(),'Preserve partial day-fold chains';folder.mkdir(parents=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];commands=[]
  for j,f in enumerate(paths,1):
   if previous is None:
    q=(-1.5,-.5,.5,1.5)[j-1];center=np.asarray(prior['prior_mean']);sd=np.asarray(prior['prior_sd']);z=center.copy();z[0]+=sd[0]*q;z[1]-=sd[1]*q;z[2]+=sd[2]*q;z[3]-=sd[3]*q;z[4]+=.5*q;z[5]-=sd[5]*q;init=dict(zip(e.PARAMETERS,np.r_[np.exp(z[:4]),z[4:]].tolist()))
    for name in ['c_i','c_j','sigma','lambda']:init[name]=max(.0101,init[name])
    init['log_r']=max(math.log(.01)+.001,init['log_r'])
    if MODEL_KIND=='C':init.update(deadline_power=(.2,.4,.6,.8)[j-1],gap_log_scale=(-.2,-.05,.05,.2)[j-1])
   else:
    c=e.read_columns(previous[j-1]);names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if MODEL_KIND=='C' else e.PARAMETERS;init={k:float(np.median(c[k])) for k in names}
   ip=folder/f'init_{j}.json';save(ip,init);commands.append([str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261025+1200000+(100000 if MODEL_KIND=="C" else 0)+cid}',f'id={j}','data',f'file={folder/"data.json"}','output',f'file={f}','refresh=500','sig_figs=18',f'init={ip}'])
  codes=[]
  for group in [range(0,2),range(2,4)]:
   jobs=[]
   for j in group:
    log=(folder/f'chain_{j+1}.log').open('w');job=subprocess.Popen(commands[j],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((job,log,j))
   state['current']=dict(contest_id=cid,attempt=attempt,samples=samples,chain_group=[j+1 for j in group],pids=[x[0].pid for x in jobs],commands=[commands[x[2]] for x in jobs]);save(status,state)
   while any(x[0].poll() is None for x in jobs):
    state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[x[0].poll() for x in jobs];save(status,state);time.sleep(2)
   codes.extend(x[0].returncode for x in jobs)
   for _,log,_ in jobs:log.close()
  if any(code!=0 for code in codes):state['failed'].append(dict(contest_id=cid,attempt=attempt,reason='Sampler exit',codes=codes));save(status,state);break
  assert all(e.csv_complete(f,samples) for f in paths);row,d=diagnose(cid,paths,data);a=dict(summary=row,diagnostics=d,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],configuration=cfg,attempt=attempt);save(folder/'diagnostics.json',a);print(MODEL_KIND,'DAY CHECK',cid,attempt,d['passed'],flush=True)
  if d['passed']:
   ap=case/'accepted.json';save(ap,a);state['finished'].append(dict(contest_id=cid,accepted=str(ap.resolve()),attempt=attempt,reused_identical_primary_target=False));save(status,state);accepted=True;break
  previous=paths
 if not accepted and not any(r['contest_id']==cid for r in state['failed']):state['failed'].append(dict(contest_id=cid,reason='Day-fold diagnostic schedule exhausted'))
 state['current']=None;save(status,state)
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_remaining_19_accepted']=len(state['finished'])==19;save(status,state);print(MODEL_KIND,'DAY BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
