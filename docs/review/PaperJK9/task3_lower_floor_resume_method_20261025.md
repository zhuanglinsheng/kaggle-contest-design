# Task 3：接续六个更低成本下界的拟合

13个原重要性失败场景已由12个NUTS接受后验及一个经过目标一致性/混合核验的MH后验补足。8219 floor=.05的旧NUTS驱动与子进程已停止，完整和部分链保留；补充接受记录另存，没有将未通过的NUTS诊断改为通过。

此批只接续原预先确定的六例floor=.001，保留原19项计划中的索引13..18及对应随机种子，不覆盖原路径。模型、先验、数据、支持与初值规则均沿用原下界放宽计划。新的状态文件独立，全部19项的政策汇总需合并原12项、本批6项及MH补充1项。

[实时状态](prior_reconstruction_20261025/lower_floor_status.json)。

```python
import csv,json,math,hashlib,os,time,subprocess,ast,types
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9');root=p/'prior_reconstruction_20261025';model=(root/'model_snapshot/common_cost').resolve();status=root/'lower_floor_status.json';assert not status.exists(),'Inspect existing prior driver before resume';assert json.loads((root/'model_snapshot/nested_verification.json').read_text())['all_passed']
mp=Path('pipeline/empirical/unified_prior_medians.py');mt=mp.read_text();mf=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');ns={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(mp)};exec(compile(ast.get_source_segment(mt,mf),str(mp),'exec'),ns);median_diagnose=ns['diagnose']
screen=json.loads((p/'task3_common_cost_prior_screen_20261025.json').read_text());order=[20270,2464,2551,5144,23652,8219];tasks=[r for cid in order for r in screen['records'] if r['contest_id']==cid and not r['diagnostics']['passed']];tasks +=[dict(contest_id=cid,scenario='cost_floor_0001',common_log_mean=0.,common_log_sd=.75/math.sqrt(2),relative_log_cost_sd=.75*math.sqrt(2),cost_floor=.001) for cid in order];assert len(tasks)==19
tasks=tasks[13:];assert len(tasks)==6 and all(t['scenario']=='cost_floor_0001' for t in tasks)
state=dict(status='running',driver_pid=os.getpid(),scope='Resume six lower-cost-floor cases; thirteen common-prior scenarios supplied by accepted NUTS plus independently validated same-target MH',tasks=[{k:t[k] for k in ['contest_id','scenario','common_log_mean','common_log_sd','relative_log_cost_sd','cost_floor']} for t in tasks],finished=[],failed=[],current=None,all_six_lower_floor_accepted=False,final_median_protocol=True,production_files_overwritten=False)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(f)
def diagnose(cid,paths,data):
 row,d=median_diagnose(cid,'full',paths,data);cc=[e.read_columns(f) for f in paths];lc=np.log(np.asarray([c['c_i'] for c in cc]));lj=np.log(np.asarray([c['c_j'] for c in cc]));arrays=dict(common_log_cost=(lc+lj)/2,relative_log_cost=lc-lj,cost_ratio=np.exp(lc-lj))
 for name,v in arrays.items():
  bulk,tail=nd.rank_ess(v);ess=nd.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));q=dict(rank_rhat=float(nd.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);d['quantities'][name]=q;row[name+'_median']=float(np.median(v));row[name+'_mean']=mean;row[name+'_mean_mcse']=mcse;d['passed']=bool(d['passed'] and np.isfinite(v).all() and q['rank_rhat']<=1.01 and min(bulk,tail)>=400)
 return row,d
save(status,state);clock=time.monotonic()
for index,t in enumerate(tasks,start=13):
 cid=t['contest_id'];label=t['scenario'];case=root/label/f'contest_{cid}';rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());data=e.stan_data(raw,'full');inputs=dict(data,**u.prior(data),cost_floor=t['cost_floor'],common_log_mean=t['common_log_mean'],common_log_sd=t['common_log_sd'],relative_log_sd=t['relative_log_cost_sd']);baseline=json.loads((p/'reconstruction_20261025/estimation/full'/f'contest_{cid}/accepted.json').read_text());assert baseline['diagnostics']['passed'];center={k:float(baseline['summary'][k+'_median']) for k in ['c_i','c_j','sigma','lambda','mu_0','r']};cfg=dict(contest_id=cid,mode='full',scenario=label,inputs=inputs,input_sha256=digest(rawpath),model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),median_protocol_sha256=digest(mp),scenario_defined_before_fit=True,baseline_initialization_only=True);save(case/'configuration.json',cfg);previous=None;accepted=False
 for attempt,(warmup,samples,delta) in enumerate([(1000,2000,.98),(2000,4000,.99),(4000,8000,.995),(2000,32000,.995),(4000,64000,.9975)],1):
  folder=case/f'attempt{attempt}';assert not folder.exists(),'Preserve interrupted chains';folder.mkdir(parents=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];jobs=[]
  for j,f in enumerate(paths,1):
   if previous is None:
    factor=(.8,1.,1.2,1.5)[j-1];init={k:max((t['cost_floor']*1.01 if k in ['c_i','c_j'] else .0101),center[k]*factor) for k in ['c_i','c_j','sigma','lambda']};init['mu_0']=center['mu_0']+(j-2.5)*.05;init['log_r']=max(math.log(.01)+.001,math.log(center['r'])+(j-2.5)*.1)
   else:
    col=e.read_columns(previous[j-1]);init={k:float(np.median(col[k])) for k in e.PARAMETERS}
   ip=folder/f'init_{j}.json';save(ip,init);log=(folder/f'chain_{j}.log').open('w');cmd=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261025+700000+cid+1000000*index}',f'id={j}','data',f'file={folder/"data.json"}','output',f'file={f}','refresh=500','sig_figs=18',f'init={ip}'];job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((job,log,cmd))
  state['current']=dict(contest_id=cid,scenario=label,attempt=attempt,samples=samples,pids=[j[0].pid for j in jobs],commands=[j[2] for j in jobs],process_return_codes=[None]*4);save(status,state);print('PRIOR FIT',cid,label,'attempt',attempt,flush=True)
  while any(j[0].poll() is None for j in jobs):
   state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[j[0].poll() for j in jobs];save(status,state);time.sleep(2)
  codes=[j[0].returncode for j in jobs]
  for _,log,_ in jobs:log.close()
  if any(c!=0 for c in codes):state['failed'].append(dict(contest_id=cid,scenario=label,attempt=attempt,reason='Nonzero sampler exit',codes=codes));save(status,state);break
  assert all(e.csv_complete(f,samples) for f in paths);row,d=diagnose(cid,paths,data);result=dict(summary=row,diagnostics=d,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],configuration=cfg,attempt=attempt);save(folder/'diagnostics.json',result);print('PRIOR DIAG',cid,label,attempt,d['passed'],flush=True)
  if d['passed']:
   ap=case/'accepted.json';save(ap,result);state['finished'].append(dict(contest_id=cid,scenario=label,accepted=str(ap.resolve()),attempt=attempt));save(status,state);accepted=True;break
  previous=paths
 if not accepted and not any(r['contest_id']==cid and r['scenario']==label for r in state['failed']):state['failed'].append(dict(contest_id=cid,scenario=label,reason='Final-median diagnostic schedule exhausted'))
 state['current']=None;save(status,state)
state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_six_lower_floor_accepted']=len(state['finished'])==6;save(status,state);print('PRIOR BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
