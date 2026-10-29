"""Recover missing original leading-order posteriors; never fit nonlinear policies."""
from pathlib import Path
import concurrent.futures as futures
import csv,hashlib,json,math,os,subprocess,threading,time
import numpy as np
from pipeline.empirical import estimate as e
from pipeline.empirical import unified_prior_reestimate as u
import ast, types
protocol_path=Path(u.__file__).with_name("unified_prior_medians.py")
protocol_tree=ast.parse(protocol_path.read_text())
protocol_function=next(n for n in protocol_tree.body if isinstance(n,ast.FunctionDef) and n.name=="diagnose")
protocol_ns=dict(u=u,np=np,RAW_DIAGNOSE=u.diagnose,__file__=str(protocol_path))
exec(compile(ast.Module(body=[protocol_function],type_ignores=[]),str(protocol_path),"exec"),protocol_ns)
final=types.SimpleNamespace(diagnose=protocol_ns["diagnose"])
from pipeline.config import MAIN_DATA_DIR
root=Path.cwd();report=root/'docs/review/PaperJK9';out=report/'nonlinear_table_rebuild_20261025';frozen=report/'reconstruction_20261025/model_snapshot/detail_model';original=u.OUT
entries={int(r['contest_id']):r for r in csv.DictReader((original/'estimation/accepted_chain_sources.csv').open()) if r['mode']=='full'}
medians={int(r['contest_id']):r for r in csv.DictReader((original/'estimation/posterior_summary_final.csv').open()) if r['mode']=='full'}
lock=threading.Lock();state=dict(status='running',pid=os.getpid(),model='original_leading_order_Bayesian_posterior',nonlinear_reestimation=False,total=len(entries),completed={},failed={},active={});dest=out/'posterior_status.json'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def update():
 state['updated_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();save(dest,state)
def accepted_ok(path):
 if not path.exists():return False
 d=json.loads(path.read_text());return d['diagnostics']['passed'] and len(d['chain_paths'])==4 and all(Path(p).is_file() and digest(p)==s for p,s in zip(d['chain_paths'],d['chain_sha256']))
def fit(cid):
 for path in [report/f'reconstruction_20261025/estimation/full/contest_{cid}/accepted.json',out/f'posterior/contest_{cid}/accepted.json']:
  if accepted_ok(path):return str(path)
 entry=entries[cid];raw=MAIN_DATA_DIR/f'contest_{cid}.json';assert digest(raw)==entry['input_sha256'];data=e.stan_data(json.loads(raw.read_text()),'full');inputs=dict(data,**u.prior(data));folder=out/f'posterior/contest_{cid}';row=medians[cid];config=dict(contest_id=cid,mode='full',raw_input_sha256=digest(raw),source_model_sha256=digest(frozen.with_suffix('.stan')),executable_sha256=digest(frozen),original_runner_sha256=digest(Path(u.__file__)),sampler='CmdStan 2.36.0 NUTS',chains=4,prior=u.prior(data),initialization='Dispersed same-prior saved marginal medians; starts only, never reused as posterior draws',nonlinear_reestimation=False);save(folder/'configuration.json',config);previous=None
 for attempt,(warmup,samples,delta) in enumerate([(1000,2000,.98),(2000,4000,.99),(4000,8000,.995)],1):
  run=folder/f'attempt{attempt}';run.mkdir(parents=True,exist_ok=True);save(run/'data.json',inputs);paths=[run/f'chain_{i}.csv' for i in range(1,5)];procs=[]
  with lock:state['active'][str(cid)]=dict(attempt=attempt,warmup=warmup,samples=samples);update()
  for i,path in enumerate(paths,1):
   if path.exists() and e.csv_complete(path,samples):continue
   if path.exists():path.rename(path.with_suffix(f'.interrupted_{time.time_ns()}.csv'))
   if previous is None:
    factor=[.8,1.,1.2,1.5][i-1];init={k:max(.0101,float(row[k+'_median'])*factor) for k in ['c_i','c_j','sigma','lambda']};init.update(mu_0=float(row['mu_0_median'])+(i-2.5)*.05,log_r=math.log(float(row['r_median']))+(i-2.5)*.1)
   else:
    cols=e.read_columns(previous[i-1]);init={k:float(np.median(cols[k])) for k in e.PARAMETERS}
   ip=run/f'init_{i}.json';save(ip,init);log=(run/f'chain_{i}.log').open('w');cmd=[str(frozen),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261025+cid}',f'id={i}','data',f"file={run/'data.json'}",'output',f'file={path}','refresh=100','sig_figs=18',f'init={ip}'];process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));procs.append((process,log))
  for process,log in procs:
   code=process.wait();log.close()
   if code:raise RuntimeError(f'CmdStan failed: {cid}, exit {code}')
  assert all(e.csv_complete(p,samples) for p in paths);summary,diag=final.diagnose(cid,'full',paths,data);result=dict(summary=summary,diagnostics=diag,chain_paths=[str(p) for p in paths],chain_sha256=[digest(p) for p in paths],attempt=attempt,configuration=config);save(run/'diagnostics.json',result)
  print('POSTERIOR',cid,'attempt',attempt,'passed',diag['passed'],flush=True)
  if diag['passed']:save(folder/'accepted.json',result);return str(folder/'accepted.json')
  previous=paths
 raise RuntimeError('Frozen sampling schedules exhausted without accepted posterior')
if __name__=='__main__':
 import fcntl
 handle=(out/'posterior_controller.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);assert len(entries)==73;update()
 with futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(fit,cid):cid for cid in sorted(entries)}
  for job in futures.as_completed(jobs):
   cid=jobs[job]
   with lock:
    try:state['completed'][str(cid)]=job.result()
    except Exception as error:state['failed'][str(cid)]=repr(error)
    state['active'].pop(str(cid),None);update()
   print('POSTERIOR COVERAGE',len(state['completed']),'/73','failed',len(state['failed']),flush=True)
 state['status']='completed' if len(state['completed'])==73 else 'completed_with_failures';update()
