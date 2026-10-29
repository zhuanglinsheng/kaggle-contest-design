# Task 4：按原最终中位数协议核验与接续

此前误把较早 `unified_prior_reestimate.diagnose` 中 sigma/lambda 均值相对 MCSE<=3% 当成最终接受门槛。实际冻结协议是 `unified_prior_medians.py`，由 `unified_prior_verify.py` 核验；`report.txt` 已披露 35 个长尾 lambda 均值未达3%，正文/表格使用中位数。原估计不存在因本次重建而新发现的门槛违规。

为纠正该口径，已核对并停止两个旧驱动及各自采样子进程，保留全部完整/部分链和历史诊断，旧状态标明 stopped_for_final_protocol_reconciliation。重新使用最终原协议逐组核验完整旧链；原文件不覆盖，新诊断及接受索引用 `_median_protocol` 后缀。缺少完整可接受链的竞赛才接续采样，新文件使用 median_attempt*，不覆写旧半成品。

最终门槛：全部所查原参数/派生 w 的 rank Rhat<=1.01、bulk/tail ESS>=400；无发散、树深饱和，BFMI>=.3。保存均值与相对 MCSE、sigma/lambda 中位指示量 ESS 与 log 中位数 MCSE。C 额外检查 a、log(b)、b 的 rank/ESS。预测积分的批次 MCSE及逐时贡献集中度另核验，不以参数诊断替代预测精度。

分别用 MODEL_KIND=D/C 执行同一快照。实时状态 `task4_D_median_training_status_20261025.json`、`task4_C_median_training_status_20261025.json`；后续评分从这些状态的 accepted 路径读取。已有六个 full 后验通过了此前更强门槛，继续有效，无需为此重建。

执行环境说明：整模块导入会因未安装 cmdstanpy 的无关辅助导入失败，两个首次调用均在创建状态/采样前退出。接续通过 AST 原样提取最终协议的 diagnose 函数，使用该函数原有依赖 u/np/RAW_DIAGNOSE 和源路径执行；不改函数、不安装包、不修改实现。源 SHA 仍指向原最终协议文件。

```python
import csv,json,hashlib,math,os,subprocess,time,re
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nonlinear
import ast,types
median_path=Path('pipeline/empirical/unified_prior_medians.py');median_text=median_path.read_text();median_tree=ast.parse(median_text);median_function=next(node for node in median_tree.body if isinstance(node,ast.FunctionDef) and node.name=='diagnose');median_namespace={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(median_path)};exec(compile(ast.get_source_segment(median_text,median_function),str(median_path),'exec'),median_namespace);med=types.SimpleNamespace(diagnose=median_namespace['diagnose'],__file__=str(median_path))
MODEL_KIND=globals().get('MODEL_KIND','D');assert MODEL_KIND in ('C','D')
p=Path('docs/review/PaperJK9');base=p/'prediction_reconstruction_20261025'/MODEL_KIND/'early';model=(p/('reconstruction_20261025/model_snapshot/detail_model' if MODEL_KIND=='D' else 'prediction_reconstruction_20261025/C/model_snapshot/free_kernel')).resolve();status=p/f'task4_{MODEL_KIND}_median_training_status_20261025.json'
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
state=dict(status='running',driver_pid=os.getpid(),model=MODEL_KIND,protocol='Final frozen primary-median protocol, not earlier raw-mean gate',median_protocol_sha256=digest(Path(med.__file__)),raw_means_retained_with_mcse=True,cases=sorted(summary),finished=[],failed=[],current=None,all_training_posteriors_accepted=False,production_results_overwritten=False);save(status,state);clock=time.monotonic()
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
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_training_posteriors_accepted']=len(state['finished'])==73;save(status,state);print(MODEL_KIND,'FINAL MEDIAN BATCH',len(state['finished']),len(state['failed']),flush=True)


```
