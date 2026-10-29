# Task 4：同过滤器的自由活动核 C

> 诊断口径更正（2026-10-25）：以下历史批次使用了较早 runner 的额外均值 MCSE<=3% 门槛。原最终冻结流程实际使用 `unified_prior_medians.py`，长尾均值误差单列，不作为最终接受门槛。本批已经通过的链仍有效；延长抽样不是最终协议的必需条件。后续接续见[最终协议方法](task4_final_protocol_training_method_20261025.md)。保留以下当时的执行源和记录供追溯。


日期：2026-10-25。审查用独立 Stan 快照，不修改原模型。C 保留原六参数先验、下界、观测似然、初始化、左端强度和净漂移；仅用自由 deadline_power=a、gap_log_scale=log(b) 替代原核两个固定系数。经验初始方差是 bar_S=sigma/sqrt(lambda)，故 Qref=bar_S+sigma²*T，而非额外估计一个初始方差。c 在此只解释为活动尺度。

额外先验 a~N(.5,.5²)、log(b)~N(0,.5²)，不使用预测期选择先验。a=.5、b=1 严格嵌套 D。编译已成功；在六竞赛 early 数据、原 early 中位参数上，两个可执行文件的完整 log density 差为 0、共同六参数梯度最大差 1.71e-13。核验记录在 `prediction_reconstruction_20261025/C/model_snapshot/nested_verification.json`，输入、参数和可执行输出另存 nested_checks/。这验证嵌套实现，不证明拟合或预测优劣。

四链 NUTS；沿用 D 的诊断门槛并额外检查 a、log(b)、b 的 rank Rhat<=1.01、bulk/tail ESS>=400。共同 sigma/lambda 均值相对 MCSE<=.03，无发散、无深度饱和、BFMI>=.3。不因额外参数难拟合放宽门槛。初值只取旧 early 汇总及固定分散的新核参数；所有抽样均为新拟合。目标全部 73 个原小时切分，不挑有利竞赛。失败保留；完整日期切分仍需另拟合。

[实时状态](task4_C_training_status_20261025.json)。同一批次运行时不得重启；若进程终止需检查已经完整的链与诊断后显式恢复。所有新文件仅在审查目录。以下是完整执行快照。

```python
import csv,json,hashlib,math,os,subprocess,time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as estimator
from pipeline.empirical import unified_prior_reestimate as original

from pipeline.empirical import nonlinear_reestimate as nonlinear
def diagnose_C(cid,paths,data):
 row,diagnostics=original.diagnose(cid,'early',paths,data)
 chains=[estimator.read_columns(f) for f in paths]
 arrays={k:np.asarray([c[k] for c in chains]) for k in ('deadline_power','gap_log_scale')};arrays['gap_scale']=np.exp(arrays['gap_log_scale'])
 for name,v in arrays.items():
  bulk,tail=nonlinear.rank_ess(v);ess=nonlinear.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(1,ess)))
  q=dict(rank_rhat=float(nonlinear.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None)
  diagnostics['quantities'][name]=q;row[name+'_mean']=mean;row[name+'_median']=float(np.median(v));row[name+'_rank_rhat']=q['rank_rhat'];row[name+'_bulk_ess']=q['bulk_ess'];row[name+'_tail_ess']=q['tail_ess'];row[name+'_mean_mcse']=mcse
  diagnostics['passed']=bool(diagnostics['passed'] and np.isfinite(v).all() and q['rank_rhat']<=1.01 and min(bulk,tail)>=400)
 return row,diagnostics

report=Path('docs/review/PaperJK9');out=report/'prediction_reconstruction_20261025/C/early';model=(report/'prediction_reconstruction_20261025/C/model_snapshot/free_kernel').resolve();status=report/'task4_C_training_status_20261025.json'
if status.exists():raise RuntimeError('Existing training reconstruction: inspect live handle before resuming; do not overwrite')
source_root=Path('results/reestimation/lambda100_sd225');entries={int(r['contest_id']):r for r in csv.DictReader((source_root/'estimation/accepted_chain_sources.csv').open()) if r['mode']=='early'}
summary={int(r['contest_id']):r for r in csv.DictReader((source_root/'estimation/posterior_summary_final.csv').open()) if r['mode']=='early'}
verified=json.loads((source_root/'verified_fits.json').read_text());budgets={r['contest_id']:r['draws_per_chain'] for r in verified['checks'] if r['scenario']=='main' and r['mode']=='early'}
checks=json.loads((report/'task4_source_information_audit_20261025.json').read_text());assert checks['replay_passed_contests']==73 and checks['initial_sources_before_cutoff_contests']==73
assert json.loads((report/'prediction_reconstruction_20261025/C/model_snapshot/nested_verification.json').read_text())['all_passed']
state=dict(status='running',driver_pid=os.getpid(),scope='Fit all 73 early posteriors for free nonstrategic activity kernel C, same filter and six shared priors as D; no full posterior or holdout fitting',cases=sorted(entries),finished=[],failed=[],current=None,all_training_posteriors_accepted=False,production_results_overwritten=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False));temp.replace(p)
save(status,state);clock=time.monotonic()
for cid in sorted(entries):
 entry=entries[cid];rawpath=source_root/'input_snapshots/main'/f"contest_{cid}_{entry['input_sha256']}.json";assert digest(rawpath)==entry['input_sha256'];raw=json.loads(rawpath.read_text());data=estimator.stan_data(raw,'early');inputs=dict(data,**original.prior(data));old=summary[cid]
 center={k:float(old[k+'_median']) for k in ('c_i','c_j','sigma','lambda','mu_0','r')};case=out/f'contest_{cid}'
 cfg=dict(contest_id=cid,mode='early',model='C_free_activity_kernel',input_sha256=digest(rawpath),inputs_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),source_model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),data_builder_sha256=digest(Path(estimator.__file__)),prior_runner_sha256=digest(Path(original.__file__)),training_cutoff=data['N_Delta'],full_constructed_deadline=data['T_contest'],prior=original.prior(data),sampler='CmdStan 2.36 NUTS, 4 chains',extra_priors=dict(deadline_power='Normal(.5,.5)',gap_log_scale='Normal(0,.5)'),initialization_source='Saved early-only posterior medians, dispersed; no holdout/full draw reuse',source_accepted_draws_per_chain=budgets[cid])
 save(case/'configuration.json',cfg)
 schedule=[(1,1000,2000,.98),(2,2000,4000,.99),(3,4000,8000,.995),(4,2000,32000,.995),(5,4000,64000,.9975)]
 schedule=[s for s in schedule if s[2]>=budgets[cid]];previous=None;accepted=False
 for attempt,warmup,samples,delta in schedule:
  folder=case/f'attempt{attempt}';folder.mkdir(parents=True,exist_ok=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];jobs=[]
  for j,path in enumerate(paths,1):
   if previous is None:
    factor=(.8,1.,1.2,1.5)[j-1];init={k:max(.0101,center[k]*factor) for k in ('c_i','c_j','sigma','lambda')};init['mu_0']=center['mu_0']+(j-2.5)*.05;init['log_r']=max(math.log(.01)+.001,math.log(center['r'])+(j-2.5)*.1);init['deadline_power']=(.2,.4,.6,.8)[j-1];init['gap_log_scale']=(-.2,-.05,.05,.2)[j-1]
   else:
    col=estimator.read_columns(previous[j-1]);init={k:float(np.median(col[k])) for k in (*estimator.PARAMETERS,'deadline_power','gap_log_scale')}
   initpath=folder/f'init_{j}.json';save(initpath,init);log=(folder/f'chain_{j}.log').open('w')
   command=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261026+200000+cid}',f'id={j}','data',f"file={folder/'data.json'}",'output',f'file={path}','refresh=500','sig_figs=18',f'init={initpath}']
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((process,log,command))
  state['current']=dict(contest_id=cid,attempt=attempt,warmup=warmup,samples=samples,delta=delta,pids=[v[0].pid for v in jobs],commands=[v[2] for v in jobs]);save(status,state);print('TRAIN C',cid,'attempt',attempt,'samples',samples,flush=True)
  while any(v[0].poll() is None for v in jobs):
   state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[v[0].poll() for v in jobs];save(status,state);time.sleep(2)
  codes=[v[0].returncode for v in jobs]
  for _,log,_ in jobs:log.close()
  if any(code!=0 for code in codes):state['failed'].append(dict(contest_id=cid,attempt=attempt,reason='CmdStan nonzero exit',codes=codes));save(status,state);break
  assert all(estimator.csv_complete(path,samples) for path in paths)
  row,diagnostics=diagnose_C(cid,paths,data);result=dict(summary=row,diagnostics=diagnostics,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],attempt=attempt,configuration=cfg);save(folder/'diagnostics.json',result)
  print('TRAIN DIAGNOSTICS',cid,'attempt',attempt,'passed',diagnostics['passed'],flush=True)
  if diagnostics['passed']:
   save(case/'accepted.json',result);state['finished'].append(dict(contest_id=cid,accepted=str((case/'accepted.json').resolve()),attempt=attempt));save(status,state);accepted=True;break
  previous=paths
 if not accepted and not any(v['contest_id']==cid for v in state['failed']):state['failed'].append(dict(contest_id=cid,reason='Diagnostic schedule exhausted'))
 state['current']=None;save(status,state)
state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_training_posteriors_accepted']=len(state['finished'])==73;save(status,state);print('Training reconstruction batch ended',len(state['finished']),'accepted',len(state['failed']),'failed',flush=True)

```
