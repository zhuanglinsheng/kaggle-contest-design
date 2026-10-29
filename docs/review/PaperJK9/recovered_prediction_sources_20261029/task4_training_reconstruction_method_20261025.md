# Task 4：73 个训练窗口后验重建

> 诊断口径更正（2026-10-25）：以下历史批次使用了较早 runner 的额外均值 MCSE<=3% 门槛。原最终冻结流程实际使用 `unified_prior_medians.py`，长尾均值误差单列，不作为最终接受门槛。本批已经通过的链仍有效；延长抽样不是最终协议的必需条件。后续接续见[最终协议方法](task4_final_protocol_training_method_20261025.md)。保留以下当时的执行源和记录供追溯。


日期：2026-10-25。用户已授权搜索无果后重建后验；这里重建当前统一先验的全部 73 个 early 后验，用于 D 的实际预测比较，不用 full 后验预测训练外计数。

使用同一原模型逐字节快照和原 prior/data builder；冻结输入已从源提交回放一致，初始化源提交均在训练内。初始化仅取原保存 early 汇总的分散中位数，不复用缺失旧抽样或 full 抽样。已知原接受抽样预算较大的案例直接从相应日程开始，避免重复已知不足的小预算。

本历史批次采用含均值精度条件的四链门槛，不合格时延长至每链 32,000/64,000；不因预测用途放宽参数采样门槛。目标是 73 个竞赛全部拟合；失败保留待处理，不删除或以已有成功案例替代。

原构造窗口终点在预测比较中作为给定条件，保留事后焦点对及日内时间为模拟的限制。当前是原小时切分的模型 D；C 和日期边界稳健性尚待实际拟合。

[实时批次状态](task4_training_reconstruction_status_20261025.json)。仅当确认原执行句柄终止后才讨论恢复；不要按旧 running 标签重复启动。新链在审查目录独立保存，不改生产模型、输入、论文或旧估计。

```python
import csv,json,hashlib,math,os,subprocess,time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as estimator
from pipeline.empirical import unified_prior_reestimate as original
report=Path('docs/review/PaperJK9');out=report/'prediction_reconstruction_20261025/D/early';model=(report/'reconstruction_20261025/model_snapshot/detail_model').resolve();status=report/'task4_training_reconstruction_status_20261025.json'
if status.exists():raise RuntimeError('Existing training reconstruction: inspect live handle before resuming; do not overwrite')
source_root=Path('results/reestimation/lambda100_sd225');entries={int(r['contest_id']):r for r in csv.DictReader((source_root/'estimation/accepted_chain_sources.csv').open()) if r['mode']=='early'}
summary={int(r['contest_id']):r for r in csv.DictReader((source_root/'estimation/posterior_summary_final.csv').open()) if r['mode']=='early'}
verified=json.loads((source_root/'verified_fits.json').read_text());budgets={r['contest_id']:r['draws_per_chain'] for r in verified['checks'] if r['scenario']=='main' and r['mode']=='early'}
checks=json.loads((report/'task4_source_information_audit_20261025.json').read_text());assert checks['replay_passed_contests']==73 and checks['initial_sources_before_cutoff_contests']==73
state=dict(status='running',driver_pid=os.getpid(),scope='Reconstruct all 73 current-prior early posteriors for primary predictive comparison D; does not use full posterior for forecasting',cases=sorted(entries),finished=[],failed=[],current=None,all_training_posteriors_accepted=False,production_results_overwritten=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False));temp.replace(p)
save(status,state);clock=time.monotonic()
for cid in sorted(entries):
 entry=entries[cid];rawpath=source_root/'input_snapshots/main'/f"contest_{cid}_{entry['input_sha256']}.json";assert digest(rawpath)==entry['input_sha256'];raw=json.loads(rawpath.read_text());data=estimator.stan_data(raw,'early');inputs=dict(data,**original.prior(data));old=summary[cid]
 center={k:float(old[k+'_median']) for k in ('c_i','c_j','sigma','lambda','mu_0','r')};case=out/f'contest_{cid}'
 cfg=dict(contest_id=cid,mode='early',model='D_restricted_kernel',input_sha256=digest(rawpath),inputs_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),source_model_sha256=digest(model.with_suffix('.stan')),executable_sha256=digest(model),data_builder_sha256=digest(Path(estimator.__file__)),prior_runner_sha256=digest(Path(original.__file__)),training_cutoff=data['N_Delta'],full_constructed_deadline=data['T_contest'],prior=original.prior(data),sampler='CmdStan 2.36 NUTS, 4 chains',initialization_source='Saved early-only posterior medians, dispersed; no holdout/full draw reuse',source_accepted_draws_per_chain=budgets[cid])
 save(case/'configuration.json',cfg)
 schedule=[(1,1000,2000,.98),(2,2000,4000,.99),(3,4000,8000,.995),(4,2000,32000,.995),(5,4000,64000,.9975)]
 schedule=[s for s in schedule if s[2]>=budgets[cid]];previous=None;accepted=False
 for attempt,warmup,samples,delta in schedule:
  folder=case/f'attempt{attempt}';folder.mkdir(parents=True,exist_ok=True);save(folder/'data.json',inputs);paths=[folder/f'chain_{j}.csv' for j in range(1,5)];jobs=[]
  for j,path in enumerate(paths,1):
   if previous is None:
    factor=(.8,1.,1.2,1.5)[j-1];init={k:max(.0101,center[k]*factor) for k in ('c_i','c_j','sigma','lambda')};init['mu_0']=center['mu_0']+(j-2.5)*.05;init['log_r']=max(math.log(.01)+.001,math.log(center['r'])+(j-2.5)*.1)
   else:
    col=estimator.read_columns(previous[j-1]);init={k:float(np.median(col[k])) for k in estimator.PARAMETERS}
   initpath=folder/f'init_{j}.json';save(initpath,init);log=(folder/f'chain_{j}.log').open('w')
   command=[str(model),'sample',f'num_warmup={warmup}',f'num_samples={samples}','adapt',f'delta={delta}','algorithm=hmc','engine=nuts','max_depth=13','metric=dense_e','random',f'seed={20261026+100000+cid}',f'id={j}','data',f"file={folder/'data.json'}",'output',f'file={path}','refresh=500','sig_figs=18',f'init={initpath}']
   process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'));jobs.append((process,log,command))
  state['current']=dict(contest_id=cid,attempt=attempt,warmup=warmup,samples=samples,delta=delta,pids=[v[0].pid for v in jobs],commands=[v[2] for v in jobs]);save(status,state);print('TRAIN D',cid,'attempt',attempt,'samples',samples,flush=True)
  while any(v[0].poll() is None for v in jobs):
   state['elapsed_seconds']=time.monotonic()-clock;state['current']['process_return_codes']=[v[0].poll() for v in jobs];save(status,state);time.sleep(2)
  codes=[v[0].returncode for v in jobs]
  for _,log,_ in jobs:log.close()
  if any(code!=0 for code in codes):state['failed'].append(dict(contest_id=cid,attempt=attempt,reason='CmdStan nonzero exit',codes=codes));save(status,state);break
  assert all(estimator.csv_complete(path,samples) for path in paths)
  row,diagnostics=original.diagnose(cid,'early',paths,data);result=dict(summary=row,diagnostics=diagnostics,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],attempt=attempt,configuration=cfg);save(folder/'diagnostics.json',result)
  print('TRAIN DIAGNOSTICS',cid,'attempt',attempt,'passed',diagnostics['passed'],flush=True)
  if diagnostics['passed']:
   save(case/'accepted.json',result);state['finished'].append(dict(contest_id=cid,accepted=str((case/'accepted.json').resolve()),attempt=attempt));save(status,state);accepted=True;break
  previous=paths
 if not accepted and not any(v['contest_id']==cid for v in state['failed']):state['failed'].append(dict(contest_id=cid,reason='Diagnostic schedule exhausted'))
 state['current']=None;save(status,state)
state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_training_posteriors_accepted']=len(state['finished'])==73;save(status,state);print('Training reconstruction batch ended',len(state['finished']),'accepted',len(state['failed']),'failed',flush=True)
```
