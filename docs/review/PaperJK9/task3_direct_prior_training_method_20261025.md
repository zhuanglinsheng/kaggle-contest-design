# Task 3：失败重要性场景与下界放宽的直接拟合

目标：13 个未通过重要性诊断的先验场景不再以旧抽样加权作结论；另对全部六例将成本下界从 .01 放宽一数量级至 .001，补足原支持之外的敏感性分析。任务表在拟合前固定为 19 项，保留全部失败。

独立审查模型仅修改成本的共同/相对 log 先验及成本下界。令 u=(log ci+log cj)/2，v=log ci-log cj，密度为 Normal(u|mean,sd)*Normal(v|0,.75*sqrt2)/(ci*cj)，在双方下界截断；手工加入 -log ci-log cj 的 Jacobian。其他先验、数据、稳态方差、过滤器、似然、奖金单位均不变。共同下界仍耦合 u/v 的截断边际，不宣称其边际独立不变。

基准 mean=0、sd=.75/sqrt2、floor=.01 通过六个 full 数据的可执行文件密度/梯度校验。propto log density 相差固定 log(2*pi)，原因为等价分布的参数无关常数处理；扣除该常数差小于 7.4e-14，共同梯度差小于 2e-15。首次错误要求密度差严格为零的检查及原输出保留在 nested_verification.json 的说明中。

使用仓库最终中位数接受协议，额外检查 u/v/cost ratio 的 rank/ESS；保留长尾均值 MCSE，预测/政策积分另核验。与 Task 4 同样通过 AST 原样调用最终 diagnose 函数，避免无关 cmdstanpy 辅助导入。四链 dense NUTS，2k/4k/8k/32k/64k 日程，失败不填结论；成本下界改变后通过 Stan 自动约束 Jacobian 采样，不将旧支持的样本当成新支持完整抽样。

[实时状态](prior_reconstruction_20261025/status.json)。通过诊断后仍需计算两种质量目标下的政策结果，核验 Monte Carlo 精度，不能将拟合完成称作反事实稳健性完成。原论文、实现、汇总和重要性初筛均不覆盖。

```python
import csv,json,math,hashlib,os,time,subprocess,ast,types
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9');root=p/'prior_reconstruction_20261025';model=(root/'model_snapshot/common_cost').resolve();status=root/'status.json';assert not status.exists(),'Inspect existing prior driver before resume';assert json.loads((root/'model_snapshot/nested_verification.json').read_text())['all_passed']
mp=Path('pipeline/empirical/unified_prior_medians.py');mt=mp.read_text();mf=next(n for n in ast.parse(mt).body if isinstance(n,ast.FunctionDef) and n.name=='diagnose');ns={'u':u,'np':np,'RAW_DIAGNOSE':u.diagnose,'__file__':str(mp)};exec(compile(ast.get_source_segment(mt,mf),str(mp),'exec'),ns);median_diagnose=ns['diagnose']
screen=json.loads((p/'task3_common_cost_prior_screen_20261025.json').read_text());order=[20270,2464,2551,5144,23652,8219];tasks=[r for cid in order for r in screen['records'] if r['contest_id']==cid and not r['diagnostics']['passed']];tasks +=[dict(contest_id=cid,scenario='cost_floor_0001',common_log_mean=0.,common_log_sd=.75/math.sqrt(2),relative_log_cost_sd=.75*math.sqrt(2),cost_floor=.001) for cid in order];assert len(tasks)==19
state=dict(status='running',driver_pid=os.getpid(),scope='Fresh joint-cost-prior full posteriors for 13 failed importance scenarios and six lower-cost-floor cases; no counterfactual result asserted before diagnosis',tasks=[{k:t[k] for k in ['contest_id','scenario','common_log_mean','common_log_sd','relative_log_cost_sd','cost_floor']} for t in tasks],finished=[],failed=[],current=None,all_19_accepted=False,final_median_protocol=True,production_files_overwritten=False)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):
 f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(f)
def diagnose(cid,paths,data):
 row,d=median_diagnose(cid,'full',paths,data);cc=[e.read_columns(f) for f in paths];lc=np.log(np.asarray([c['c_i'] for c in cc]));lj=np.log(np.asarray([c['c_j'] for c in cc]));arrays=dict(common_log_cost=(lc+lj)/2,relative_log_cost=lc-lj,cost_ratio=np.exp(lc-lj))
 for name,v in arrays.items():
  bulk,tail=nd.rank_ess(v);ess=nd.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));q=dict(rank_rhat=float(nd.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);d['quantities'][name]=q;row[name+'_median']=float(np.median(v));row[name+'_mean']=mean;row[name+'_mean_mcse']=mcse;d['passed']=bool(d['passed'] and np.isfinite(v).all() and q['rank_rhat']<=1.01 and min(bulk,tail)>=400)
 return row,d
save(status,state);clock=time.monotonic()
for index,t in enumerate(tasks):
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
state['status']='completed' if not state['failed'] else 'completed_with_failures';state['all_19_accepted']=len(state['finished'])==19;save(status,state);print('PRIOR BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
