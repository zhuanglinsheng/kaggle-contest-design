# Task 4：最终中位数协议下 C/D 的统一评分

从新最终协议状态读取接受链；原始完整链通过同一最终门槛即可使用，lambda 长尾均值 MCSE 保留。每个模型仍使用实际联合后验积分，均值或中位数点估计均不能替代该积分。此前首例严格均值门槛的评分记录保留，新表单独用 `_median_forecast` 命名。

计数归一化、时序、嵌套核、独立标量验证和32个链内连续块 MCSE沿用之前评分方法。实际 C 递推显式 free=True；D 固定核；改变当前末读数不能影响已发出的当前预测。均值分数数值初筛目标 MCSE<=.001 nats/小时且链内分层半样本差<=max(.002,4MCSE)，这只是计算精度诊断，模型差仍须与自身积分误差比较，不能据此声称统一误差上界。原小时切分参数固定，更新仅发生于公共状态。

分别以 MODEL_KIND=D/C 执行。只覆盖执行时已接受的子集，全部73例之前不报告整体模型优劣。

```python
import csv,json,math,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from pipeline.empirical import estimate as e
MODEL_KIND=globals().get('MODEL_KIND','D');assert MODEL_KIND in ('C','D')
p=Path('docs/review/PaperJK9')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def counts_for(raw):
 N=raw['N_Delta'];out=[]
 for k in ('i','j'):
  t=np.asarray([float(v) for v in raw['hat_t_'+k] if 0<float(v)<=N]);out.append(np.bincount(np.maximum(1,np.ceil(t).astype(int))-1,minlength=N)[:N])
 return np.asarray(out)
def predictions(raw,columns,begin,end=None,free=False):
 data=e.stan_data(raw,'full');N=data['N_Delta'] if end is None else end;dt=data['Delta2f'];T=data['T_contest'];theta=data['theta'];cnt=counts_for(raw);n=len(columns['sigma']);y=columns['mu_0'].copy();sigma=columns['sigma'];lam=columns['lambda'];barS=sigma/np.sqrt(lam);K=-np.expm1(-np.sqrt(lam)*sigma*dt);Qref=barS+sigma*sigma*T
 logs=np.empty((n,N-begin));means=np.empty((n,N-begin,2));zeros=np.empty_like(means)
 for h in range(N):
  Q=barS+sigma*sigma*(T-h*dt)
  if free:logkernel=-.5*np.log(2*np.pi*Qref)-columns['deadline_power']*np.log(Q/Qref)-np.exp(columns['gap_log_scale'])*y*y/(2*Q)
  else:logkernel=-.5*np.log(2*np.pi*Q)-y*y/(2*Q)
  logmi=math.log(theta)-np.log(columns['c_i'])+logkernel;logmj=math.log(theta)-np.log(columns['c_j'])+logkernel
  li=logmi+columns['log_r']+math.log(dt);lj=logmj+columns['log_r']+math.log(dt);vi=np.exp(li);vj=np.exp(lj)
  if h>=begin:
   ki,kj=cnt[:,h];j=h-begin;logs[:,j]=ki*li+kj*lj-vi-vj-gammaln(ki+1)-gammaln(kj+1);means[:,j,0]=vi;means[:,j,1]=vj;zeros[:,j,0]=np.exp(-vi);zeros[:,j,1]=np.exp(-vj)
  y=y+(np.exp(logmi)-np.exp(logmj))*dt+K*(data['hat_y'][h]-y)
 assert np.isfinite(logs).all() and np.isfinite(means).all()
 return logs,means,zeros
# Literal scalar left-point recursion independently checks prediction and training likelihood.
def scalar_check(raw,col):
 d=e.stan_data(raw,'full');dt=d['Delta2f'];cnt=counts_for(raw);y=float(col['mu_0'][0]);s=float(col['sigma'][0]);la=float(col['lambda'][0]);ci=float(col['c_i'][0]);cj=float(col['c_j'][0]);r=math.exp(float(col['log_r'][0]));K=-math.expm1(-math.sqrt(la)*s*dt);rows=[]
 for h in range(d['N_Delta']):
  Q=s/math.sqrt(la)+s*s*(d['T_contest']-h*dt);ker=math.exp(-y*y/(2*Q))/math.sqrt(2*math.pi*Q);mi=d['theta']*ker/ci;mj=d['theta']*ker/cj;li=math.log(d['theta']/ci)-.5*math.log(2*math.pi*Q)-y*y/(2*Q)+math.log(r*dt);lj=math.log(d['theta']/cj)-.5*math.log(2*math.pi*Q)-y*y/(2*Q)+math.log(r*dt);ki,kj=cnt[:,h];rows.append(ki*li+kj*lj-r*(mi+mj)*dt-gammaln(ki+1)-gammaln(kj+1));y=y+(mi-mj)*dt+K*(d['hat_y'][h]-y)
 return np.asarray(rows)
state=json.loads((p/f'task4_{MODEL_KIND}_median_training_status_20261025.json').read_text());records=[];hourrows=[];clock=time.monotonic()
for entry in state['finished']:
 accepted=Path(entry['accepted']);a=json.loads(accepted.read_text());assert a['diagnostics']['passed'];cid=entry['contest_id'];paths=[Path(x) for x in a['chain_paths']];assert all(digest(f)==h for f,h in zip(paths,a['chain_sha256']));rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());assert digest(rawpath)==a['configuration']['input_sha256'];cut=e.early_cutoff(raw);group_logs=[];group_squares=[];group_means=[];group_zeros=[];ns=[];halves=[];chain_sizes=[];checks={}
 for chain,f in enumerate(paths):
  cols=e.read_columns(f);n=len(cols['sigma']);chain_sizes.append(n)
  if chain==0:
   one={k:v[:1] for k,v in cols.items()};log0,_,_=predictions(raw,one,0);ref=scalar_check(raw,one);checks['scalar_vector_max_logscore_difference']=float(np.max(abs(ref-log0[0])));assert checks['scalar_vector_max_logscore_difference']<1e-8
   freecols=dict(one,deadline_power=np.full(1,.5),gap_log_scale=np.zeros(1));lf,_,_=predictions(raw,freecols,0,free=True);checks['nested_kernel_max_logscore_difference']=float(np.max(abs(lf-log0)));assert checks['nested_kernel_max_logscore_difference']<1e-8
   changed=dict(raw,hat_y=list(raw['hat_y']));changed['hat_y'][cut]+=10.;lc,_,_=predictions(changed,one,cut,free=(MODEL_KIND=='C'));original_model,_,_=predictions(raw,one,0,free=(MODEL_KIND=='C'));orig=original_model[:,cut:];checks['future_readout_first_forecast_difference']=float(abs(lc[0,0]-orig[0,0]));assert checks['future_readout_first_forecast_difference']==0
  for b,idx in enumerate(np.array_split(np.arange(n),8)):
   ll,mm,zz=predictions(raw,{k:v[idx] for k,v in cols.items()},cut,free=(MODEL_KIND=='C'));group_logs.append(logsumexp(ll,axis=0)-math.log(len(idx)));group_squares.append(logsumexp(2*ll,axis=0)-math.log(len(idx)));group_means.append(mm.mean(axis=0));group_zeros.append(zz.mean(axis=0));ns.append(len(idx));halves.append(b//4)
 gl=np.asarray(group_logs);ns=np.asarray(ns);lw=np.log(ns/ns.sum());lp=logsumexp(gl+lw[:,None],axis=0);means=np.einsum('b,bhi->hi',np.exp(lw),group_means);zeros=np.einsum('b,bhi->hi',np.exp(lw),group_zeros);lsecond=logsumexp(np.asarray(group_squares)+lw[:,None],axis=0);raw_eff=ns.sum()*np.exp(2*lp-lsecond)
 influence=np.mean(np.exp(gl-lp[None,:])-1,axis=1);assert len(set(ns))==1;mcse=float(np.std(influence,ddof=1)/math.sqrt(len(influence)));halves=np.asarray(halves);halfmeans=[]
 for h in (0,1):
  choose=halves==h;ww=np.log(ns[choose]/ns[choose].sum());halfmeans.append(float(np.mean(logsumexp(gl[choose]+ww[:,None],axis=0))))
 row=dict(contest_id=cid,split='primary_hour',cutoff_hour=cut,forecast_hours=len(lp),draws=int(ns.sum()),draws_per_chain=chain_sizes,mean_joint_logscore=float(lp.mean()),mean_logscore_batch_influence_mcse=mcse,chain_stratified_half_score_difference=float(abs(halfmeans[0]-halfmeans[1])),minimum_raw_probability_weight_ess=float(raw_eff.min()),parameter_diagnostics_passed=True,checks=checks,accepted_sha256=digest(accepted),input_sha256=digest(rawpath),posterior_source='early-only, unchanged training weights; state updates after each forecast')
 row['model']=MODEL_KIND;row['mean_score_numerical_screen_passed']=bool(mcse<=.001 and row['chain_stratified_half_score_difference']<=max(.002,4*mcse));records.append(row);cnt=counts_for(raw)
 for j in range(len(lp)):hourrows.append(dict(contest_id=cid,split='primary_hour',hour_index=cut+j+1,count_i=int(cnt[0,cut+j]),count_j=int(cnt[1,cut+j]),D_joint_logscore=float(lp[j]),D_mean_i=float(means[j,0]),D_mean_j=float(means[j,1]),D_zero_i=float(zeros[j,0]),D_zero_j=float(zeros[j,1]),raw_probability_weight_ess=float(raw_eff[j])))
 print(MODEL_KIND+' scored',cid,'hours',len(lp),'mean',row['mean_joint_logscore'],'MCSE',mcse,flush=True)
result=dict(status='completed_current_accepted_subset',scope=MODEL_KIND+' primary-hour prequential count prediction with all currently accepted early draws; not a complete 73-contest ranking',records=records,scored_contests=len(records),required_contests=73,all_73_scored=len(records)==73,all_four_models_compared=False,elapsed_seconds=time.monotonic()-clock,mcse_scope='32 contiguous chain blocks, delta-method influence MCSE; raw probability-weight ESS ignores MCMC dependence and is only a concentration diagnostic')
(p/f'task4_{MODEL_KIND}_median_forecast_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if MODEL_KIND=='C':
 hourrows=[{(k.replace('D_','C_',1) if k.startswith('D_') else k):v for k,v in r.items()} for r in hourrows]
if hourrows:
 with (p/f'task4_{MODEL_KIND}_median_forecast_hours_20261025.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=hourrows[0]);w.writeheader();w.writerows(hourrows)


```
