# Task 4：训练后验的逐时联合计数评分

仅从已经通过原诊断的 early 链积分，逐链核对 SHA256。使用完整构造期限与原 prize 单位；递推从初始化 mu_0 开始，每小时先评分，再加入当小时末公共读数。不使用私人测试读数、不用测试计数更新参数权重。

对每个抽样计算双方 Poisson 概率乘积，再以 logsumexp 积分；保留规范化阶乘。训练估计中的事件似然地板不作为预测 PMF 的裁剪，预测以未裁剪的解析 log 强度计算，避免改变概率归一化。均值、零计数概率同步积分。

独立标量递推核对向量实现；C 固定额外参数时精确退回 D；改变第一个预测小时末读数不能影响此前已发出的该小时预测。每链八个连续段给出平均分数的批次影响函数 MCSE，另比较每链前后半段；逐时 raw probability-weight ESS 只描述概率贡献集中程度，不能替代 MCMC 有效样本量。

当前输出仅覆盖批次执行时已经接受的竞赛，不可据此给出 73 竞赛模型排名；后续可在确认输入及链哈希后更新同一审查表。

```python
import csv,json,math,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from pipeline.empirical import estimate as e
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
state=json.loads((p/'task4_training_reconstruction_status_20261025.json').read_text());records=[];hourrows=[];clock=time.monotonic()
for entry in state['finished']:
 accepted=Path(entry['accepted']);a=json.loads(accepted.read_text());assert a['diagnostics']['passed'];cid=entry['contest_id'];paths=[Path(x) for x in a['chain_paths']];assert all(digest(f)==h for f,h in zip(paths,a['chain_sha256']));rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());assert digest(rawpath)==a['configuration']['input_sha256'];cut=e.early_cutoff(raw);group_logs=[];group_squares=[];group_means=[];group_zeros=[];ns=[];halves=[];chain_sizes=[];checks={}
 for chain,f in enumerate(paths):
  cols=e.read_columns(f);n=len(cols['sigma']);chain_sizes.append(n)
  if chain==0:
   one={k:v[:1] for k,v in cols.items()};log0,_,_=predictions(raw,one,0);ref=scalar_check(raw,one);checks['scalar_vector_max_logscore_difference']=float(np.max(abs(ref-log0[0])));assert checks['scalar_vector_max_logscore_difference']<1e-8
   freecols=dict(one,deadline_power=np.full(1,.5),gap_log_scale=np.zeros(1));lf,_,_=predictions(raw,freecols,0,free=True);checks['nested_kernel_max_logscore_difference']=float(np.max(abs(lf-log0)));assert checks['nested_kernel_max_logscore_difference']<1e-8
   changed=dict(raw,hat_y=list(raw['hat_y']));changed['hat_y'][cut]+=10.;lc,_,_=predictions(changed,one,cut);orig=log0[:,cut:];checks['future_readout_first_forecast_difference']=float(abs(lc[0,0]-orig[0,0]));assert checks['future_readout_first_forecast_difference']==0
  for b,idx in enumerate(np.array_split(np.arange(n),8)):
   ll,mm,zz=predictions(raw,{k:v[idx] for k,v in cols.items()},cut);group_logs.append(logsumexp(ll,axis=0)-math.log(len(idx)));group_squares.append(logsumexp(2*ll,axis=0)-math.log(len(idx)));group_means.append(mm.mean(axis=0));group_zeros.append(zz.mean(axis=0));ns.append(len(idx));halves.append(b//4)
 gl=np.asarray(group_logs);ns=np.asarray(ns);lw=np.log(ns/ns.sum());lp=logsumexp(gl+lw[:,None],axis=0);means=np.einsum('b,bhi->hi',np.exp(lw),group_means);zeros=np.einsum('b,bhi->hi',np.exp(lw),group_zeros);lsecond=logsumexp(np.asarray(group_squares)+lw[:,None],axis=0);raw_eff=ns.sum()*np.exp(2*lp-lsecond)
 influence=np.mean(np.exp(gl-lp[None,:])-1,axis=1);assert len(set(ns))==1;mcse=float(np.std(influence,ddof=1)/math.sqrt(len(influence)));halves=np.asarray(halves);halfmeans=[]
 for h in (0,1):
  choose=halves==h;ww=np.log(ns[choose]/ns[choose].sum());halfmeans.append(float(np.mean(logsumexp(gl[choose]+ww[:,None],axis=0))))
 row=dict(contest_id=cid,split='primary_hour',cutoff_hour=cut,forecast_hours=len(lp),draws=int(ns.sum()),draws_per_chain=chain_sizes,mean_joint_logscore=float(lp.mean()),mean_logscore_batch_influence_mcse=mcse,chain_stratified_half_score_difference=float(abs(halfmeans[0]-halfmeans[1])),minimum_raw_probability_weight_ess=float(raw_eff.min()),parameter_diagnostics_passed=True,checks=checks,accepted_sha256=digest(accepted),input_sha256=digest(rawpath),posterior_source='early-only, unchanged training weights; state updates after each forecast')
 records.append(row);cnt=counts_for(raw)
 for j in range(len(lp)):hourrows.append(dict(contest_id=cid,split='primary_hour',hour_index=cut+j+1,count_i=int(cnt[0,cut+j]),count_j=int(cnt[1,cut+j]),D_joint_logscore=float(lp[j]),D_mean_i=float(means[j,0]),D_mean_j=float(means[j,1]),D_zero_i=float(zeros[j,0]),D_zero_j=float(zeros[j,1]),raw_probability_weight_ess=float(raw_eff[j])))
 print('D scored',cid,'hours',len(lp),'mean',row['mean_joint_logscore'],'MCSE',mcse,flush=True)
result=dict(status='completed_current_accepted_subset',scope='D primary-hour prequential count prediction with all currently accepted early draws; not a complete 73-contest ranking',records=records,scored_contests=len(records),required_contests=73,all_73_scored=len(records)==73,all_four_models_compared=False,elapsed_seconds=time.monotonic()-clock,mcse_scope='32 contiguous chain blocks, delta-method influence MCSE; raw probability-weight ESS ignores MCMC dependence and is only a concentration diagnostic')
(p/'task4_D_accepted_forecast_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if hourrows:
 with (p/'task4_D_accepted_forecast_hours_20261025.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=hourrows[0]);w.writeheader();w.writerows(hourrows)

```
