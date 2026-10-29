# Task 3：8219更紧成本下界的同目标全局混合核查

floor=.05场景四链NUTS有明显sigma/lambda/r等量混合不足，前两个预算未通过。使用独立冻结混合提议Metropolis Hastings核查同一个后验，避免只增加相同局部轨迹。新快速似然先与该成本先验Stan在八点核对目标差仅常数；正确log参数Jacobian与新成本支持均保留。

提议从前两轮完整NUTS的log坐标聚类构造，16个局部Student-t成分加广尾成分，之后固定。每个独立跳跃使用完整q(current)/q(proposed)纠偏；20%为固定对称局部步。四条新链，2k预热+50k保留，pilot不计入抽样。Student-t密度与SciPy独立核对，目标不改变。

检查全部原参数、w、共同/相对成本、成本比及sigma模态侧指示量的rank/ESS；MH不填写虚假的HMC发散/树深/BFMI。通过前不用于政策结论；即使通过，也只说明同目标数值混合核验，不是已证明发现所有潜在模态。原NUTS继续运行，后续需比较并接续剩余场景，当前不替换原结果。

[实时状态](prior_reconstruction_20261025/cost_floor_05/contest_8219/global_mh/status.json)。

```python
import ast,csv,ctypes,json,math,hashlib,subprocess,time,os,threading,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import multivariate_t
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9')
def report_code(path):return Path(path).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
backend=report_code(p/'task4_predictive_density_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0];exec(compile(backend,'verified-density-backend','exec'))
root=p/'prior_reconstruction_20261025/cost_floor_05/contest_8219/global_mh';assert not root.exists(),'Inspect existing global same-target check before resume';root.mkdir();status=root/'status.json';data=json.loads((root.parent/'attempt2/data.json').read_text());rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob('contest_8219_*.json'));raw=json.loads(rawpath.read_text())
class FullCostTarget(ReviewTarget):
 def __init__(self,raw,data):
  self.raw=raw;self.data=e.stan_data(raw,'full');self.cut=self.data['N_Delta'];self.hour=self.cut-1;self.kind='D';self.dim=6;self.held=np.ascontiguousarray(self.data['hat_y'],dtype=float);self.counts=[]
  for k in ['i','j']:
   ts=np.asarray(self.data['hat_t_'+k]);self.counts.append(np.ascontiguousarray(np.bincount(np.maximum(1,np.ceil(ts).astype(int))-1,minlength=self.cut)[:self.cut],dtype=np.int32))
  self.cfg=data;self.mean4=np.asarray(data['prior_mean'][2:]);self.sd4=np.asarray(data['prior_sd'][2:])
 def posterior_and_prediction(self,z):
  z=np.atleast_2d(np.asarray(z,dtype=float));out=np.full(len(z),-np.inf);pred=np.full(len(z),-np.inf);floor=self.cfg['cost_floor'];support=np.isfinite(z).all(axis=1)&(z[:,:2]>=math.log(floor)).all(axis=1)&(z[:,[2,3,5]]>=math.log(.01)).all(axis=1)
  if support.any():
   zz=z[support];uv=(zz[:,0]+zz[:,1])/2;vv=zz[:,0]-zz[:,1];su=self.cfg['common_log_sd'];sv=self.cfg['relative_log_sd'];prior=-.5*((uv-self.cfg['common_log_mean'])/su)**2-.5*(vv/sv)**2-math.log(su*sv)-math.log(2*math.pi)-.5*np.sum(((zz[:,2:]-self.mean4)/self.sd4)**2,axis=1)-np.log(self.sd4).sum()-2*math.log(2*math.pi);ll,lp=self.components(zz);out[support]=prior+ll;pred[support]=lp
  return out,pred
 def posterior(self,z):return float(self.posterior_and_prediction(np.asarray(z))[0][0])
target=FullCostTarget(raw,data)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,value):
 tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(f)
pilot=[];sourcepaths=[]
for attempt in [1,2]:
 for j in range(1,5):
  f=root.parent/f'attempt{attempt}'/f'chain_{j}.csv';c=e.read_columns(f);z=np.array([c[k] for k in e.PARAMETERS]).T;z[:,:4]=np.log(z[:,:4]);pilot.append(z);sourcepaths.append(str(f.resolve()))
points=np.concatenate(pilot);parity=[];pd=root/'parity';pd.mkdir();dp=pd/'data.json';dp.write_text(json.dumps(data));exe=(p/'prior_reconstruction_20261025/model_snapshot/common_cost').resolve()
for j,index in enumerate(np.linspace(0,len(points)-1,8,dtype=int)):
 z=points[index];params=dict(zip(e.PARAMETERS,np.r_[np.exp(z[:4]),z[4:]].tolist()));pp=pd/f'params_{j}.json';pp.write_text(json.dumps(params));out=pd/f'lp_{j}.csv';run=subprocess.run([str(exe),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'],capture_output=True,text=True);(pd/f'log_{j}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;row=next(csv.DictReader(line for line in out.open() if not line.startswith('#')));parity.append(target.posterior(z)-float(row['lp__'])-float(z[:4].sum()))
assert np.ptp(parity)<1e-7;save(root/'target_parity.json',dict(passed=True,relative_density_differences=parity,range_after_constant=float(np.ptp(parity)),model_sha256=digest(exe.with_suffix('.stan')),fast_binary_sha256=digest(p/'predictive_integration_20261025/density_components.so')))
# Frozen mixture: local pilot clusters plus a broad tail component, all corrected by exact MH ratios.
rng=np.random.default_rng(202610258219);center=points.mean(0);cov=np.cov(points,rowvar=False)+np.eye(6)*1e-7;L=np.linalg.cholesky(cov);white=np.linalg.solve(L,(points-center).T).T;centers=white[rng.choice(len(white),16,replace=False)].copy()
for _ in range(30):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(16):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
components=[]
for k in range(16):
 group=points[labels==k]
 if len(group)>=30:components.append((group.mean(0),np.cov(group,rowvar=False)+np.eye(6)*1e-7))
K=len(components);weights=np.r_[np.full(K,.95/K),.05];components.append((center,4*cov));df=7.;means=np.array([x[0] for x in components]);scales=np.array([x[1]*(df-2)/df for x in components]);chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);const=gammaln((df+6)/2)-gammaln(df/2)-3*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1]
def logq(z):
 z=np.atleast_2d(z);diff=z[:,None,:]-means;dist=np.einsum('nki,kij,nkj->nk',diff,inverses,diff);return logsumexp(np.log(weights)[None,:]+const[None,:]-(df+6)/2*np.log1p(dist/df),axis=1)
for j in range(3):
 zz=points[j];reference=logsumexp([math.log(weights[k])+multivariate_t.logpdf(zz,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(float(logq(zz)[0])-reference)<1e-8
save(root/'proposal.json',dict(df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),source_chain_paths=sourcepaths,source_chain_sha256=[digest(f) for f in sourcepaths],frozen_after_pilot=True,local_move_probability=.2,local_covariance=(.01*cov).tolist()))
state=dict(status='running',driver_pid=os.getpid(),scope='Independent fixed-mixture MH for same full posterior, cost floor .05; no target modification, not NUTS',chains={},target_parity_passed=True,samples=50000,warmup=2000);lock=threading.Lock();save(status,state);clock=time.monotonic();local_chol=np.linalg.cholesky(.01*cov)
def sample(j):
 rng=np.random.default_rng(20261025+8219*100+j);total=52000;idx=rng.choice(len(weights),size=total,p=weights);normal=rng.normal(size=(total,6));chi=np.sqrt(rng.chisquare(df,size=total)/df);proposal=means[idx]+np.einsum('nij,nj->ni',chols[idx],normal)/chi[:,None];proposed_q=logq(proposal);proposed_lp=target.posterior_and_prediction(proposal)[0];current=pilot[j-1][-1].copy();cur_lp=target.posterior(current);cur_q=float(logq(current)[0]);draws=np.empty((50000,6));lpdraws=np.empty(50000);accepted=0;global_accepted=0;global_count=0
 for step in range(total):
  local=rng.uniform()<.2
  if local:
   candidate=current+local_chol@rng.normal(size=6);new_lp=target.posterior(candidate);ratio=new_lp-cur_lp;new_q=None
  else:
   candidate=proposal[step];new_lp=proposed_lp[step];new_q=float(proposed_q[step]);ratio=new_lp-cur_lp+cur_q-new_q;global_count+=1
  yes=math.log(rng.uniform())<min(0.,ratio)
  if yes:
   current=candidate.copy();cur_lp=float(new_lp);cur_q=float(logq(current)[0]) if local else new_q;accepted+=1;global_accepted+=int(not local)
  if step>=2000:draws[step-2000]=current;lpdraws[step-2000]=cur_lp
  if (step+1)%5000==0:
   with lock:state['chains'][str(j)]=dict(transitions=step+1,acceptance=accepted/(step+1),global_acceptance=global_accepted/max(global_count,1));state['elapsed_seconds']=time.monotonic()-clock;save(status,state)
 np.savez_compressed(root/f'chain_{j}.npz',z=draws,log_posterior=lpdraws);physical=draws.copy();physical[:,:4]=np.exp(physical[:,:4]);path=root/f'chain_{j}.csv'
 with path.open('w',newline='') as h:
  writer=csv.writer(h);writer.writerow(['lp__',*e.PARAMETERS]);writer.writerows(np.column_stack((lpdraws,physical)))
 return draws,path
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(sample,range(1,5)))
z=np.array([r[0] for r in result]);paths=[r[1] for r in result];arrays={name:(np.exp(z[:,:,k]) if k<4 else (np.exp(z[:,:,5]) if name=='r' else z[:,:,k])) for k,name in enumerate(['c_i','c_j','sigma','lambda','mu_0','r'])};arrays['w_i']=data['theta']/(arrays['sigma']**2*arrays['c_i']);arrays['w_j']=data['theta']/(arrays['sigma']**2*arrays['c_j']);arrays['max_w']=np.maximum(arrays['w_i'],arrays['w_j']);arrays['common_log_cost']=(z[:,:,0]+z[:,:,1])/2;arrays['relative_log_cost']=z[:,:,0]-z[:,:,1];arrays['cost_ratio']=np.exp(z[:,:,0]-z[:,:,1]);arrays['pilot_sigma_side_indicator']=(z[:,:,2]<=float(np.median(points[:,2]))).astype(float);diagnostics={};summary=dict(contest_id=8219,mode='full',scenario='cost_floor_05',divergences=None,max_treedepth=None)
for name,v in arrays.items():
 bulk,tail=nd.rank_ess(v);ess=nd.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));diagnostics[name]=dict(rank_rhat=float(nd.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);summary[name+'_median']=float(np.median(v));summary[name+'_mean']=mean;summary[name+'_sd']=float(v.std(ddof=1));summary[name+'_mean_mcse']=mcse
passed=all(np.isfinite(q['rank_rhat']) and q['rank_rhat']<=1.01 and min(q['bulk_ess'],q['tail_ess'])>=400 for q in diagnostics.values());d=dict(passed=passed,quantities=diagnostics,sampler_family='fixed_mixture_metropolis_hastings',hmc_applicable=False,saturated_transitions=None,energy_bfmi=[],primary_point_summary='Posterior median, means retained with MCSE',median_protocol_sha256=digest('pipeline/empirical/unified_prior_medians.py'));cfg=json.loads((root.parent/'configuration.json').read_text());cfg['sampler']='Independent fixed-mixture Metropolis Hastings, four chains';accepted_record=dict(summary=summary,diagnostics=d,configuration=cfg,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],attempt='global_mh_50000',target_parity=str((root/'target_parity.json').resolve()),frozen_proposal_sha256=digest(root/'proposal.json'));save(root/'diagnostics.json',accepted_record)
if passed:save(root/'accepted.json',accepted_record)
state['status']='completed' if passed else 'completed_diagnostics_failed';state['passed']=passed;state['elapsed_seconds']=time.monotonic()-clock;save(status,state);print('SAME TARGET GLOBAL MH',passed,'maxRhat',max(q['rank_rhat'] for q in diagnostics.values()),'minESS',min(min(q['bulk_ess'],q['tail_ess']) for q in diagnostics.values()),flush=True)

```
