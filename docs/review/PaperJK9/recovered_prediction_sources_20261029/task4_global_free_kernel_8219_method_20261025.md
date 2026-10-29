# Task 4：C8219训练后验的同目标全局核查

该例自由核NUTS三个完整阶段未通过混合检查，32k阶段仍在运行。沿用D已验证的独立冻结混合MH，扩展到实际八维参数与精确C目标；提议使用本训练窗全部完整NUTS阶段，未来值与计数在额外未使用区间置零，目标只含训练似然。

Student-t归一化、协方差、抽样维度与CSV均使用八维，原六参数之外另检查a、log(b)、b。先与原C可执行文件核对八点目标密度，再四链2k预热+50k抽样。通过前不接入预测，不将MH标为HMC。

[实时状态](prediction_reconstruction_20261025/C/early/contest_8219/global_mh/status.json)。

```python
import ast,csv,ctypes,json,math,hashlib,subprocess,time,os,threading,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import multivariate_t
from pipeline.empirical import estimate as e,unified_prior_reestimate as u,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9')
def report_code(path):return Path(path).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
backend=report_code(p/'task4_predictive_density_v2_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0];exec(compile(backend,'verified-density-backend','exec'))
root=p/'prediction_reconstruction_20261025/C/early/contest_8219/global_mh';assert not root.exists(),'Inspect existing global same-target check before resume';root.mkdir();status=root/'status.json';
def complete_csv(path):
 if not path.exists():return False
 with path.open('rb') as h:h.seek(max(0,path.stat().st_size-4096));return b'#  Elapsed Time:' in h.read()
completed_folders=[f for f in sorted(root.parent.glob('median_attempt*'),key=lambda f:int(f.name.replace('median_attempt',''))) if all(complete_csv(f/('chain_'+str(j)+'.csv')) for j in range(1,5))]
assert completed_folders,'No complete four-chain pilot stage'
data=json.loads((completed_folders[0]/'data.json').read_text());rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob('contest_8219_*.json'));raw=json.loads(rawpath.read_text())
DIM=8;parameters=(*e.PARAMETERS,'deadline_power','gap_log_scale')
class EarlyPosterior(ReviewTarget):
 def __init__(self,raw,data):
  super().__init__(raw,data['N_Delta'],data['N_Delta'],'C')
  # The extra, unused evaluation interval is zero padded: no holdout values enter the posterior.
  self.held[self.cut:]=0.
  for counts in self.counts:counts[self.cut:]=0
 def posterior(self,z):return float(self.posterior_and_prediction(np.asarray(z))[0][0])
target=EarlyPosterior(raw,data)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,value):
 tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(f)
pilot=[];sourcepaths=[]
for source_folder in completed_folders:
 for j in range(1,5):
  f=source_folder/f'chain_{j}.csv';c=e.read_columns(f);z=np.array([c[k] for k in parameters]).T;z[:,:4]=np.log(z[:,:4]);pilot.append(z);sourcepaths.append(str(f.resolve()))
points=np.concatenate(pilot);parity=[];pd=root/'parity';pd.mkdir();dp=pd/'data.json';dp.write_text(json.dumps(data));exe=(p/'prediction_reconstruction_20261025/C/model_snapshot/free_kernel').resolve()
for j,index in enumerate(np.linspace(0,len(points)-1,8,dtype=int)):
 z=points[index];params=dict(zip(parameters,np.r_[np.exp(z[:4]),z[4:]].tolist()));pp=pd/f'params_{j}.json';pp.write_text(json.dumps(params));out=pd/f'lp_{j}.csv';run=subprocess.run([str(exe),'log_prob',f'constrained_params={pp}','jacobian=false','data',f'file={dp}','output',f'file={out}','sig_figs=18'],capture_output=True,text=True);(pd/f'log_{j}.txt').write_text(run.stdout+run.stderr);assert run.returncode==0;row=next(csv.DictReader(line for line in out.open() if not line.startswith('#')));parity.append(target.posterior(z)-float(row['lp__'])-float(z[:4].sum()))
assert np.ptp(parity)<1e-7;save(root/'target_parity.json',dict(passed=True,relative_density_differences=parity,range_after_constant=float(np.ptp(parity)),model_sha256=digest(exe.with_suffix('.stan')),fast_binary_sha256=digest(p/'predictive_integration_20261025/density_components_v2.so')))
# Frozen mixture: local pilot clusters plus a broad tail component, all corrected by exact MH ratios.
rng=np.random.default_rng(202610258219);center=points.mean(0);cov=np.cov(points,rowvar=False)+np.eye(DIM)*1e-7;L=np.linalg.cholesky(cov);white=np.linalg.solve(L,(points-center).T).T;centers=white[rng.choice(len(white),16,replace=False)].copy()
for _ in range(30):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(16):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
components=[]
for k in range(16):
 group=points[labels==k]
 if len(group)>=30:components.append((group.mean(0),np.cov(group,rowvar=False)+np.eye(DIM)*1e-7))
K=len(components);weights=np.r_[np.full(K,.95/K),.05];components.append((center,4*cov));df=7.;means=np.array([x[0] for x in components]);scales=np.array([x[1]*(df-2)/df for x in components]);chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);const=gammaln((df+DIM)/2)-gammaln(df/2)-DIM/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1]
def logq(z):
 z=np.atleast_2d(z);diff=z[:,None,:]-means;dist=np.einsum('nki,kij,nkj->nk',diff,inverses,diff);return logsumexp(np.log(weights)[None,:]+const[None,:]-(df+DIM)/2*np.log1p(dist/df),axis=1)
for j in range(3):
 zz=points[j];reference=logsumexp([math.log(weights[k])+multivariate_t.logpdf(zz,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(float(logq(zz)[0])-reference)<1e-8
save(root/'proposal.json',dict(df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),source_chain_paths=sourcepaths,source_chain_sha256=[digest(f) for f in sourcepaths],frozen_after_pilot=True,local_move_probability=.2,local_covariance=(.01*cov).tolist()))
state=dict(status='running',driver_pid=os.getpid(),scope='Independent fixed-mixture MH for same C early-only posterior; no target modification, not NUTS',chains={},target_parity_passed=True,samples=50000,warmup=2000);lock=threading.Lock();save(status,state);clock=time.monotonic();local_chol=np.linalg.cholesky(.01*cov)
def sample(j):
 rng=np.random.default_rng(20261025+8219*100+j+910000000);total=52000;idx=rng.choice(len(weights),size=total,p=weights);normal=rng.normal(size=(total,DIM));chi=np.sqrt(rng.chisquare(df,size=total)/df);proposal=means[idx]+np.einsum('nij,nj->ni',chols[idx],normal)/chi[:,None];proposed_q=logq(proposal);proposed_lp=target.posterior_and_prediction(proposal)[0];current=pilot[j-1][-1].copy();cur_lp=target.posterior(current);cur_q=float(logq(current)[0]);draws=np.empty((50000,DIM));lpdraws=np.empty(50000);accepted=0;global_accepted=0;global_count=0
 for step in range(total):
  local=rng.uniform()<.2
  if local:
   candidate=current+local_chol@rng.normal(size=DIM);new_lp=target.posterior(candidate);ratio=new_lp-cur_lp;new_q=None
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
  writer=csv.writer(h);writer.writerow(['lp__',*parameters]);writer.writerows(np.column_stack((lpdraws,physical)))
 return draws,path
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(sample,range(1,5)))
z=np.array([r[0] for r in result]);paths=[r[1] for r in result];arrays={name:(np.exp(z[:,:,k]) if k<4 else (np.exp(z[:,:,5]) if name=='r' else z[:,:,k])) for k,name in enumerate(['c_i','c_j','sigma','lambda','mu_0','r'])};arrays['deadline_power']=z[:,:,6];arrays['gap_log_scale']=z[:,:,7];arrays['gap_scale']=np.exp(z[:,:,7]);arrays['w_i']=data['theta']/(arrays['sigma']**2*arrays['c_i']);arrays['w_j']=data['theta']/(arrays['sigma']**2*arrays['c_j']);arrays['max_w']=np.maximum(arrays['w_i'],arrays['w_j']);arrays['common_log_cost']=(z[:,:,0]+z[:,:,1])/2;arrays['relative_log_cost']=z[:,:,0]-z[:,:,1];arrays['cost_ratio']=np.exp(z[:,:,0]-z[:,:,1]);indicator=(z[:,:,2]<=float(np.median(points[:,2]))).astype(float);diagnostics={};summary=dict(contest_id=8219,mode='early',scenario='C_primary_prediction',divergences=None,max_treedepth=None)
for name,v in arrays.items():
 bulk,tail=nd.rank_ess(v);ess=nd.effective_sample_size(v);mean=float(v.mean());mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));diagnostics[name]=dict(rank_rhat=float(nd.rank_rhat(v)),bulk_ess=float(bulk),tail_ess=float(tail),mean_mcse=mcse,relative_mean_mcse=mcse/abs(mean) if mean else None);summary[name+'_median']=float(np.median(v));summary[name+'_mean']=mean;summary[name+'_sd']=float(v.std(ddof=1));summary[name+'_mean_mcse']=mcse
passed=all(np.isfinite(q['rank_rhat']) and q['rank_rhat']<=1.01 and min(q['bulk_ess'],q['tail_ess'])>=400 for q in diagnostics.values());mode_ess=float(nd.effective_sample_size(indicator));mode_complement_ess=float(nd.effective_sample_size(1-indicator));mode_rhat=float(nd.rank_rhat(indicator));passed=bool(passed and mode_rhat<=1.01 and min(mode_ess,mode_complement_ess)>=400);d=dict(passed=passed,quantities=diagnostics,sampler_family='fixed_mixture_metropolis_hastings',hmc_applicable=False,saturated_transitions=None,energy_bfmi=[],primary_point_summary='Posterior median, means retained with MCSE',median_protocol_sha256=digest('pipeline/empirical/unified_prior_medians.py'));d['mode_occupancy_diagnostics']=dict(indicator_ess=mode_ess,complement_ess=mode_complement_ess,rank_rhat=mode_rhat,occupation_fraction=float(indicator.mean()),quantile_tail_ess_not_applicable=True);cfg=json.loads((root.parent/'configuration_median_protocol.json').read_text());cfg['sampler']='Independent fixed-mixture Metropolis Hastings, four chains';accepted_record=dict(summary=summary,diagnostics=d,configuration=cfg,chain_paths=[str(f.resolve()) for f in paths],chain_sha256=[digest(f) for f in paths],attempt='global_mh_50000',target_parity=str((root/'target_parity.json').resolve()),frozen_proposal_sha256=digest(root/'proposal.json'));save(root/'diagnostics.json',accepted_record)
if passed:save(root/'accepted.json',accepted_record)
state['status']='completed' if passed else 'completed_diagnostics_failed';state['passed']=passed;state['elapsed_seconds']=time.monotonic()-clock;save(status,state);print('SAME TARGET GLOBAL MH',passed,'maxRhat',max(q['rank_rhat'] for q in diagnostics.values()),'minESS',min(min(q['bulk_ess'],q['tail_ess']) for q in diagnostics.values()),flush=True)

```
