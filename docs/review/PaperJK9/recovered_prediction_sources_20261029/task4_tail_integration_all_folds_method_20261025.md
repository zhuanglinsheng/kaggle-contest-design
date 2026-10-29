# Task 4：按实际训练目标补足尾部概率

推广已通过核对的固定后验重要性积分，支持主切分及完整日期切分，按最新接受索引合并主训练、接续和同目标补充链。日期切分使用独立目标和积分目录，不复用不同窗口积分。记录接受链、输入、训练截止、模型、先验和全部密度哈希。数值提议冻结后方取生产样本，门槛与既有两小时核验一致；通过仍须在其重要区域与原Stan和独立Python核对后才进入统一评分索引。没有改变模型、后验或测试期参数权重。失败后的独立修复可用RUN_LABEL保留原批次，以旧生产样本最大分子权重点作为EXTRA_START_POINTS探索遗漏模式；提议再次冻结后使用新的RNG_RUN_OFFSET生产，门槛不变。

```python
def accepted_day_entries(root,model):
 import json,hashlib
 from pathlib import Path
 root=Path(root);entries={}
 for name in [f'task4_{model}_complete_day_training_status_20261025.json',f'task4_{model}_complete_day_resume_status_20261025.json']:
  sf=root/name
  if sf.exists():
   for row in json.loads(sf.read_text())['finished']:entries[row['contest_id']]=row
 sf=root/'task4_complete_day_supplemental_acceptances_20261025.json'
 if sf.exists():
  for row in json.loads(sf.read_text())['entries']:
   if row['model']==model:
    assert hashlib.sha256(Path(row['accepted']).read_bytes()).hexdigest()==row['accepted_sha256'];entries[row['contest_id']]=row
 return [entries[c] for c in sorted(entries)]
import json,math,hashlib,time,os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln,logsumexp
from scipy.stats import multivariate_t
from pipeline.empirical import estimate as e,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9')
def code(path):return Path(path).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
backend=code(p/'task4_predictive_density_v2_parity_method_20261025.md').split('# Load the exact reported Python prediction function')[0];exec(compile(backend,'verified-density-v2','exec'))
KIND=globals().get('KIND','D');CID=globals().get('CID',3288);HOUR=globals().get('HOUR',760);SPLIT=globals().get('SPLIT','primary_hour');assert SPLIT in ['primary_hour','complete_day'];stage=p/'predictive_integration_20261025'/(f'{KIND}_{CID}_hour_{HOUR}' if SPLIT=='primary_hour' else f'{KIND}_{CID}_complete_day_hour_{HOUR}');stage=Path(str(stage)+globals().get('RUN_LABEL',''));assert not stage.exists(),'Inspect existing tail integration before resume';stage.mkdir();status=stage/'status.json';state=dict(status='preparing',driver_pid=os.getpid(),model=KIND,contest_id=CID,hour_onebased=HOUR,records=[],target_unchanged=True,uses_outcome_only_to_evaluate_fixed_predictive_probability=True)
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(status)
save();clock=time.monotonic()
if SPLIT=='primary_hour':
 entries={}
 for name in [f'task4_{KIND}_median_training_status_20261025.json',f'task4_{KIND}_primary_resume_status_20261025.json']:
  sf=p/name
  if sf.exists():
   for r in json.loads(sf.read_text())['finished']:entries[r['contest_id']]=r
 for r in json.loads((p/'task4_primary_supplemental_acceptances_20261025.json').read_text())['entries']:
  if r['model']==KIND:assert digest(r['accepted'])==r['accepted_sha256'];entries[r['contest_id']]=r
 entry=entries[CID]
else:entry=next(r for r in accepted_day_entries(p,KIND) if r['contest_id']==CID)
ap=Path(entry['accepted']);ix=p/'task4_predictive_chain_index_20261025.json'
if ix.exists():
 override=json.loads(ix.read_text())['entries'].get(KIND+':'+('complete_day:' if SPLIT=='complete_day' else '')+str(CID))
 if override:ap=Path(override['accepted']);assert digest(ap)==override['accepted_sha256']
a=json.loads(ap.read_text());assert a['diagnostics']['passed'];assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{CID}_*.json'));raw=json.loads(rawpath.read_text());assert digest(rawpath)==a['configuration']['input_sha256'];cut=a['configuration']['training_cutoff'];target=ReviewTarget(raw,cut,HOUR-1,KIND);assert a['configuration']['prior']==u.prior(target.data);state.update(split=SPLIT,training_cutoff=cut,training_accepted=str(ap.resolve()),training_accepted_sha256=digest(ap),input_sha256=digest(rawpath),model_sha256=a['configuration']['model_sha256'],training_prior=a['configuration']['prior']);save();names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if KIND=='C' else e.PARAMETERS;chains=[]
for f in a['chain_paths']:
 c=e.read_columns(Path(f));z=np.array([c[k] for k in names]).T;z[:,:4]=np.log(z[:,:4]);chains.append(z)
reference=np.array(chains);points=np.concatenate(chains);reference_mean=points.mean(0);reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(target.dim)]);pilot=points[np.linspace(0,len(points)-1,min(4096,len(points)),dtype=int)];dim=target.dim;cov=np.cov(pilot,rowvar=False)+np.eye(dim)*1e-7;L=np.linalg.cholesky(cov);center=pilot.mean(0);white=np.linalg.solve(L,(pilot-center).T).T;rng=np.random.default_rng(20261025+CID+HOUR);centers=white[rng.choice(len(white),16,replace=False)].copy()
for _ in range(20):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(16):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
base=[]
for k in range(16):
 g=pilot[labels==k]
 if len(g)>=20:base.append((g.mean(0),np.cov(g,rowvar=False)+np.eye(dim)*1e-6))
base.append((center,4*cov));lp0,lpf=target.posterior_and_prediction(pilot);order=np.argsort(lp0+lpf)[-4:];starts=[pilot[i] for i in order]+[np.median(pilot,axis=0),target.mean.copy()]
for factor in [2.,4.]:
 v=np.median(pilot,axis=0);v[2]+=math.log(factor);v[3]-=math.log(factor);starts.append(v)
lower=math.log(.01)+1e-7;bounds=[(lower,None) if j in [0,1,2,3,5] else (None,None) for j in range(dim)]
def objective(z):
 lp,pr=target.posterior_and_prediction(z);value=float(lp[0]+pr[0]);return -value if np.isfinite(value) else 1e100
starts.extend([np.asarray(v,dtype=float) for v in globals().get('EXTRA_START_POINTS',[])]);modes=[]
for j,start in enumerate(starts):
 start=np.asarray(start).copy();start[[0,1,2,3,5]]=np.maximum(start[[0,1,2,3,5]],lower+1e-4);r=minimize(objective,start,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=600,ftol=1e-11,gtol=1e-6,maxls=40));modes.append(dict(point=r.x.tolist(),negative_log_numerator=float(r.fun),optimizer_converged=bool(r.success),message=str(r.message)));print('TAIL MODE',KIND,CID,HOUR,j,float(r.fun),bool(r.success),flush=True)
best=min(m['negative_log_numerator'] for m in modes);chosen=[]
for m in sorted(modes,key=lambda m:m['negative_log_numerator']):
 x=np.array(m['point'])
 if m['negative_log_numerator']>best+30:continue
 if any(np.linalg.norm(np.linalg.solve(L,x-v))<.05 for v in chosen):continue
 chosen.append(x)
 if len(chosen)==globals().get('MAX_NUMERATOR_MODES',4):break
assert chosen
num=[]
for x in chosen:
 h=1e-3*np.maximum(1,abs(x))
 for j in [0,1,2,3,5]:h[j]=min(h[j],max(1e-7,(x[j]-lower)*.2))
 H=np.empty((dim,dim));f0=objective(x)
 for i in range(dim):
  ei=np.eye(dim)[i]*h[i];H[i,i]=(objective(x+ei)-2*f0+objective(x-ei))/h[i]**2
  for j in range(i):
   ej=np.eye(dim)[j]*h[j];H[i,j]=H[j,i]=(objective(x+ei+ej)-objective(x+ei-ej)-objective(x-ei+ej)+objective(x-ei-ej))/(4*h[i]*h[j])
 if not np.isfinite(H).all():C=cov.copy()
 else:
  val,vec=np.linalg.eigh((H+H.T)/2);C=(vec/np.clip(val,1e-3,1e8))@vec.T+np.eye(dim)*1e-7
 for scale in [.5,1.,2.]:num.append((x,scale*C))
means=np.array([m for m,c in base+num]);covs=np.array([c for m,c in base+num]);weights=np.r_[np.full(len(base),.5/len(base)),np.full(len(num),.5/len(num))];df=7.;scales=covs*(df-2)/df;
# Conditioning changes only the numerical proposal, never the target density.
for covariance_index in range(len(scales)):
 eigenvalues,eigenvectors=np.linalg.eigh((scales[covariance_index]+scales[covariance_index].T)/2);eigenvalues=np.maximum(eigenvalues,max(eigenvalues.max()*1e-6,1e-10));scales[covariance_index]=(eigenvectors*eigenvalues)@eigenvectors.T
chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1]
def logq(z):
 out=[]
 for start in range(0,len(z),8192):
  d=z[start:start+8192,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);out.extend(logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1).tolist())
 return np.asarray(out)
for j in range(3):
 x=pilot[j];exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(x,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(logq(x[None,:])[0]-exact)<1e-7
proposal=dict(proposal_condition_limit=1e6,proposal_regularization_does_not_change_target=True,df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),modes=modes,training_accepted_sha256=digest(ap),frozen_before_production=True,full_support_unbounded_t_components=True,numerator_centers_use_current_count_only_for_exact_probability_integration=True);(stage/'proposal.json').write_text(json.dumps(proposal,indent=2));state['status']='integrating';state['frozen_proposal_sha256']=digest(stage/'proposal.json');state['parity_v2_sha256']=digest(p/'predictive_integration_20261025/parity_v2.json');save();previous=None;passed=False
for repeat,n in enumerate([32768,65536,131072]):
 rng=np.random.default_rng(20261025+globals().get('RNG_RUN_OFFSET',0)+CID*1009+HOUR*37+repeat*1000000);idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];q=logq(x);lp,pr=target.posterior_and_prediction(x);w0=lp-q;w1=w0+pr;Z0=float(logsumexp(w0)-math.log(n));Z1=float(logsumexp(w1)-math.log(n));a0=np.exp(w0-logsumexp(w0));a1=np.exp(w1-logsumexp(w1));prob=Z1-Z0;se=math.sqrt(float(n/(n-1)*np.sum((a1-a0)**2)));ess0=float(1/(a0@a0));ess1=float(1/(a1@a1));mu=a0@x;means_se=np.sqrt(n/(n-1)*np.sum(a0[:,None]**2*(x-mu)**2,axis=0));allowed=5*np.hypot(means_se,reference_mcse)+.005;moment_check=bool(np.all(abs(mu-reference_mean)<=allowed));finite=np.isfinite(prob) and prob<=1e-10;consistent=True if previous is None else abs(prob-previous['joint_logscore'])<=5*math.hypot(se,previous['logscore_mcse'])+.001;screen=bool(finite and min(ess0,ess1)>=1000 and max(a0.max(),a1.max())<.01 and se<=.02 and moment_check);rec=dict(repeat=repeat,samples=n,joint_logscore=prob,logscore_mcse=se,log_training_normalizer=Z0,log_numerator_normalizer=Z1,denominator_ess=ess0,numerator_ess=ess1,maximum_denominator_weight=float(a0.max()),maximum_numerator_weight=float(a1.max()),denominator_log_coordinate_means=mu.tolist(),reference_log_coordinate_means=reference_mean.tolist(),denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=allowed.tolist(),denominator_moment_check=moment_check,numerical_screen_passed=screen,consistent_with_previous=consistent);state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training=lp,log_predictive=pr,log_proposal=q);print('TAIL INTEGRAL',KIND,CID,HOUR,n,prob,se,ess0,ess1,screen,consistent,flush=True)
 if previous is not None and previous['numerical_screen_passed'] and screen and consistent:passed=True;break
 previous=rec
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;state['fixed_posterior_predictive_probability_established']=passed;state['scope']='Self-normalized importance estimate of the unchanged joint count predictive probability. Proposal depends on scoring argument, target posterior does not update.';save();print('TAIL INTEGRATION FINISHED',passed,flush=True)

```
