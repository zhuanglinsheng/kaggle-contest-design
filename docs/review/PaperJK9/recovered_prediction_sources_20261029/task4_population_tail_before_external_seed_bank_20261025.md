# Task 4：已发现重要区域的群体提议探索

局部Hessian提议对少数弯曲区域覆盖不足。利用大预算实际高重要性权重点、高分子密度点和已有模式分散初始化群体；两组交替差分移动与独立MH跳跃探索当前分子密度。群体只用于拟合全支持Student混合提议，不作为参数后验或最终积分结果。

新提议保留旧混合的一半权重，协方差条件数限制1e6；冻结后用三个独立新预算的精确密度重要性积分，原ESS/权重/矩/MCSE/一致性门槛全部保留。通过后仍须原Stan及独立Python重要区域核验，再更新挑战条目。未通过保持未决。

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
reference=np.array(chains);points=np.concatenate(chains);dim=target.dim;reference_mean=points.mean(0);reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(dim)]);training_cov=np.cov(points,rowvar=False)+np.eye(dim)*1e-7;training_chol=np.linalg.cholesky(training_cov);basis=Path(globals()['BASE_STAGE']);bs=json.loads((basis/'status.json').read_text());assert bs['training_cutoff']==cut and bs['input_sha256']==digest(rawpath) and bs['model_sha256']==a['configuration']['model_sha256'] and bs['training_prior']==a['configuration']['prior'];old=json.loads((basis/'proposal.json').read_text());oldmeans=np.array(old['means']);oldscales=np.array(old['scales']);oldweights=np.array(old['weights']);df=old['df'];oldchol=np.linalg.cholesky(oldscales);oldinv=np.linalg.inv(oldscales);oldconstant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(oldscales)[1]
def old_logq(z):
 d=z[:,None,:]-oldmeans;v=np.einsum('nki,kij,nkj->nk',d,oldinv,d);return logsumexp(np.log(oldweights)[None,:]+oldconstant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1)
def old_draw(n,rng):
 k=rng.choice(len(oldweights),size=n,p=oldweights);return oldmeans[k]+np.einsum('nij,nj->ni',oldchol[k],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None]
seeds=[];seed_sources=[]
for folder in [basis,Path(globals()['LARGE_BUDGET_STAGE'])]:
 st=json.loads((folder/'status.json').read_text())
 for rec in st['records']:
  path=folder/f"production_{rec['samples']}.npz"
  if not path.exists():continue
  with np.load(path) as prod:
   w=prod['log_training']+prod['log_predictive']-prod['log_proposal'];num=prod['log_training']+prod['log_predictive'];z=prod['z'];nseed=min(16,len(w));idx=np.argpartition(w,-nseed)[-nseed:];seeds.extend(z[idx].tolist());idx=np.argpartition(num,-nseed)[-nseed:];seeds.extend(z[idx].tolist())
  seed_sources.append(dict(path=str(path.resolve()),sha256=digest(path)))
seeds.extend([r['point'] for r in old['modes']]);lp,pr=target.posterior_and_prediction(points);seeds.extend(points[np.argsort(lp+pr)[-64:]].tolist());bank=np.array(seeds);bp,br=target.posterior_and_prediction(bank);bank=bank[np.isfinite(bp+br)];assert len(bank)>=16;rng=np.random.default_rng(20261025+CID*1009+HOUR*37+245000000);walkers=256;x=bank[rng.choice(len(bank),walkers,replace=True)].copy();proposal=x+.005*(rng.normal(size=x.shape)@training_chol.T);pp,rr=target.posterior_and_prediction(proposal);good=np.isfinite(pp+rr);x[good]=proposal[good];post,pr=target.posterior_and_prediction(x);lp=post+pr;cloud=[];pilot_records=[];acceptance=0;proposals=0;state['status']='population_proposal_exploration';state['seed_sources']=seed_sources;save()
for it in range(2000):
 for group in [np.arange(0,walkers//2),np.arange(walkers//2,walkers)]:
  other=np.setdiff1d(np.arange(walkers),group);j=rng.integers(len(other),size=len(group));k=(j+1+rng.integers(len(other)-1,size=len(group)))%len(other);gamma=np.where(rng.uniform(size=len(group))<.1,1.,2.38/math.sqrt(2*dim));candidate=x[group]+gamma[:,None]*(x[other[j]]-x[other[k]])+1e-4*(rng.normal(size=(len(group),dim))@training_chol.T);p2,r2=target.posterior_and_prediction(candidate);l2=p2+r2;take=np.isfinite(l2)&(np.log(rng.uniform(size=len(group)))<l2-lp[group]);x[group[take]]=candidate[take];post[group[take]]=p2[take];pr[group[take]]=r2[take];lp[group[take]]=l2[take];acceptance+=int(take.sum());proposals+=len(group)
 if (it+1)%10==0:
  candidate=old_draw(walkers,rng);p2,r2=target.posterior_and_prediction(candidate);l2=p2+r2;ratio=l2-lp+old_logq(x)-old_logq(candidate);take=np.isfinite(l2)&(np.log(rng.uniform(size=walkers))<ratio);x[take]=candidate[take];post[take]=p2[take];pr[take]=r2[take];lp[take]=l2[take]
 if it>=1000 and (it+1)%20==0:cloud.append(x.copy())
 if (it+1)%200==0:
  pilot_records.append(dict(iteration=it+1,proposal_move_acceptance=acceptance/proposals,maximum_log_numerator=float(lp.max()),mean_log_numerator=float(lp.mean())));state['population_progress']=pilot_records;save();print('POPULATION PROPOSAL',CID,HOUR,it+1,acceptance/proposals,float(lp.max()),flush=True)
cloud=np.concatenate(cloud);np.savez_compressed(stage/'proposal_population.npz',z=cloud,final_population=x);center=cloud.mean(0);cov=np.cov(cloud,rowvar=False)+1e-6*training_cov;L=np.linalg.cholesky(cov);white=np.linalg.solve(L,(cloud-center).T).T;centers=white[rng.choice(len(white),16,replace=False)].copy()
for _ in range(40):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(16):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
num=[]
for k in range(16):
 g=cloud[labels==k]
 if len(g)>=20:num.append((g.mean(0),np.cov(g,rowvar=False)+1e-5*training_cov))
assert num;means=np.vstack([oldmeans,np.array([m for m,c in num])]);scales=np.concatenate([oldscales,np.array([c for m,c in num])*(df-2)/df]);weights=np.r_[.5*oldweights,np.full(len(num),.5/len(num))]
for j in range(len(scales)):
 val,vec=np.linalg.eigh((scales[j]+scales[j].T)/2);val=np.maximum(val,max(val.max()*1e-6,1e-10));scales[j]=(vec*val)@vec.T
chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1];modes=old['modes'];pilot=points[np.linspace(0,len(points)-1,3,dtype=int)];state['population_used_only_for_proposal_not_posterior_inference']=True;save()
def logq(z):
 out=[]
 for start in range(0,len(z),8192):
  d=z[start:start+8192,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);out.extend(logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1).tolist())
 return np.asarray(out)
for j in range(3):
 x=pilot[j];exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(x,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(logq(x[None,:])[0]-exact)<1e-7
proposal=dict(proposal_condition_limit=1e6,proposal_regularization_does_not_change_target=True,df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),modes=modes,training_accepted_sha256=digest(ap),frozen_before_production=True,full_support_unbounded_t_components=True,numerator_centers_use_current_count_only_for_exact_probability_integration=True);(stage/'proposal.json').write_text(json.dumps(proposal,indent=2));state['status']='integrating';state['frozen_proposal_sha256']=digest(stage/'proposal.json');state['parity_v2_sha256']=digest(p/'predictive_integration_20261025/parity_v2.json');save();previous=None;passed=False
for repeat,n in enumerate(globals().get('PRODUCTION_BUDGETS',[131072,262144,524288,1048576])):
 rng=np.random.default_rng(20261025+globals().get('RNG_RUN_OFFSET',0)+CID*1009+HOUR*37+repeat*1000000);idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];q=logq(x);lp,pr=target.posterior_and_prediction(x);w0=lp-q;w1=w0+pr;Z0=float(logsumexp(w0)-math.log(n));Z1=float(logsumexp(w1)-math.log(n));a0=np.exp(w0-logsumexp(w0));a1=np.exp(w1-logsumexp(w1));prob=Z1-Z0;se=math.sqrt(float(n/(n-1)*np.sum((a1-a0)**2)));ess0=float(1/(a0@a0));ess1=float(1/(a1@a1));mu=a0@x;means_se=np.sqrt(n/(n-1)*np.sum(a0[:,None]**2*(x-mu)**2,axis=0));allowed=5*np.hypot(means_se,reference_mcse)+.005;moment_check=bool(np.all(abs(mu-reference_mean)<=allowed));finite=np.isfinite(prob) and prob<=1e-10;consistent=True if previous is None else abs(prob-previous['joint_logscore'])<=5*math.hypot(se,previous['logscore_mcse'])+.001;screen=bool(finite and min(ess0,ess1)>=1000 and max(a0.max(),a1.max())<.01 and se<=.02 and moment_check);rec=dict(repeat=repeat,samples=n,joint_logscore=prob,logscore_mcse=se,log_training_normalizer=Z0,log_numerator_normalizer=Z1,denominator_ess=ess0,numerator_ess=ess1,maximum_denominator_weight=float(a0.max()),maximum_numerator_weight=float(a1.max()),denominator_log_coordinate_means=mu.tolist(),reference_log_coordinate_means=reference_mean.tolist(),denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=allowed.tolist(),denominator_moment_check=moment_check,numerical_screen_passed=screen,consistent_with_previous=consistent);state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training=lp,log_predictive=pr,log_proposal=q);print('TAIL INTEGRAL',KIND,CID,HOUR,n,prob,se,ess0,ess1,screen,consistent,flush=True)
 if len(state['records'])>=3 and all(r['numerical_screen_passed'] and r['consistent_with_previous'] for r in state['records'][-3:]):passed=True;break
 previous=rec
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;state['fixed_posterior_predictive_probability_established']=passed;state['scope']='Self-normalized importance estimate of the unchanged joint count predictive probability. Proposal depends on scoring argument, target posterior does not update.';save();print('TAIL INTEGRATION FINISHED',passed,flush=True)

```
