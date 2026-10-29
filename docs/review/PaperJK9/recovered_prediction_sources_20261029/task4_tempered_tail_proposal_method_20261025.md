# Task 4：困难分子区域的退火提议探索

对局部模式与最高权重点修复后仍失败的小时，仅用退火粒子探索改善数值提议。训练后验抽样为初始粒子，逐步从p0探索到p0*L；每层重采样，并交替作对称随机游走与完整密度比修正的旧混合独立MH移动。粒子用于拟合新Student混合提议，退火粒子和其归一化常数不作为最终预测概率或参数后验。

新提议保留旧全支持混合的一半权重，再加入粒子簇；提议冻结后仍用独立新样本同时求精确训练/分子密度，沿用ESS、最大权重、分母均值、独立预算与MCSE的全部门槛，随后原Stan/Python重要区域核验才可登记。失败仍保留。没有改变训练后验或预测模型；当前计数只定义被数值积分的函数。输出独立tempered1目录。

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
reference=np.array(chains);points=np.concatenate(chains);dim=target.dim;reference_mean=points.mean(0);reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(dim)]);training_cov=np.cov(points,rowvar=False)+np.eye(dim)*1e-7;base_stage=Path(globals()['BASE_STAGE'])
if not (base_stage/'proposal.json').exists():
 alternatives=[]
 for candidate in sorted((p/'predictive_integration_20261025').glob(f'{KIND}_{CID}_hour_*/proposal.json')):
  sf=candidate.parent/'status.json'
  if not sf.exists():continue
  ss=json.loads(sf.read_text())
  if ss.get('training_cutoff')==cut and ss.get('input_sha256')==digest(rawpath) and ss.get('model_sha256')==a['configuration']['model_sha256'] and ss.get('training_prior')==a['configuration']['prior'] and ss.get('frozen_proposal_sha256')==digest(candidate):alternatives.append(candidate.parent)
 assert alternatives,'No validated proposal basis exists for the same training target';base_stage=alternatives[0]
state['proposal_basis_selected']=str(base_stage.resolve());save();oldstate=json.loads((base_stage/'status.json').read_text());assert oldstate['training_cutoff']==cut and oldstate['model_sha256']==a['configuration']['model_sha256'] and oldstate['training_prior']==a['configuration']['prior'];oldprop=json.loads((base_stage/'proposal.json').read_text());oldmeans=np.array(oldprop['means']);oldscales=np.array(oldprop['scales']);oldweights=np.array(oldprop['weights']);olddf=oldprop['df'];oldchol=np.linalg.cholesky(oldscales);oldinv=np.linalg.inv(oldscales);oldconstant=gammaln((olddf+dim)/2)-gammaln(olddf/2)-dim/2*math.log(olddf*math.pi)-.5*np.linalg.slogdet(oldscales)[1]
def old_logq(z):
 d=z[:,None,:]-oldmeans;v=np.einsum('nki,kij,nkj->nk',d,oldinv,d);return logsumexp(np.log(oldweights)[None,:]+oldconstant[None,:]-(olddf+dim)/2*np.log1p(v/olddf),axis=1)
def old_draw(n,rng):
 ids=rng.choice(len(oldweights),size=n,p=oldweights);return oldmeans[ids]+np.einsum('nij,nj->ni',oldchol[ids],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(olddf,size=n)/olddf)[:,None]
rng=np.random.default_rng(20261025+CID*1009+HOUR*37+40000000+globals().get('PILOT_RNG_OFFSET',0));nparticle=globals().get('PARTICLE_COUNT',1024);pilot_moves=globals().get('PILOT_MOVES',24);x=points[rng.choice(len(points),nparticle,replace=False)].copy();post,pr=target.posterior_and_prediction(x);assert np.isfinite(post).all() and np.isfinite(pr).all();beta=0.;pilot_records=[];state['status']='tempered_proposal_pilot';save()
for temperature in range(100):
 if beta>=1-1e-12:break
 def weight_ess(delta):
  w=np.exp(delta*pr-logsumexp(delta*pr));return float(1/(w@w))
 if weight_ess(1-beta)>=.75*nparticle:newbeta=1.
 else:
  lo=0.;hi=1-beta
  for _ in range(60):
   mid=(lo+hi)/2
   if weight_ess(mid)<.75*nparticle:hi=mid
   else:lo=mid
  newbeta=beta+lo
 assert newbeta>beta+1e-12,'Tempered proposal pilot stalled';w=np.exp((newbeta-beta)*pr-logsumexp((newbeta-beta)*pr));ids=rng.choice(nparticle,size=nparticle,p=w);x=x[ids].copy();post=post[ids].copy();pr=pr[ids].copy();beta=newbeta;cov=np.cov(x,rowvar=False)+1e-5*training_cov+np.eye(dim)*1e-9;L=np.linalg.cholesky(cov);accepted=0;jump_accepted=0;jump_proposed=0
 for move in range(pilot_moves):
  scale=(2.38/math.sqrt(dim))*rng.choice([.25,.5,1.],size=nparticle);proposal=x+(rng.normal(size=(nparticle,dim))@L.T)*scale[:,None];jump=rng.uniform(size=nparticle)<.5;proposal[jump]=old_draw(int(jump.sum()),rng);p2,r2=target.posterior_and_prediction(proposal);ratio=p2+beta*r2-post-beta*pr;ratio[jump]+=old_logq(x[jump])-old_logq(proposal[jump]);take=np.log(rng.uniform(size=nparticle))<ratio;accepted+=int(take.sum());jump_accepted+=int((take&jump).sum());jump_proposed+=int(jump.sum());x[take]=proposal[take];post[take]=p2[take];pr[take]=r2[take]
 rec=dict(temperature=temperature,beta=beta,move_acceptance=accepted/(pilot_moves*nparticle),independent_move_acceptance=jump_accepted/max(jump_proposed,1),mean_log_predictive=float(pr.mean()),maximum_log_numerator=float(np.max(post+pr)));pilot_records.append(rec);state['tempered_proposal_progress']=pilot_records;save();print('TEMPERED PROPOSAL',KIND,CID,HOUR,temperature,beta,rec['move_acceptance'],rec['mean_log_predictive'],flush=True)
assert beta>=1-1e-12;np.savez_compressed(stage/'tempered_pilot.npz',z=x,log_training=post,log_predictive=pr);(stage/'tempered_pilot.json').write_text(json.dumps(dict(records=pilot_records,not_used_as_final_integration=True,original_proposal=str((base_stage/'proposal.json').resolve()),original_proposal_sha256=digest(base_stage/'proposal.json')),indent=2));center=x.mean(0);pcov=np.cov(x,rowvar=False)+1e-6*training_cov;L=np.linalg.cholesky(pcov);white=np.linalg.solve(L,(x-center).T).T;centers=white[rng.choice(len(white),8,replace=False)].copy()
for _ in range(30):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(8):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
num=[];sizes=[]
for k in range(8):
 g=x[labels==k]
 if len(g)>=20:num.append((g.mean(0),np.cov(g,rowvar=False)+1e-3*training_cov));sizes.append(len(g))
assert num;df=7.;means=np.vstack([oldmeans,np.array([m for m,c in num])]);scales=np.concatenate([oldscales,np.array([c for m,c in num])*(df-2)/df]);weights=np.r_[.5*oldweights,.5*np.array(sizes)/sum(sizes)];
# Conditioning changes only the numerical proposal, never the target density.
for covariance_index in range(len(scales)):
 eigenvalues,eigenvectors=np.linalg.eigh((scales[covariance_index]+scales[covariance_index].T)/2);eigenvalues=np.maximum(eigenvalues,max(eigenvalues.max()*1e-6,1e-10));scales[covariance_index]=(eigenvectors*eigenvalues)@eigenvectors.T
chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1];modes=oldprop['modes'];pilot=points[np.linspace(0,len(points)-1,3,dtype=int)]
def logq(z):
 out=[]
 for start in range(0,len(z),8192):
  d=z[start:start+8192,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);out.extend(logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1).tolist())
 return np.asarray(out)
for j in range(3):
 x=pilot[j];exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(x,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(logq(x[None,:])[0]-exact)<1e-7
proposal=dict(proposal_condition_limit=1e6,proposal_regularization_does_not_change_target=True,df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),modes=modes,training_accepted_sha256=digest(ap),frozen_before_production=True,full_support_unbounded_t_components=True,numerator_centers_use_current_count_only_for_exact_probability_integration=True);(stage/'proposal.json').write_text(json.dumps(proposal,indent=2));state['status']='integrating';state['frozen_proposal_sha256']=digest(stage/'proposal.json');state['parity_v2_sha256']=digest(p/'predictive_integration_20261025/parity_v2.json');save();previous=None;passed=False
for repeat,n in enumerate(globals().get('PRODUCTION_BUDGETS',[32768,65536,131072])):
 rng=np.random.default_rng(20261025+globals().get('RNG_RUN_OFFSET',0)+CID*1009+HOUR*37+repeat*1000000);idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];q=logq(x);lp,pr=target.posterior_and_prediction(x);w0=lp-q;w1=w0+pr;Z0=float(logsumexp(w0)-math.log(n));Z1=float(logsumexp(w1)-math.log(n));a0=np.exp(w0-logsumexp(w0));a1=np.exp(w1-logsumexp(w1));prob=Z1-Z0;se=math.sqrt(float(n/(n-1)*np.sum((a1-a0)**2)));ess0=float(1/(a0@a0));ess1=float(1/(a1@a1));mu=a0@x;means_se=np.sqrt(n/(n-1)*np.sum(a0[:,None]**2*(x-mu)**2,axis=0));allowed=5*np.hypot(means_se,reference_mcse)+.005;moment_check=bool(np.all(abs(mu-reference_mean)<=allowed));finite=np.isfinite(prob) and prob<=1e-10;consistent=True if previous is None else abs(prob-previous['joint_logscore'])<=5*math.hypot(se,previous['logscore_mcse'])+.001;screen=bool(finite and min(ess0,ess1)>=1000 and max(a0.max(),a1.max())<.01 and se<=.02 and moment_check);rec=dict(repeat=repeat,samples=n,joint_logscore=prob,logscore_mcse=se,log_training_normalizer=Z0,log_numerator_normalizer=Z1,denominator_ess=ess0,numerator_ess=ess1,maximum_denominator_weight=float(a0.max()),maximum_numerator_weight=float(a1.max()),denominator_log_coordinate_means=mu.tolist(),reference_log_coordinate_means=reference_mean.tolist(),denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=allowed.tolist(),denominator_moment_check=moment_check,numerical_screen_passed=screen,consistent_with_previous=consistent);state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training=lp,log_predictive=pr,log_proposal=q);print('TAIL INTEGRAL',KIND,CID,HOUR,n,prob,se,ess0,ess1,screen,consistent,flush=True)
 if previous is not None and previous['numerical_screen_passed'] and screen and consistent:passed=True;break
 previous=rec
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;state['fixed_posterior_predictive_probability_established']=passed;state['scope']='Self-normalized importance estimate of the unchanged joint count predictive probability. Proposal depends on scoring argument, target posterior does not update.';save();print('TAIL INTEGRATION FINISHED',passed,flush=True)

```
