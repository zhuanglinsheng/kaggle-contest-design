# Task 4：复用同一训练目标的退火提议

C3960小时1344的退火提议经独立生产预算通过后，可用于同一训练后验下其他小时的概率积分。复用的只是已冻结且全支持的数值提议；每小时仍重新计算其精确训练分母和当前联合计数分子，并使用独立新预算及原有全部门槛。输入/窗口/模型/先验必须相同，其他竞赛不能复用。其他小时的旧模式只作为提议参考点，不声称是当前分子模式。失败则保留，必要时单独进行当前小时退火探索。

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
reference=np.array(chains);points=np.concatenate(chains);dim=target.dim;reference_mean=points.mean(0);reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(dim)]);source_stage=Path(globals()['BASE_PROPOSAL_STAGE']);ss=json.loads((source_stage/'status.json').read_text());assert ss['passed'] and ss['input_sha256']==digest(rawpath) and ss['training_cutoff']==cut and ss['model_sha256']==a['configuration']['model_sha256'] and ss['training_prior']==a['configuration']['prior'];sp=source_stage/'proposal.json';assert digest(sp)==ss['frozen_proposal_sha256'];prop=json.loads(sp.read_text());df=prop['df'];means=np.array(prop['means']);scales=np.array(prop['scales']);weights=np.array(prop['weights']);chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1];modes=prop['modes'];pilot=points[np.linspace(0,len(points)-1,3,dtype=int)];state['reused_frozen_proposal']=dict(source=str(sp.resolve()),sha256=digest(sp),source_probability_hour=ss['hour_onebased'],current_probability_hour=HOUR,same_training_target_checked=True);save()
def logq(z):
 out=[]
 for start in range(0,len(z),8192):
  d=z[start:start+8192,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);out.extend(logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1).tolist())
 return np.asarray(out)
for j in range(3):
 x=pilot[j];exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(x,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(logq(x[None,:])[0]-exact)<1e-7
proposal=dict(df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),modes=modes,training_accepted_sha256=digest(ap),frozen_before_production=True,full_support_unbounded_t_components=True,numerator_centers_use_current_count_only_for_exact_probability_integration=True);(stage/'proposal.json').write_text(json.dumps(proposal,indent=2));state['status']='integrating';state['frozen_proposal_sha256']=digest(stage/'proposal.json');state['parity_v2_sha256']=digest(p/'predictive_integration_20261025/parity_v2.json');save();previous=None;passed=False
for repeat,n in enumerate([32768,65536,131072]):
 rng=np.random.default_rng(20261025+globals().get('RNG_RUN_OFFSET',0)+CID*1009+HOUR*37+repeat*1000000);idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];q=logq(x);lp,pr=target.posterior_and_prediction(x);w0=lp-q;w1=w0+pr;Z0=float(logsumexp(w0)-math.log(n));Z1=float(logsumexp(w1)-math.log(n));a0=np.exp(w0-logsumexp(w0));a1=np.exp(w1-logsumexp(w1));prob=Z1-Z0;se=math.sqrt(float(n/(n-1)*np.sum((a1-a0)**2)));ess0=float(1/(a0@a0));ess1=float(1/(a1@a1));mu=a0@x;means_se=np.sqrt(n/(n-1)*np.sum(a0[:,None]**2*(x-mu)**2,axis=0));allowed=5*np.hypot(means_se,reference_mcse)+.005;moment_check=bool(np.all(abs(mu-reference_mean)<=allowed));finite=np.isfinite(prob) and prob<=1e-10;consistent=True if previous is None else abs(prob-previous['joint_logscore'])<=5*math.hypot(se,previous['logscore_mcse'])+.001;screen=bool(finite and min(ess0,ess1)>=1000 and max(a0.max(),a1.max())<.01 and se<=.02 and moment_check);rec=dict(repeat=repeat,samples=n,joint_logscore=prob,logscore_mcse=se,log_training_normalizer=Z0,log_numerator_normalizer=Z1,denominator_ess=ess0,numerator_ess=ess1,maximum_denominator_weight=float(a0.max()),maximum_numerator_weight=float(a1.max()),denominator_log_coordinate_means=mu.tolist(),reference_log_coordinate_means=reference_mean.tolist(),denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=allowed.tolist(),denominator_moment_check=moment_check,numerical_screen_passed=screen,consistent_with_previous=consistent);state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training=lp,log_predictive=pr,log_proposal=q);print('TAIL INTEGRAL',KIND,CID,HOUR,n,prob,se,ess0,ess1,screen,consistent,flush=True)
 if previous is not None and previous['numerical_screen_passed'] and screen and consistent:passed=True;break
 previous=rec
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;state['fixed_posterior_predictive_probability_established']=passed;state['scope']='Self-normalized importance estimate of the unchanged joint count predictive probability. Proposal depends on scoring argument, target posterior does not update.';save();print('TAIL INTEGRATION FINISHED',passed,flush=True)

```
