# Task 4：普通小时后验积分的独立加密

以冻结训练抽样构造16簇加宽Student混合提议，并保留全支持宽分量。仅使用训练后验拟合提议，不用测试计数选参数或更新后验。生产样本在提议冻结后独立产生，32块累计每小时训练分母和联合概率分子；一次前向递推输出全部小时，避免重复前缀。

32768/65536/131072/262144/524288/1048576独立预算；扩展批次使用独立RUN_LABEL与RNG_RUN_OFFSET保留此前未达精度记录，不改变门槛，分母ESS>=1000、最大权重<.01，训练参数均值与原链在合并误差内一致，未来非有限路径的后验加权质量<1e-8；未修正概率ESS<400或log概率<-20继续标记。平均分数MCSE<=.001、半预算一致，且连续两个独立预算通过并在5倍合并MCSE+1e-4内一致才接受。已有独立尾部积分仅在目标哈希/先验/窗口一致时沿用，其MCSE与残余IS块MCSE仍相加。

保留每预算参数、训练重要性权重、每块分母/分子/二阶矩和逐小时分数，可由冻结源码完整重放；未通过不升级为评分来源。此计算诊断不构成全局误差界。边际CDF/PMF仍保留原合格链的同后验积分；之后汇总须明确各自数值来源。默认试算C13333，只有核验实际改善后再推广。

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
import json,math,hashlib,os,time
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import multivariate_t
from pipeline.empirical import estimate as e,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9')
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
exec(compile(code(p/'task4_full_path_parity_method_20261025.md').split('# Verify complete held-out paths')[0],'full-path-target','exec'))
parity=json.loads((p/'predictive_integration_20261025/full_path_parity.json').read_text());assert parity['passed'] and digest(p/'predictive_integration_20261025/full_path_components.so')==parity['binary_sha256'];KIND=globals().get('KIND','C');CID=globals().get('CID',13333);SPLIT=globals().get('SPLIT','primary_hour');assert SPLIT in ['primary_hour','complete_day'];RUN_LABEL=globals().get('RUN_LABEL','');stage=p/'global_path_importance_20261025'/f'{KIND}_{SPLIT}_{CID}{RUN_LABEL}';assert not stage.exists();stage.mkdir(parents=True);status=stage/'status.json'
if SPLIT=='primary_hour':entry=next(r for r in json.loads((p/'task4_primary_training_index_20261025.json').read_text())['models'][KIND]['entries'] if r['contest_id']==CID)
else:entry=next(r for r in accepted_day_entries(p,KIND) if r['contest_id']==CID)
ap=Path(entry['accepted']);index=p/'task4_predictive_chain_index_20261025.json';key=KIND+':'+('complete_day:' if SPLIT=='complete_day' else '')+str(CID)
if index.exists():
 override=json.loads(index.read_text())['entries'].get(key)
 if override:ap=Path(override['accepted']);assert digest(ap)==override['accepted_sha256']
a=json.loads(ap.read_text());assert a['diagnostics']['passed'] and all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{CID}_*.json'));raw=json.loads(rawpath.read_text());assert digest(rawpath)==a['configuration']['input_sha256'];cut=a['configuration']['training_cutoff'];target=FullPathTarget(raw,cut,KIND);assert a['configuration']['prior']==u.prior(target.data);names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if KIND=='C' else e.PARAMETERS;chains=[]
for f in a['chain_paths']:
 c=e.read_columns(Path(f));z=np.array([c[k] for k in names]).T;z[:,:4]=np.log(z[:,:4]);chains.append(z)
reference=np.array(chains);points=np.concatenate(chains);reference_mean=points.mean(0);reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(target.dim)]);pilot=points[np.linspace(0,len(points)-1,min(4096,len(points)),dtype=int)];dim=target.dim;cov=np.cov(pilot,rowvar=False)+np.eye(dim)*1e-7;L=np.linalg.cholesky(cov);center=pilot.mean(0);white=np.linalg.solve(L,(pilot-center).T).T;rng=np.random.default_rng(20261025+CID+7000000+(100000 if KIND=='C' else 0));centers=white[rng.choice(len(white),16,replace=False)].copy()
for _ in range(30):
 labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
 for k in range(16):
  if np.any(labels==k):centers[k]=white[labels==k].mean(0)
base=[];sizes=[]
for k in range(16):
 g=pilot[labels==k]
 if len(g)>=20:base.append((g.mean(0),np.cov(g,rowvar=False)+np.eye(dim)*1e-6));sizes.append(len(g))
base.append((center,4*cov));weights=np.r_[.95*np.asarray(sizes)/sum(sizes),.05];means=np.array([m for m,c in base]);covs=np.array([c for m,c in base]);df=7.;scales=covs*(df-2)/df;chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1]
def logq(z):
 d=z[:,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);return logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1)
for j in range(3):
 z=pilot[j];exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(z,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(logq(z[None,:])[0]-exact)<1e-7
prop=dict(df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),training_accepted=str(ap.resolve()),training_accepted_sha256=digest(ap),frozen_before_production=True,uses_no_test_counts_to_fit_proposal=True);(stage/'proposal.json').write_text(json.dumps(prop,indent=2));H=target.data['N_Delta']-cut;tailmask=np.zeros(H,dtype=bool);tailvalues=np.zeros(H);tailse=np.zeros(H);tailentries={}
for j in range(H):
 t=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries'].get(f'{KIND}:{CID}:{SPLIT}:{cut+j+1}')
 if t is None or t.get('precision_challenged',False):continue
 assert digest(t['integral_status'])==t['integral_status_sha256'] and digest(t['rare_mode_parity'])==t['rare_mode_parity_sha256'];assert t['training_cutoff']==cut and t['input_sha256']==digest(rawpath) and t['model_sha256']==a['configuration']['model_sha256'] and t['training_prior']==a['configuration']['prior'];tailmask[j]=True;tailvalues[j]=t['joint_logscore'];tailse[j]=t['logscore_mcse'];tailentries[str(cut+j+1)]=t
state=dict(status='integrating',driver_pid=os.getpid(),model=KIND,contest_id=CID,split=SPLIT,training_cutoff=cut,input_sha256=digest(rawpath),training_accepted_sha256=digest(ap),model_sha256=a['configuration']['model_sha256'],training_prior=a['configuration']['prior'],proposal_sha256=digest(stage/'proposal.json'),backend_parity_sha256=digest(p/'predictive_integration_20261025/full_path_parity.json'),tail_integrated_hours=tailentries,records=[])
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(status)
save();previous=None;passed=False;clock=time.monotonic()
if globals().get('PREVIOUS_STAGE') is not None:
 previous_stage=Path(globals()['PREVIOUS_STAGE']);prior_state=json.loads((previous_stage/'status.json').read_text());assert all(prior_state[k]==state[k] for k in ['model','contest_id','split','training_cutoff','input_sha256','model_sha256','training_prior','tail_integrated_hours']);assert prior_state['proposal_sha256']==state['proposal_sha256'];previous=dict(prior_state['records'][-1]);assert previous['screen_passed'] and previous['consistent_with_previous'];state['previous_independent_passing_budget']=dict(stage=str(previous_stage.resolve()),status_sha256=digest(previous_stage/'status.json'),record=previous,prior_stage_itself_need_not_have_two_passing_budgets=True);save()

for repeat,n in enumerate(globals().get('PRODUCTION_BUDGETS',[32768,65536,131072,262144,524288,1048576])):
 rng=np.random.default_rng(20261025+8000000+globals().get('RNG_RUN_OFFSET',0)+CID*1009+repeat*1000000+(100000 if KIND=='C' else 0));idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];g0=[];g1=[];g2=[];lweights=[];failures=[]
 for ids in np.array_split(np.arange(n),32):
  lp,ll=target.posterior_and_path(x[ids]);w=lp-logq(x[ids]);assert not np.isnan(w).any() and not np.isnan(ll).any();lw=w[:,None]+ll;assert not np.isnan(lw).any();g0.append(float(logsumexp(w)-math.log(len(ids))));g1.append(logsumexp(lw,axis=0)-math.log(len(ids)));g2.append(logsumexp(2*lw,axis=0)-math.log(len(ids)));lweights.append(w);failures.append(~np.isfinite(ll).all(axis=1))
 w=np.concatenate(lweights);badpath=np.concatenate(failures);den=logsumexp(w)-math.log(n);normw=np.exp(w-logsumexp(w));g0=np.asarray(g0);g1=np.asarray(g1);g2=np.asarray(g2);lp=logsumexp(g1,axis=0)-math.log(32)-den;rawlp=lp.copy();rawess=n*np.exp(2*(rawlp+den)-(logsumexp(g2,axis=0)-math.log(32)));lp[tailmask]=tailvalues[tailmask];assert np.isfinite(lp).all() and (lp<=1e-8).all();mu=normw@x;meanse=np.sqrt(n/(n-1)*np.sum(normw[:,None]**2*(x-mu)**2,axis=0));tolerance=5*np.hypot(meanse,reference_mcse)+.005;momentcheck=bool(np.all(abs(mu-reference_mean)<=tolerance));influence=np.sum(np.exp(g1[:,~tailmask]-(rawlp+den)[None,~tailmask])-np.exp(g0-den)[:,None],axis=1)/H;residual=float(np.std(influence,ddof=1)/math.sqrt(32));tailmcse=float(tailse.sum()/H);mcse=residual+tailmcse;half=[]
 for ids in [np.arange(16),np.arange(16,32)]:
  hlp=logsumexp(g1[ids],axis=0)-math.log(16)-(logsumexp(g0[ids])-math.log(16));hlp[tailmask]=tailvalues[tailmask];half.append(float(hlp.mean()))
 rare=[cut+j+1 for j in range(H) if not tailmask[j] and (rawess[j]<400 or lp[j]<-20)];ess=float(1/(normw@normw));failuremass=float(normw[badpath].sum());score=float(lp.mean());halfdiff=abs(half[0]-half[1]);consistent=previous is None or abs(score-previous['mean_joint_logscore'])<=5*math.hypot(mcse,previous['mean_logscore_mcse'])+1e-4;screen=bool(ess>=1000 and normw.max()<.01 and momentcheck and failuremass<1e-8 and not rare and mcse<=.001 and halfdiff<=max(.002,4*mcse));rec=dict(repeat=repeat,samples=n,mean_joint_logscore=score,mean_logscore_mcse=mcse,residual_mcse=residual,tail_mcse=tailmcse,half_score_difference=halfdiff,denominator_ess=ess,maximum_denominator_weight=float(normw.max()),denominator_moment_check=momentcheck,denominator_log_coordinate_means=mu.tolist(),denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=tolerance.tolist(),future_nonfinite_path_posterior_mass=failuremass,unrepaired_rare_probability_hours=rare,minimum_unrepaired_probability_ess=float(rawess[~tailmask].min()) if (~tailmask).any() else None,screen_passed=screen,consistent_with_previous=bool(consistent));state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training_importance=w,block_log_denominator=g0,block_log_numerators=g1,block_log_numerator_squares=g2,joint_logscores=lp,raw_joint_logscores=rawlp,probability_weight_ess=rawess);print('GLOBAL PATH IS',KIND,CID,n,score,mcse,ess,momentcheck,len(rare),screen,consistent,flush=True)
 if previous is not None and previous['screen_passed'] and screen and consistent:passed=True;break
 previous=rec
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;state['scope']='Independent importance integration of all joint count probabilities under unchanged training posterior, with frozen full-support proposal; existing separately validated tail corrections retained. Marginal calibration distributions unchanged.';save();print('GLOBAL PATH IS ENDED',KIND,CID,passed,flush=True)

```
