# Task 4：同一目标所有小时的联合重要性积分

日期3960已接受小时的误差保守和接近整例预算。单独反复计算每小时需要重复积分整个训练前缀。这里以训练后验提议和已核验的四个分子提议构成冻结全支持混合，一次前向递推评估全部小时。测试计数只定义被积函数/数值提议，不更新训练后验。

所有小时在同一独立生产样本上积分，平均分数的MCSE由32块联合影响函数直接计算，保留共同分母及跨小时协方差。它没有与独立尾部估计相加，故不再需要原混合估计的逐小时MCSE之和。门槛并未放宽：每小时分子ESS≥1000、最大归一权重<.01、块MCSE≤.02，训练分母ESS≥1000、最大权重<.01、均值与原链一致；整例MCSE≤.001、半预算一致、未来非有限路径质量<1e-8。要求连续三个新独立预算的全部小时及整例均通过并一致。保存全路径块矩、每小时最大权重点和原目标核验所需数据；旧单小时及失败结果保留。

此为数值诊断，不是严格积分误差证书。只有完整原Stan/独立Python核验并明确登记为direct_all_hours后，评分才能采用该联合估计和协方差。不能把它与旧独立尾部均值任意拼接。

```python
import json,math,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,gammaln
from scipy.stats import multivariate_t
p=Path('docs/review/PaperJK9')
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
# Reuse the verified training target and training-only proposal preparation verbatim.
source=code(p/'task4_global_path_importance_method_20261025.md');prefix=source.split('prop=dict(df=df,weights=weights.tolist()')[0];env=dict(KIND='C',CID=3960,SPLIT='complete_day',RUN_LABEL=globals().get('RUN_LABEL','_day_joint_direct1'));exec(compile(prefix,'unchanged-training-target-and-proposal','exec'),env);globals().update({k:v for k,v in env.items() if not k.startswith('__')})
basis=[Path(v) for v in globals()['BASIS_PATHS']] if 'BASIS_PATHS' in globals() else [p/f'predictive_integration_20261025/C_3960_complete_day_hour_{h}{suffix}' for h,suffix in [(1338,'_day_population3'),(1340,'_day_tempered2'),(1344,'_day_tempered2'),(1349,'_day_tempered2')]];allmeans=[means.copy()];allscales=[scales.copy()];allweights=[.5*weights.copy()];basis_records=[]
for folder in basis:
 st=json.loads((folder/'status.json').read_text());bp=json.loads((folder/'proposal.json').read_text());assert st['passed'] and st['training_cutoff']==cut and st['input_sha256']==digest(rawpath) and st['model_sha256']==a['configuration']['model_sha256'] and st['training_prior']==a['configuration']['prior'] and bp['df']==df
 selected=np.arange(max(0,len(bp['means'])-16),len(bp['means'])) if globals().get('NUMERATOR_COMPONENTS_ONLY',False) else np.arange(len(bp['means']));bw=np.asarray(bp['weights'])[selected];bw=bw/bw.sum();allmeans.append(np.asarray(bp['means'])[selected]);allscales.append(np.asarray(bp['scales'])[selected]);allweights.append((.5/len(basis))*bw);basis_records.append(dict(stage=str(folder.resolve()),status_sha256=digest(folder/'status.json'),proposal_sha256=digest(folder/'proposal.json')))
means=np.concatenate(allmeans);scales=np.concatenate(allscales);weights=np.concatenate(allweights);assert abs(weights.sum()-1)<1e-12;chols=np.linalg.cholesky(scales);inverses=np.linalg.inv(scales);constant=gammaln((df+dim)/2)-gammaln(df/2)-dim/2*math.log(df*math.pi)-.5*np.linalg.slogdet(scales)[1]
def logq(z):
 out=[]
 for start in range(0,len(z),4096):
  d=z[start:start+4096,None,:]-means;v=np.einsum('nki,kij,nkj->nk',d,inverses,d);out.extend(logsumexp(np.log(weights)[None,:]+constant[None,:]-(df+dim)/2*np.log1p(v/df),axis=1))
 return np.asarray(out)
for z in pilot[:3]:
 exact=logsumexp([math.log(weights[k])+multivariate_t.logpdf(z,loc=means[k],shape=scales[k],df=df) for k in range(len(weights))]);assert abs(float(logq(z[None,:])[0])-exact)<1e-7
prop=dict(df=df,weights=weights.tolist(),means=means.tolist(),scales=scales.tolist(),training_accepted=str(ap.resolve()),training_accepted_sha256=digest(ap),frozen_before_production=True,uses_test_counts_only_to_fit_numerical_proposal_not_posterior=True,numerator_proposal_sources=basis_records);(stage/'proposal.json').write_text(json.dumps(prop,indent=2));H=target.data['N_Delta']-cut;status=stage/'status.json';state=dict(status='integrating',driver_pid=os.getpid(),model=KIND,contest_id=CID,split=SPLIT,training_cutoff=cut,input_sha256=digest(rawpath),training_accepted_sha256=digest(ap),model_sha256=a['configuration']['model_sha256'],training_prior=a['configuration']['prior'],proposal_sha256=digest(stage/'proposal.json'),backend_parity_sha256=digest(p/'predictive_integration_20261025/full_path_parity.json'),direct_all_hours=True,tail_integrated_hours={},records=[])
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(status)
save();clock=time.monotonic();previous=None;passed=False
for repeat,n in enumerate([262144,524288,1048576,2097152,4194304]):
 rng=np.random.default_rng(20261025+globals().get('RNG_RUN_OFFSET',375000000)+CID*1009+repeat*1000000);idx=rng.choice(len(weights),size=n,p=weights);x=means[idx]+np.einsum('nij,nj->ni',chols[idx],rng.normal(size=(n,dim)))/np.sqrt(rng.chisquare(df,size=n)/df)[:,None];g0=[];g1=[];g2=[];lweights=[];failures=[];maxlw=np.full(H,-np.inf);maxidx=np.zeros(H,dtype=int)
 for ids in np.array_split(np.arange(n),32):
  lp,ll=target.posterior_and_path(x[ids]);w=lp-logq(x[ids]);assert not np.isnan(w).any() and not np.isnan(ll).any();lw=w[:,None]+ll;assert not np.isnan(lw).any();g0.append(float(logsumexp(w)-math.log(len(ids))));g1.append(logsumexp(lw,axis=0)-math.log(len(ids)));g2.append(logsumexp(2*lw,axis=0)-math.log(len(ids)));lweights.append(w);failures.append(~np.isfinite(ll).all(axis=1));localidx=np.argmax(lw,axis=0);values=lw[localidx,np.arange(H)];better=values>maxlw;maxlw[better]=values[better];maxidx[better]=ids[localidx[better]]
 w=np.concatenate(lweights);badpath=np.concatenate(failures);den=logsumexp(w)-math.log(n);normw=np.exp(w-logsumexp(w));g0=np.asarray(g0);g1=np.asarray(g1);g2=np.asarray(g2);rawlp=logsumexp(g1,axis=0)-math.log(32)-den;ess=n*np.exp(2*(rawlp+den)-(logsumexp(g2,axis=0)-math.log(32)));maxnum=np.exp(maxlw-(rawlp+den+math.log(n)));influence=np.exp(g1-(rawlp+den)[None,:])-np.exp(g0-den)[:,None];hour_mcse=np.std(influence,axis=0,ddof=1)/math.sqrt(32);score_influence=influence.mean(axis=1);mcse=float(np.std(score_influence,ddof=1)/math.sqrt(32));half=[]
 for ids in [np.arange(16),np.arange(16,32)]:half.append(float(np.mean(logsumexp(g1[ids],axis=0)-math.log(16)-(logsumexp(g0[ids])-math.log(16)))))
 mu=normw@x;meanse=np.sqrt(n/(n-1)*np.sum(normw[:,None]**2*(x-mu)**2,axis=0));tolerance=5*np.hypot(meanse,reference_mcse)+.005;momentcheck=bool(np.all(abs(mu-reference_mean)<=tolerance));deness=float(1/(normw@normw));badmass=float(normw[badpath].sum());perhour=np.isfinite(rawlp)&(rawlp<=1e-8)&(ess>=1000)&(maxnum<.01)&(hour_mcse<=.02);consistent=True if previous is None else bool(np.all(abs(rawlp-previous['raw_joint_logscores'])<=5*np.hypot(hour_mcse,previous['hour_mcse'])+.001) and abs(float(rawlp.mean())-previous['mean_joint_logscore'])<=5*math.hypot(mcse,previous['mean_logscore_mcse'])+.0001);screen=bool(perhour.all() and deness>=1000 and normw.max()<.01 and momentcheck and badmass<1e-8 and mcse<=.001 and abs(half[0]-half[1])<=max(.002,4*mcse));rec=dict(samples=n,mean_joint_logscore=float(rawlp.mean()),mean_logscore_mcse=mcse,residual_mcse=mcse,tail_mcse=0.,half_score_difference=abs(half[0]-half[1]),denominator_ess=deness,maximum_denominator_weight=float(normw.max()),denominator_moment_check=momentcheck,future_nonfinite_path_posterior_mass=badmass,minimum_all_hour_probability_ess=float(ess.min()),maximum_all_hour_numerator_weight=float(maxnum.max()),maximum_all_hour_logscore_mcse=float(hour_mcse.max()),all_hour_precision_passed=bool(perhour.all()),unrepaired_rare_probability_hours=[int(cut+j+1) for j in np.where(~perhour)[0]],screen_passed=screen,consistent_with_previous=consistent);state['records'].append(rec);state['elapsed_seconds']=time.monotonic()-clock;save();np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training_importance=w,block_log_denominator=g0,block_log_numerators=g1,block_log_second_numerators=g2,raw_joint_logscores=rawlp,probability_weight_ess=ess,maximum_numerator_weights=maxnum,hour_logscore_mcse=hour_mcse,numerator_representative_points=x[np.unique(maxidx)]);print('JOINT DIRECT',n,rec,flush=True)
 if len(state['records'])>=3 and all(r['screen_passed'] and r['consistent_with_previous'] for r in state['records'][-3:]):passed=True;break
 previous=dict(raw_joint_logscores=rawlp.copy(),hour_mcse=hour_mcse.copy(),mean_joint_logscore=float(rawlp.mean()),mean_logscore_mcse=mcse)
state['status']='completed' if passed else 'completed_with_uncertainty';state['passed']=passed;save();print('JOINT DIRECT FINISHED',passed,flush=True)
```
