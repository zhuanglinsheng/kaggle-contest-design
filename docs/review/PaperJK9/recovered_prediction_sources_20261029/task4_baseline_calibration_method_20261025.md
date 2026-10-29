# Task 4：两个基准的离散概率校准输入

沿用已通过评分数值核验的全部73例×两种切分的训练后验与固定先验，不重新拟合或调参。Gamma-Poisson 边际为负二项分布；F(k-1)=I_p(a,k)（k=0时为0），p=B/(B+exposure)。B共享beta时对同一beta后验积分。

额外将beta求积阶数加倍，CDF和观测PMF最大差需小于1e-7；零计数PMF与原评分表的零概率交叉核对。输出仅供同一随机化规则的PIT与零事件校准，不把边际概率相乘代替已正确计算的联合得分。

```python
import csv,json,math,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp,roots_hermitenorm
from scipy.optimize import brentq
from scipy.integrate import quad
from pipeline.empirical.estimate import early_cutoff
report=Path('docs/review/PaperJK9');audit=json.loads((report/'task4_source_information_audit_20261025.json').read_text());assert audit['status']=='completed' and audit['replay_passed_contests']==73 and audit['initial_sources_before_cutoff_contests']==73
out=report/'task4_baseline_comparison_20261025.json'

a0=.5;b0=.5

def log_den_and_moments(beta,z,dt,extra_z=None):
    xx=beta*z+math.log(dt)
    if extra_z is not None:xx=np.r_[xx,beta*extra_z+math.log(dt)];zz=np.r_[z,extra_z]
    else:zz=z
    den=float(np.logaddexp(math.log(b0),logsumexp(xx)));w=np.exp(xx-den);m=float(w@zz);v=float(w@(zz*zz)-m*m)
    return den,m,max(0.,v)
def mode_shape(A,Z,z,dt,extra_z=None):
    derivative=lambda beta:-beta+Z-A*log_den_and_moments(beta,z,dt,extra_z)[1]
    bound=16.
    while derivative(-bound)*derivative(bound)>0:
        bound*=2
        if bound>128:raise RuntimeError('Could not bracket posterior mode')
    mode=float(brentq(derivative,-bound,bound,xtol=1e-11));den,m,var=log_den_and_moments(mode,z,dt,extra_z)
    return mode,1/math.sqrt(1+A*var)
def log_integral(A,Z,z,dt,extra_z=None):
    mode,sd=mode_shape(A,Z,z,dt,extra_z)
    logp=lambda beta:-.5*beta*beta+beta*Z-A*log_den_and_moments(beta,z,dt,extra_z)[0]
    peak=logp(mode)
    def integrand(u):
        v=logp(mode+sd*u)-peak
        return math.exp(v) if v>-745 else 0.
    left,el=quad(integrand,-np.inf,0,epsabs=1e-11,epsrel=1e-10,limit=120)
    right,er=quad(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-10,limit=120)
    return peak+math.log(sd*(left+right)),float((el+er)/(left+right))
def quadrature_posterior(A,Z,z,dt,order):
    mode,sd=mode_shape(A,Z,z,dt);proposal_sd=2*sd;nodes,weights=roots_hermitenorm(order);ok=weights>0;nodes=nodes[ok];weights=weights[ok];beta=mode+proposal_sd*nodes
    den=np.logaddexp(math.log(b0),logsumexp(math.log(dt)+beta[:,None]*z[None,:],axis=1))
    lw=np.log(weights)-.5*beta*beta+beta*Z-A*den+.5*nodes*nodes
    normal=float(logsumexp(lw));lw-=normal
    return beta,den,lw,dict(mode=mode,proposal_sd=proposal_sd,mean=float(np.exp(lw)@beta),log_integral=normal+math.log(proposal_sd),order=order)
def forecast(beta,den,lw,ai,aj,zfuture,counts,dt):
    scores=[];means=[];zeros=[]
    for start in range(0,len(zfuture),128):
        z=zfuture[start:start+128];k=counts[:,start:start+128];v=math.log(dt)+beta[:,None]*z[None,:]-den[:,None];C=gammaln(ai+k[0])-gammaln(ai)-gammaln(k[0]+1)+gammaln(aj+k[1])-gammaln(aj)-gammaln(k[1]+1)
        ll=C[None,:]-(ai+aj)*np.logaddexp(0,v)-(k[0]+k[1])[None,:]*np.logaddexp(0,-v)
        scores.extend(logsumexp(lw[:,None]+ll,axis=0).tolist());meanrate=np.exp(logsumexp(lw[:,None]+v,axis=0));means.extend(np.column_stack((ai*meanrate,aj*meanrate)).tolist());zeros.extend(np.column_stack((np.exp(logsumexp(lw[:,None]-ai*np.logaddexp(0,v),axis=0)),np.exp(logsumexp(lw[:,None]-aj*np.logaddexp(0,v),axis=0)))).tolist())
    return np.array(scores),np.asarray(means),np.asarray(zeros)

from scipy.special import betainc
p=report;previous=json.loads((p/'task4_baseline_refined_comparison_20261025.json').read_text());assert previous['all_numerical_checks_passed'];data_rows=list(csv.DictReader((p/'task4_baseline_refined_predictions_20261025.csv').open()));buckets={}
for row in data_rows:buckets.setdefault((int(row['contest_id']),row['split']),[]).append(row)
records=[];newrows=[]
def marginal(beta,den,lw,a,z,k,dt):
 below=[];pmfs=[]
 for start in range(0,len(z),128):
  zz=z[start:start+128];kk=k[start:start+128];v=math.log(dt)+beta[:,None]*zz[None,:]-den[:,None];lp=-np.logaddexp(0,v);lq=-np.logaddexp(0,-v);pr=np.exp(lp);bb=np.zeros_like(pr);has=kk>0
  if has.any():bb[:,has]=betainc(a,kk[has][None,:],pr[:,has])
  pp=np.exp(gammaln(a+kk)[None,:]-gammaln(a)-gammaln(kk+1)[None,:]+a*lp+kk[None,:]*lq);below.extend((np.exp(lw)@bb).tolist());pmfs.extend((np.exp(lw)@pp).tolist())
 return np.asarray(below),np.asarray(pmfs)
for r in previous['records']:
 cid=r['contest_id'];raw=json.loads(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json')).read_text());N=raw['N_Delta'];dt=raw['Delta2f'];start=raw['count_start_hour'];cut=r['cutoff_hour'];nt=np.array([r['training_count_i'],r['training_count_j']]);shapes=nt+a0;counts=[]
 for who in ('i','j'):
  ts=np.asarray([float(t) for t in raw['hat_t_'+who] if 0<float(t)<=N]);counts.append(np.bincount(np.maximum(1,np.ceil(ts).astype(int))-1,minlength=N)[:N])
 counts=np.asarray(counts);train=counts[:,start:cut];test=counts[:,cut:];zt=np.log((N-np.arange(start,cut))/N);zf=np.log((N-np.arange(cut,N))/N);A=float(shapes.sum());Z=float(train.sum(axis=0)@zt);cc={};checks=[]
 for label in ['A','B']:
  if label=='A':b=np.array([0.]);d=np.array([math.log(b0+(cut-start)*dt)]);w=np.array([0.])
  else:b,d,w,_=quadrature_posterior(A,Z,zt,dt,r['quadrature_order'])
  vals=[marginal(b,d,w,float(shapes[k]),zf,test[k],dt) for k in [0,1]]
  if label=='B':
   b2,d2,w2,_=quadrature_posterior(A,Z,zt,dt,2*r['quadrature_order']);fine=[marginal(b2,d2,w2,float(shapes[k]),zf,test[k],dt) for k in [0,1]];err=max(float(np.max(abs(x-y))) for pair,pair2 in zip(vals,fine) for x,y in zip(pair,pair2));assert err<1e-7;checks.append(err);vals=fine
  cc[label]=vals
 oldrows=sorted(buckets[(cid,r['split'])],key=lambda h:int(h['hour_index']));assert len(oldrows)==len(zf);zero_error=0.
 for j,h in enumerate(oldrows):
  outrow=dict(h)
  for label in ['A','B']:
   for k,who in enumerate(['i','j']):
    bb,pp=float(cc[label][k][0][j]),float(cc[label][k][1][j]);assert bb>=0 and pp>=0 and bb+pp<=1+1e-10;outrow[f'{label}_cdf_below_{who}']=bb;outrow[f'{label}_observed_pmf_{who}']=pp
    if test[k,j]==0:zero_error=max(zero_error,abs(pp-float(h[f'{label}_zero_{who}'])))
  newrows.append(outrow)
 assert zero_error<1e-7;records.append(dict(contest_id=cid,split=r['split'],max_CDF_PMF_quadrature_difference=max(checks),zero_probability_crosscheck_error=zero_error))
 if len(records)%20==0:print('Baseline CDF folds',len(records),'/146',flush=True)
with (p/'task4_baseline_calibrated_predictions_20261025.csv').open('w',newline='') as h:
 writer=csv.DictWriter(h,fieldnames=newrows[0]);writer.writeheader();writer.writerows(newrows)
(p/'task4_baseline_calibration_input_check_20261025.json').write_text(json.dumps(dict(status='completed',folds=len(records),pair_hours=len(newrows),all_numerical_checks_passed=True,records=records,scope='Normalized marginal CDF/PMF for subsequent randomized PIT, not an extra fitted model'),indent=2))

```
