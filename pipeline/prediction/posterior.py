"""Functions extracted verbatim from task4_calibrated_posterior_prediction_method_20261025.md.

The provenance manifest records the original Git blob and content digest.
"""
import math
import numpy as np
from scipy.special import gammaln,logsumexp,gammaincc
from pipeline.empirical import estimate as e


def counts_for(raw):
 N=raw['N_Delta'];out=[]
 for k in ('i','j'):
  t=np.asarray([float(v) for v in raw['hat_t_'+k] if 0<float(v)<=N]);out.append(np.bincount(np.maximum(1,np.ceil(t).astype(int))-1,minlength=N)[:N])
 return np.asarray(out)

def predictions(raw,columns,begin,end=None,free=False):
 data=e.stan_data(raw,'full');N=data['N_Delta'] if end is None else end;dt=data['Delta2f'];T=data['T_contest'];theta=data['theta'];cnt=counts_for(raw);n=len(columns['sigma']);y=columns['mu_0'].copy();sigma=columns['sigma'];lam=columns['lambda'];barS=sigma/np.sqrt(lam);K=-np.expm1(-np.sqrt(lam)*sigma*dt);Qref=barS+sigma*sigma*T
 logs=np.empty((n,N-begin));means=np.empty((n,N-begin,2));zeros=np.empty_like(means);below=np.empty_like(means);pmf=np.empty_like(means)
 for h in range(N):
  Q=barS+sigma*sigma*(T-h*dt)
  if free:logkernel=-.5*np.log(2*np.pi*Qref)-columns['deadline_power']*np.log(Q/Qref)-np.exp(columns['gap_log_scale'])*y*y/(2*Q)
  else:logkernel=-.5*np.log(2*np.pi*Q)-y*y/(2*Q)
  logmi=math.log(theta)-np.log(columns['c_i'])+logkernel;logmj=math.log(theta)-np.log(columns['c_j'])+logkernel
  li=logmi+columns['log_r']+math.log(dt);lj=logmj+columns['log_r']+math.log(dt);vi=np.exp(li);vj=np.exp(lj)
  if h>=begin:
   ki,kj=cnt[:,h];j=h-begin;logs[:,j]=ki*li+kj*lj-vi-vj-gammaln(ki+1)-gammaln(kj+1);means[:,j,0]=vi;means[:,j,1]=vj;zeros[:,j,0]=np.exp(-vi);zeros[:,j,1]=np.exp(-vj);below[:,j,0]=0. if ki==0 else gammaincc(ki,vi);below[:,j,1]=0. if kj==0 else gammaincc(kj,vj);pmf[:,j,0]=np.exp(ki*li-vi-gammaln(ki+1));pmf[:,j,1]=np.exp(kj*lj-vj-gammaln(kj+1))
  y=y+(np.exp(logmi)-np.exp(logmj))*dt+K*(data['hat_y'][h]-y)
 assert np.isfinite(logs).all() and np.isfinite(means).all()
 return logs,means,zeros,below,pmf

def scalar_check(raw,col):
 d=e.stan_data(raw,'full');dt=d['Delta2f'];cnt=counts_for(raw);y=float(col['mu_0'][0]);s=float(col['sigma'][0]);la=float(col['lambda'][0]);ci=float(col['c_i'][0]);cj=float(col['c_j'][0]);r=math.exp(float(col['log_r'][0]));K=-math.expm1(-math.sqrt(la)*s*dt);rows=[]
 for h in range(d['N_Delta']):
  Q=s/math.sqrt(la)+s*s*(d['T_contest']-h*dt);ker=math.exp(-y*y/(2*Q))/math.sqrt(2*math.pi*Q);mi=d['theta']*ker/ci;mj=d['theta']*ker/cj;li=math.log(d['theta']/ci)-.5*math.log(2*math.pi*Q)-y*y/(2*Q)+math.log(r*dt);lj=math.log(d['theta']/cj)-.5*math.log(2*math.pi*Q)-y*y/(2*Q)+math.log(r*dt);ki,kj=cnt[:,h];rows.append(ki*li+kj*lj-r*(mi+mj)*dt-gammaln(ki+1)-gammaln(kj+1));y=y+(mi-mj)*dt+K*(d['hat_y'][h]-y)
 return np.asarray(rows)
