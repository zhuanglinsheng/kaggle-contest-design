"""Functions extracted verbatim from task4_nonstrategic_baseline_method_20261025.md.

The provenance manifest records the original Git blob and content digest.
"""
import math
import numpy as np
from scipy.special import gammaln,logsumexp,roots_hermitenorm
from scipy.optimize import brentq
from scipy.integrate import quad
a0=.5
b0=.5


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
