"""Bounded joint-design robustness to the organizer's output valuation."""
import argparse
import json

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from archive.experiments.exact_joint_design import exact_B, beta
from pipeline.config import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--polish',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results/theory/precision_conditions/joint_preferences.json'
    if args.polish:
        saved=json.loads(out.read_text())
        for case in saved['results']:
            A=case['A']
            if not case['finite']['success']:
                candidate=case['finite']
                def loss(x):
                    theta,T,loglam=x
                    try:
                        b=exact_B(theta,T,float(np.exp(loglam)),2,dx=.04,steps=500)[0]
                    except (RuntimeError,FloatingPointError):
                        return 1e6
                    return theta+.175*T*T-A*np.log(b)
                fit=minimize(loss,[candidate['theta'],candidate['T'],np.log(candidate['lambda_'])],
                    method='Powell',bounds=[(.1,2.5),(.1,4),(np.log(.01),np.log(1e4))],
                    options=dict(maxiter=12,xtol=1e-4,ftol=1e-8))
                theta,T,loglam=fit.x
                case['powell']=dict(theta=float(theta),T=float(T),lambda_=float(np.exp(loglam)),
                    score=float(-fit.fun),success=bool(fit.success),message=str(fit.message))
                print(json.dumps(case['powell']),flush=True)
        saved['polished_qualification']='Powell results are additional local searches; original line-search diagnostics retained.'
        out.write_text(json.dumps(saved,indent=2)+'\n')
        return
    results=[]
    cache={}
    failures=[]
    for A in [5.5,5.8,6.2,6.5]:
        zero=minimize_scalar(lambda t: .175*t*t-A*np.log(beta(2,1+t)),bounds=(0,4),method='bounded')
        case=dict(A=A,zero=dict(theta=0,T=float(zero.x),score=float(-zero.fun)))
        for label,precision,starts in [
            ('finite',None,[[2.0133,2.8602,np.log(.4655)],[1.4,2.6,np.log(30)]]),
            ('high',1e5,[[1.4,2.64],[.5,1.5]])]:
            def loss(x):
                theta,T=x[:2]
                ell=float(np.exp(x[2])) if precision is None else precision
                key=tuple(np.round([theta,T,ell],11))
                if key not in cache:
                    try:
                        cache[key]=exact_B(theta,T,ell,2,dx=.04,steps=500)[0]
                    except (RuntimeError,FloatingPointError) as error:
                        failures.append(dict(theta=theta,T=T,lambda_=ell,error=str(error)))
                        cache[key]=np.nan
                b=cache[key]
                return theta+.175*T*T-A*np.log(b) if np.isfinite(b) else 1e6
            bounds=[(0,2.5),(.1,4)]
            if precision is None: bounds.append((np.log(.01),np.log(1e4)))
            fits=[minimize(loss,start,bounds=bounds,method='L-BFGS-B',
                options=dict(maxiter=45,ftol=1e-10,gtol=1e-5,eps=2e-4)) for start in starts]
            fit=min(fits,key=lambda r:r.fun)
            theta,T=fit.x[:2]
            lam=float(np.exp(fit.x[2])) if precision is None else precision
            case[label]=dict(theta=float(theta),T=float(T),lambda_=lam,score=float(-fit.fun),
                success=bool(fit.success),message=str(fit.message),at_upper_bound=bool(theta>2.499 or T>3.999))
        results.append(case)
        print(json.dumps(case),flush=True)
    out.write_text(json.dumps(dict(results=results,failures=failures,evaluations=len(cache),
        qualification='Multiple-start local optimization on theta in [0,2.5], T in [.1,4]; not certified global optimization.',
        fixed=dict(mu=2,S0=1,sigma=1,c=1,delay=.35)),indent=2)+'\n')


if __name__=='__main__': main()
