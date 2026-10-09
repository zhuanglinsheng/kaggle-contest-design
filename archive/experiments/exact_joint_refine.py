#!/usr/bin/env python3
"""Continuous refinement of an interior finite-precision exact-HJB case."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from archive.experiments.exact_joint_design import exact_B
from pipeline.config import ROOT

OUT = ROOT/'results/theory/exact_joint_design'
A, DELAY, MU = 6.2, .35, 2.0


def objective(theta, horizon, precision, *, dx, steps):
    try:
        output,_ = exact_B(theta,horizon,precision,MU,dx=dx,steps=steps)
    except (RuntimeError, FloatingPointError):
        return -1e10, np.nan
    return A*np.log(output)-theta-DELAY*horizon*horizon/2, output


def optimize(high_precision=None):
    cache={}
    if high_precision is None:
        bounds=[(.5,2.5),(.5,4.),(np.log(.03),np.log(30.))]
    else:
        bounds=[(.5,2.5),(.5,4.)]

    def loss(x):
        if high_precision is None:
            theta,horizon,log_precision=x; precision=float(np.exp(log_precision))
        else:
            theta,horizon=x; precision=high_precision
        key=tuple(np.round([theta,horizon,precision],10))
        if key not in cache:
            cache[key]=objective(theta,horizon,precision,dx=.05,steps=350)[0]
        return -cache[key]

    fit=differential_evolution(loss,bounds,seed=2718,popsize=7,maxiter=18,
        tol=2e-3,polish=True,workers=1,updating='immediate')
    if high_precision is None:
        theta,horizon,log_precision=fit.x; precision=float(np.exp(log_precision))
    else:
        theta,horizon=fit.x; precision=high_precision
    return dict(theta=float(theta),T=float(horizon),lambda_=precision,
        score=float(-fit.fun),evaluations=fit.nfev,success=bool(fit.success),message=fit.message)


def main():
    finite=optimize()
    high=optimize(1e4)
    validation=[]
    for name,design in [('finite',finite),('high',high)]:
        for dx,steps in [(.05,350),(.035,600),(.025,1000)]:
            score,output=objective(design['theta'],design['T'],design['lambda_'],dx=dx,steps=steps)
            validation.append(dict(design=name,dx=dx,steps=steps,B=output,score=score))
    high_tail=[]
    for precision in [1e4,1e5,1e6]:
        score,output=objective(high['theta'],high['T'],precision,dx=.018,steps=1400)
        high_tail.append(dict(lambda_=precision,B=output,score=score))
    result=dict(specification=dict(A=A,delay=DELAY,delay_cost='delay*T^2/2',mu=MU,
        sigma=1,c=1,S0=1,theta_bounds=[.5,2.5],T_bounds=[.5,4],lambda_bounds=[.03,30]),
        finite=finite,high_precision=high,validation=validation,high_precision_tail=high_tail,
        refined_score_gap=validation[2]['score']-validation[5]['score'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'refined.json').write_text(json.dumps(result,indent=2)+'\n')
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-contest-design-matplotlib')
    import matplotlib.pyplot as plt
    grid=np.load(OUT/'grid.npz')['rows']
    grid=grid[grid[:,0]==MU]
    precisions=np.unique(grid[:,3]); profile=[]
    for precision in precisions:
        subset=grid[grid[:,3]==precision]
        scores=A*np.log(subset[:,4])-subset[:,1]-DELAY*subset[:,2]**2/2
        profile.append(scores.max())
    fig,axis=plt.subplots(figsize=(5.8,3.8),constrained_layout=True)
    axis.plot(precisions,profile,marker='o',markersize=3,label='Profiled grid objective')
    axis.scatter([finite['lambda_']],[validation[2]['score']],color='tab:red',zorder=3,
        label='Refined finite design')
    axis.scatter([high['lambda_']],[validation[5]['score']],color='black',zorder=3,
        label='Reoptimized high precision')
    axis.set_xscale('log');axis.set_xlabel(r'Precision $\lambda$')
    axis.set_ylabel(r'$6.2\log B-\theta-0.35T^2/2$')
    axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.2)
    axis.legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'refined_profile.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
