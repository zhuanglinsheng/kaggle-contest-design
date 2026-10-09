"""Explore costless precision, normalized incentives, and exact output channels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from archive.experiments.exact_joint_design import solve_exact
from pipeline.config import ROOT

OUT = ROOT / 'results/theory/precision_conditions'


def evaluate(w, s0, ell, zs, *, dx=.045, steps=400):
    result = solve_exact(w=w, ell=ell, s0=s0, x0=0, dx=dx, steps=steps, decompose=True)
    p = result.profile
    rows = []
    for z in zs:
        x0 = z*np.sqrt(1+s0)
        row = dict(w=w, s0=s0, ell=ell, z=z, x0=float(x0))
        for name in ['B', 'half_effort', 'gap_contribution', 'baseline']:
            row[name] = float(np.interp(x0, p['x'], p[name]))
        row['identity_error'] = row['B']-row['baseline']-row['half_effort']-row['gap_contribution']
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ws = [.5, 1., 1.25, 1.5, 1.75, 2., 2.25]
    s0s = [.1, .5, 1., 2., 4.]
    ells = [0., .001, .01, .1, .3, 1., 3., 10., 30., 100., 1e3, 1e4, 1e5]
    zs = [0., .5, 1., 1.5, 2.]
    rows, failures = [], []
    for s0 in s0s:
        for w in ws:
            for ell in ells:
                try:
                    rows.extend(evaluate(w,s0,ell,zs))
                except (RuntimeError, FloatingPointError) as error:
                    failures.append(dict(w=w,s0=s0,ell=ell,error=str(error)))
        print(f'completed s0={s0}; failed solves={len(failures)}', flush=True)
    profiles=[]
    for s0 in s0s:
        for w in ws:
            for z in zs:
                candidates=[r for r in rows if r['s0']==s0 and r['w']==w and r['z']==z]
                high=next((r for r in candidates if r['ell']==ells[-1]),None)
                if high is None:
                    continue
                best=max(candidates,key=lambda r:r['B'])
                zero=next((r for r in candidates if r['ell']==0),None)
                profiles.append(dict(s0=s0,w=w,z=z,best=best,high=high,zero=zero,
                    improvement=best['B']-high['B'],valid_precisions=len(candidates)))
    validation=[]
    for s0 in [.1,1.,4.]:
        for w in [1.,1.5,2.]:
            selected=next(r for r in profiles if r['s0']==s0 and r['w']==w and r['z']==0)
            for ell in sorted(set([selected['best']['ell'],1e5])):
                try:
                    validation.extend(evaluate(w,s0,ell,[0.,1.],dx=.022,steps=1000))
                except (RuntimeError,FloatingPointError) as error:
                    validation.append(dict(w=w,s0=s0,ell=ell,error=str(error)))
    result=dict(normalization='sigma=c=T=1; w=theta/(sigma^2*c), s0=S0/(sigma^2*T), ell=lambda*sigma^2*T^2, z=mu/sqrt(S0+sigma^2*T)',
        rows=rows,failures=failures,profiles=profiles,validation=validation)
    (OUT/'scan.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved scan and refinement.',flush=True)
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-contest-design-matplotlib')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for w in [.5,1.,1.5,2.]:
        selected=[r for r in rows if r['s0']==1 and r['w']==w and r['z']==0 and r['ell']>0]
        high=selected[-1]['B']
        axes[0].plot([r['ell'] for r in selected],[r['B']-high for r in selected],label=f'w = {w}')
    axes[0].axhline(0,color='grey',lw=.7)
    axes[0].set_ylabel('Output relative to precision $10^5$')
    selected=[r for r in rows if r['s0']==1 and r['w']==2 and r['z']==0 and r['ell']>0]
    for key,label in [('half_effort','Half total effort'),('gap_contribution','Gap-drift contribution'),('B','Best output')]:
        axes[1].plot([r['ell'] for r in selected],[r[key]-selected[-1][key] for r in selected],label=label)
    axes[1].axhline(0,color='grey',lw=.7)
    axes[1].set_ylabel('Contribution relative to precision $10^5$')
    for axis in axes:
        axis.set_xscale('log');axis.set_xlabel(r'Normalized precision $\ell$')
        axis.legend(frameon=False,fontsize=8)
        axis.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'mechanism.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
