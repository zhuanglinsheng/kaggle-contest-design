"""Summarize ALL draws and diagnostics without selecting seeds or favorable fits."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from pipeline.config import ROOT, MATPLOTLIB_CACHE_DIR
from pipeline.synthetic.chapter4 import OUT, NAMES, PARAMS, STAN

os.environ.setdefault('MPLCONFIGDIR',str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    summaries={name:json.loads((OUT/f'{name}_summary.json').read_text()) for name in NAMES}
    for name in NAMES:
        run=json.loads((OUT/f'{name}_run.json').read_text())
        if run['data_sha256']!=hashlib.sha256((OUT/f'{name}.json').read_bytes()).hexdigest():
            raise RuntimeError(f'Data have changed since the retained fit: {name}')
        if run['stan_sha256']!=hashlib.sha256(STAN.read_bytes()).hexdigest():
            raise RuntimeError(f'Stan model has changed since the retained fit: {name}')
    with (OUT/'posterior_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['experiment','parameter','true','mean','sd','rmsd','q2_5','q97_5','rhat','ess_bulk','ess_tail'])
        writer.writeheader()
        for name,s in summaries.items():
            for p,v in s.items():
                if p!='diagnostics': writer.writerow(dict(experiment=name,parameter=p,**v))
    with np.load(OUT/'baseline_paths.npz') as z:
        path,events=z['path'],z['events']
    fig,axs=plt.subplots(3,1,figsize=(9,6.4),gridspec_kw={'height_ratios':[3,1.3,1]},layout='constrained')
    for column,label,color in [(1,'Latent gap', '#202020'),(2,'Perceived gap','#2166ac')]:
        axs[0].plot(path[:,0],path[:,column],label=label,color=color,lw=1)
    axs[0].scatter(events[:,0],events[:,2],s=14,label='Public readout',facecolors='none',edgecolors='#b35806',linewidths=.7)
    axs[0].set(ylabel='Score gap',xlim=(0,90))
    axs[0].legend(ncol=3,frameon=False,fontsize=9)
    early=path[:,0]<=2
    axs[1].plot(path[early,0],path[early,4],color='#2166ac',lw=1.5)
    axs[1].axhline(2,color='gray',ls='--',lw=.8,label='Steady state')
    axs[1].set(xlim=(0,2),ylim=(1.9,4.1),xlabel='Days since start (early variance path)',ylabel=r'$S_t$')
    axs[1].legend(frameon=False,fontsize=8)
    for k in (0,1):
        axs[2].vlines(events[events[:,1]==k,0],1-k-.18,1-k+.18,color=['#2166ac','#b35806'][k],lw=.8)
    axs[2].set(xlim=(0,90),ylim=(-.4,1.4),yticks=[0,1],yticklabels=['Player j','Player i'],xlabel='Days since start')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.savefig(ROOT/'paper/synthetic_nonsteady_hjb.png',dpi=240)
    plt.close(fig)

    lines=['# Chapter 4: nonlinear, non-steady-state rerun','',
           'DGP: numerical nonlinear HJB equilibrium, S0=4, sigma=2, lambda=1, theta=1, ci=1.2, cj=1.5, mu0=0. Time unit: day.',
           'Estimator: first-order effort, freely estimated S0, continuous held-readout perceived-state approximation, sequential Gaussian observation innovations. No private-score path enters estimation.',
           '', '## Reproduction','',
           '```sh','.venv/bin/python -m pipeline.synthetic.chapter4 --prepare',
           '.venv/bin/python -m pipeline.synthetic.chapter4 --fit all --warmup 1000 --samples 1000',
           '.venv/bin/python -m pipeline.synthetic.chapter4_checks',
           '.venv/bin/python -m pipeline.synthetic.chapter4_report','```','',
           '## Sampling diagnostics','',
           '| Experiment | i / j submissions | r | max R-hat | min bulk / tail ESS | Divergences | Passed |',
           '|---|---:|---:|---:|---:|---:|---|']
    for name,s in summaries.items():
        d=s['diagnostics']; contests=manifest['experiments'][name]['contests']
        ni=sum(sum(int(e[1])==0 for e in c['events']) for c in contests)
        nj=sum(sum(int(e[1])==1 for e in c['events']) for c in contests)
        lines.append(f'| {name} | {ni} / {nj} | {contests[0]["ratio"]:.4f} | {d["max_rhat"]:.4f} | {d["min_ess_bulk"]:.0f} / {d["min_ess_tail"]:.0f} | {d["divergences"]} | {d["passed"]} |')
    warnings=[]
    for name in NAMES:
        run=json.loads((OUT/f'{name}_run.json').read_text())
        first_csv=Path(run['csv_files'][0])
        prefix=first_csv.stem.rsplit('_',1)[0]
        counts={'warmup':0,'sampling':0}
        for log in first_csv.parent.glob(prefix+'_*stdout.txt'):
            phase='warmup'
            for line in log.read_text().splitlines():
                if 'Iteration:' in line and '(Sampling)' in line: phase='sampling'
                if 'Exception:' in line: counts[phase]+=1
        if sum(counts.values()): warnings.append(f'{name}: {counts["warmup"]} warmup, {counts["sampling"]} sampling')
    if warnings:
        lines+=['','Rejected non-finite proposal warnings in the retained-run logs: '+ '; '.join(warnings)+
                '. These are reported separately from divergent transitions; the retained draws and final diagnostics are finite.']
    lines+=['','## Posterior results','',
            'RMSD is sqrt(mean((draw - truth)^2)) within the reported posterior, not frequentist RMSE across Monte Carlo repetitions. Pools of 10 and 20 contests are nested. All preassigned seeds are retained.','']
    for name,s in summaries.items():
        lines += [f'### {name}','','| Parameter | Truth | Mean | SD | Posterior RMSD | 95% interval |','|---|---:|---:|---:|---:|---|']
        for p,v in s.items():
            if p=='diagnostics': continue
            lines.append(f'| {p} | {v["true"]:.2f} | {v["mean"]:.3f} | {v["sd"]:.3f} | {v["rmsd"]:.3f} | [{v["q2_5"]:.3f}, {v["q97_5"]:.3f}] |')
        lines.append('')
    lines+=['## Interpretation and limits','',
            '- These runs implement the maintained approximate estimator; they do not establish consistency or validate the continuous-update/endpoint-observation approximations.',
            '- The mu0 prior follows Section 3: N(first observed public gap,1). For a common mu0 across pooled contests, its center is the arithmetic mean of first readouts, with the prior included once. This empirical-Bayes choice is documented, not tuned to recovery.',
            '- Each single contest supplies only one initial-state draw. S0 can remain prior-sensitive even with clean MCMC diagnostics. A large posterior interval is not a sampling failure.',
            '- For longer horizons, r is calibrated to match the realized integrated intensity per day of the baseline (relative tolerance 0.5%), using fixed Brownian paths and without inspecting posterior estimates.',
            '- HJB grid: dy=0.2, dt=1/48 day; verification at dy=0.1, dt=1/96. DGP diffusion step <=1/96 day. Likelihood integration step <=1/24 day, split at every exact event time.',
            '- Raw draws, manifest, exact event times, path arrays, and numerical HJB policies are retained alongside this report.','']
    first=manifest['experiments']['baseline_fixed_r']['contests'][0]['events'][0][0]
    sensitivity=(np.cosh(2*first)+2*np.sinh(2*first))**-2
    lines += ['## Initial-condition information','',
              f'The baseline first public readout occurs at day {first:.4f}. At that time, dS(t)/dS0={sensitivity:.3g}. This quantifies loss of the filtering-variance channel, not loss of all information about S0 from the joint data.',
              f'The first observed public gap is {manifest["experiments"]["baseline_fixed_r"]["mu_prior_center"]:.4f}; Section 3 therefore places the single-contest prior for mu0 far from its generating value zero. This rule was retained rather than tuned after seeing the result.','']
    checks_path=OUT/'integration_checks.json'
    if checks_path.exists():
        checks=json.loads(checks_path.read_text())
        lines+=['## Likelihood integration check','',
                'At 24 retained posterior draws per fit, halve the maximum integration step from 1/24 to 1/48 day. These are numerical sensitivity checks, not extra posterior fits.','',
                '| Experiment | SD of log-density correction | Importance ESS / 24 |',
                '|---|---:|---:|']
        for name,c in checks.items():
            lines.append(f'| {name} | {c["sd_log_density_correction"]:.5f} | {c["importance_ess_fraction"]:.6f} |')
        lines.append('')
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(OUT/'REPORT.md')


if __name__=='__main__': main()
