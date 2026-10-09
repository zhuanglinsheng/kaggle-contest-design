"""Nonlinear HJB policies in the preserved hourly simulator; no estimation.

Only the effort provider differs from the variance-only generator. Both the
S0=2 and S0=4 experiments use the numerical policy of the theoretical model.
The held-readout approximation is retained, not replaced by continuous signals.
"""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from pipeline.config import ROOT
from pipeline.synthetic.generate import LONG_R, generate_contest, save_contest, plot_baseline
from pipeline.synthetic.nonlinear import Primitives, solve_equilibrium, variance

OUT = ROOT / "results/synthetic/optimal_hourly"


def checked_policy(eq):
    def evaluate(t, y):
        if not eq.y[0] <= y <= eq.y[-1] or not 0 <= t <= eq.p.T:
            raise ValueError("Simulated state outside the solved HJB grid")
        return eq.at(t, y)
    return evaluate


def grid_check(eq):
    fine = solve_equilibrium(eq.p, dy=.1, dt=1/96)
    errors, levels = [], []
    for t in np.unique(np.r_[0., .1, .5, 1., eq.p.T/2, eq.p.T-1, eq.p.T]):
        Q = float(variance(t, eq.p))+eq.p.sigma**2*(eq.p.T-t)
        for y in np.linspace(-3,3,31)*np.sqrt(Q):
            reference = fine.at(t,y)
            errors.extend(abs(eq.at(t,y)-reference))
            levels.extend(reference)
    result = dict(max_absolute_error=float(max(errors)),
                  max_error_over_peak=float(max(errors)/max(levels)),
                  rms_error_over_rms=float(np.linalg.norm(errors)/np.linalg.norm(levels)))
    if result['max_error_over_peak'] > .01:
        raise RuntimeError(f"HJB refinement check failed: {result}")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="data generation only; no Bayesian estimation",
                    policy="numerical nonlinear HJB equilibrium m; not leading-order m^(1)",
                    observation="original hourly held-readout approximation",
                    simulation_dt=1/24, hjb_dy=.2, hjb_dt=1/48,
                    refinement_dy=.1, refinement_dt=1/96,
                    changed_since_variance_only=["effort policy only"],
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                        ['synthetic_data/synthetic_data.py','pipeline/synthetic/nonlinear.py',
                         'pipeline/synthetic/generate.py','pipeline/synthetic/optimal_generate.py']},
                    policies={}, experiments={},
                    pools={str(n):[f"pool_contest{i+1:02}" for i in range(n)] for n in (10,20)})
    for days in (90,180,360):
        experiments = ([("baseline",1000,15.)]+
                       [(f"pool_contest{i+1:02}",3000+10*i,15.) for i in range(20)]
                       if days == 90 else [(f"days{days}",5000+days,LONG_R[days])])
        for variant, S0 in (("steady",2.),("nonsteady",4.)):
            folder = OUT/variant
            folder.mkdir(exist_ok=True)
            p = Primitives(T=float(days), S0=S0)
            print(f"Solving HJB: T={days}, S0={S0}",flush=True)
            eq = solve_equilibrium(p, dy=.2, dt=1/48)
            refinement = grid_check(eq)
            eq.save(folder/f"policy_T{days}.npz")
            manifest['policies'][f'{variant}_{days}'] = dict(primitives=asdict(p),
                diagnostics=eq.diagnostics, refinement=refinement)
            print('Refinement:',refinement,flush=True)
            policy = checked_policy(eq)
            for name, seed, ratio in experiments:
                if ratio*eq.diagnostics['max_policy']/24 > 1:
                    raise RuntimeError("Original thinning bound is insufficient for the HJB policy")
                result = generate_contest(days,seed,ratio,S0,effort_policy=policy)
                summary = save_contest(folder,name,result,S0,ratio)
                summary['max_thinning_acceptance'] = float(ratio*max(result[1].max(),result[2].max())/24)
                if not summary['finite']:
                    raise RuntimeError('Non-finite generated path')
                entry = manifest['experiments'].setdefault(name,dict(days=days,ratio=ratio,
                    seeds=dict(brownian=seed,poisson=seed+1,uniform=seed+2,initial=seed+3),results={}))
                entry['results'][variant] = summary
                if name == 'baseline':
                    plot_baseline(result,folder/'figure3.png')
                print(name,variant,summary,flush=True)
            del eq, policy
    lines=['# Optimal-policy hourly data generation','',
           'Theoretical nonlinear HJB policy m replaces the first-order policy. No Bayesian estimation run.',
           'Both steady (S0=2) and nonsteady (S0=4) samples retain the original hourly grid, held public readouts,',
           'readout-noise intervals, thinning, RNG seeds, initial draws, and long-horizon r values.',
           'The hourly belief recursion remains an empirical approximation to the theoretical continuous-signal filter.','',
           '| Experiment | Steady i / j | S0=4 i / j |','|---|---:|---:|']
    for name,entry in manifest['experiments'].items():
        a,b=entry['results']['steady'],entry['results']['nonsteady']
        lines.append(f'| {name} | {a["submissions_i"]} / {a["submissions_j"]} | {b["submissions_i"]} / {b["submissions_j"]} |')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
