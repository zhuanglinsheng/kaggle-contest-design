"""Exploratory joint-design comparisons; not a certification of global optima.

All preferences and cost parameters are fixed across the innovation-risk scan.
The nonlinear HJB is solved directly. Numerical failures remain recorded and
are never treated as evidence that the corresponding designs are inferior.
"""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from archive.experiments.exact_joint_design import solve_exact
from pipeline.theory.joint_design_exact import full_disclosure_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.5,0.75,1.0,1.5])
    parser.add_argument('--output', default='disclosure_region_audit.json')
    args = parser.parse_args()
    # u(B)=3[1-exp(-5(B-sqrt(1/(2*pi))))], D(T)=T^2/2.
    # The utility shift sets immediate termination's payoff to zero.
    baseline = 1 / math.sqrt(2 * math.pi)
    failures = []
    results = []
    for sigma in args.sigmas:
        cache = {}

        def utility(b):
            return -3 * math.expm1(-5 * (b - baseline))

        def output(theta, horizon, precision, fine=False):
            if theta < 1e-10:
                return math.sqrt((1 + sigma * sigma * horizon) / (2 * math.pi))
            if math.isinf(precision):
                return full_disclosure_output(
                    theta=theta, horizon=horizon, initial_variance=1,
                    sigma=sigma, dy=(0.012 if fine else 0.035) * math.sqrt(1 + sigma*sigma*horizon),
                    dt=horizon/(1800 if fine else 500),
                )
            s0 = 1 / (sigma*sigma*horizon)
            ell = precision*sigma*sigma*horizon*horizon
            from archive.experiments.exact_joint_design import filtering_variance
            dx = min((0.014 if fine else 0.035)*math.sqrt(1+s0),
                     (0.10 if fine else 0.18)*math.sqrt(float(filtering_variance(1, s0, ell))))
            sol = solve_exact(w=theta/(sigma*sigma), ell=ell, s0=s0, x0=0,
                              dx=dx, steps=1200 if fine else 350)
            return sigma*math.sqrt(horizon)*sol.best_output

        def loss(x, full=False):
            theta, horizon = x[:2]
            precision = math.inf if full else math.exp(x[2])
            key = (float(theta),float(horizon),float(precision))
            if key not in cache:
                try:
                    b = output(*key)
                    score = utility(b)-theta-horizon*horizon/2
                    if not math.isfinite(score):
                        raise ValueError('nonfinite objective')
                    cache[key] = -score
                except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
                    failures.append(dict(sigma=sigma,theta=theta,T=horizon,precision=precision,error=str(exc)))
                    cache[key] = 1e4
            return cache[key]

        # Bounded utility implies theta+T^2/2 <= 3 for any nonnegative payoff.
        bounds = [(0,3), (0.005,math.sqrt(6))]
        zero = minimize_scalar(lambda t: -utility(math.sqrt((1+sigma*sigma*t)/(2*math.pi)))+t*t/2,
                               bounds=(0,math.sqrt(6)),method='bounded')
        case = dict(sigma=sigma,zero=dict(theta=0,T=float(zero.x),score=float(-zero.fun)))
        for full in (True,False):
            starts = [[0.3,0.8],[1.0,1.2]]
            if not full:
                starts = [v+[math.log(l)] for v,l in zip(starts,[30,1])]
            fits = [minimize(lambda x: loss(x,full),start,method='Powell',
                             bounds=bounds if full else bounds+[(math.log(0.03),math.log(3000))],
                             options=dict(maxiter=7,xtol=0.002,ftol=2e-5)) for start in starts]
            fit=min(fits,key=lambda v:v.fun)
            theta,horizon=map(float,fit.x[:2])
            precision=math.inf if full else math.exp(float(fit.x[2]))
            row=dict(theta=theta,T=horizon,precision=precision,score=float(-fit.fun),success=bool(fit.success),message=str(fit.message))
            try:
                b=output(theta,horizon,precision,True)
                row['fine_output']=b
                row['fine_score']=utility(b)-theta-horizon*horizon/2
            except (RuntimeError,FloatingPointError,ValueError,OverflowError) as exc:
                row['refinement_failure']=str(exc)
            case['full' if full else 'finite']=row
            print(json.dumps(dict(sigma=sigma,kind='full' if full else 'finite',result=row)),flush=True)
        case['evaluations']=len(cache)
        results.append(case)
    destination=Path('results/theory/precision_conditions') / args.output
    destination.write_text(json.dumps(dict(
        qualification='Two-start local searches, finite precision in [0.03,3000]; no global or tail certification; failed HJB points retained below.',
        primitives=dict(c=1,S0=1,utility='3*(1-exp(-5*(B-sqrt(1/(2*pi)))))',delay='T^2/2'),
        results=results,failures=failures),indent=2)+'\n')


if __name__=='__main__':
    main()
