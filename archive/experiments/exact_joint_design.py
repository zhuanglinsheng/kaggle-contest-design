#!/usr/bin/env python3
"""Exact-HJB numerical search for finite costless precision.

The dimensionless equilibrium depends on w=theta/(sigma^2*c),
ell=lambda*sigma^2*T^2, s0=S0/(sigma^2*T), and x0=mu/(sigma*sqrt(T)).
After solving the nonlinear player HJBs, a separate linear backward equation
computes the organizer's exact expected best output.  The reported benchmark
sets sigma=c=S0=1 and uses u(B)=A*log(B), D(T)=d*T.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr

from pipeline.config import ROOT

OUT = ROOT / "results" / "theory" / "exact_joint_design"


def filtering_variance(tau, s0, ell):
    tau = np.asarray(tau)
    if ell == 0:
        return s0 + tau
    root = np.sqrt(ell)
    steady = 1 / root
    z = np.tanh(root * tau)
    return steady * (s0 + steady * z) / (steady + s0 * z)


def beta(x, variance):
    scale = np.sqrt(variance)
    z = x / scale
    return scale / np.sqrt(2 * np.pi) * np.exp(-z*z/2) + x * (ndtr(z)-.5)


@dataclass
class SolveResult:
    best_output: float
    diagnostics: dict
    profile: dict | None = None


def solve_exact(*, w, ell, s0, x0, dx=.04, steps=400, extent=7., decompose=False) -> SolveResult:
    """Solve the two player HJBs and the organizer equation."""
    width = np.ceil(extent * np.sqrt(s0 + 1) / dx) * dx
    x = np.arange(-width, width + dx/2, dx)
    uniform_tau = np.linspace(0., 1., steps + 1)
    if ell > 0:
        # Resolve equal increments of posterior-mean quadratic variation as
        # well as calendar time.  This captures the O(1/ell) initial
        # revelation layer when precision is large.
        dense_tau = np.r_[0., np.geomspace(1e-12, 1., max(20000, 50*steps))]
        dense_s = filtering_variance(dense_tau, s0, ell)
        dense_a = s0 + 1 - (dense_s + 1 - dense_tau)
        target_a = np.linspace(0., dense_a[-1], steps + 1)
        variance_tau = np.interp(target_a, dense_a, dense_tau)
        tau = np.unique(np.r_[uniform_tau, variance_tau])
    else:
        tau = uniform_tau
    nsteps = len(tau)-1
    svar = filtering_variance(tau, s0, ell)
    u = np.column_stack((ndtr(x/np.sqrt(svar[-1])), ndtr(-x/np.sqrt(svar[-1]))))
    effort = np.empty((nsteps + 1, len(x), 2), dtype=np.float64)
    band = np.zeros((3, len(x)-2))
    min_effort = 0.

    def gradient(v):
        return np.gradient(v, dx, axis=0, edge_order=2)

    def nonlinear(v):
        g = gradient(v); cross = g[:, 0]*g[:, 1]
        return w*np.column_stack((g[:, 0]**2/2+cross, g[:, 1]**2/2+cross))

    for k in range(nsteps, -1, -1):
        g = gradient(u)
        effort[k, :, 0] = w*g[:, 0]
        effort[k, :, 1] = -w*g[:, 1]
        min_effort = min(min_effort, float(effort[k].min()))
        if min_effort < -2e-6:
            raise RuntimeError(f"negative equilibrium effort {min_effort}")
        effort[k] = np.maximum(effort[k], 0)
        if k == 0:
            break
        dt = tau[k]-tau[k-1]
        diffusion_variance = (svar[k-1]+1-tau[k-1])-(svar[k]+1-tau[k])
        r = diffusion_variance/(4*dx**2)
        band[0, 1:] = -r; band[1] = 1+2*r; band[2, :-1] = -r
        base = u[1:-1]+r*(u[:-2]-2*u[1:-1]+u[2:])
        base[0] += r*u[0]; base[-1] += r*u[-1]
        force = nonlinear(u)
        predictor = u.copy()
        predictor[1:-1] = solve_banded((1,1), band, base+dt*force[1:-1], check_finite=False)
        u[1:-1] = solve_banded((1,1), band,
            base+dt/2*(force[1:-1]+nonlinear(predictor)[1:-1]), check_finite=False)

    # Organizer's normalized value. Boundary values |x|/2 are valid because
    # equilibrium effort is exponentially negligible at the chosen boundary.
    organizer = beta(x, svar[-1])
    if decompose:
        # Also integrate the exact HJB effort identity: H = w*h(0,0) + K - w*D.
        organizer = np.column_stack((organizer, np.zeros((len(x),5))))
    boundary = np.abs(x[[0,-1]])/2
    if decompose:
        boundary = np.column_stack((boundary, np.zeros((2,5))))
    organizer[[0,-1]] = boundary
    for k in range(nsteps, 0, -1):
        dt = tau[k]-tau[k-1]
        diffusion_variance = (svar[k-1]+1-tau[k-1])-(svar[k]+1-tau[k])
        r = diffusion_variance/(4*dx**2)
        band[0, 1:] = -r; band[1] = 1+2*r; band[2, :-1] = -r
        base = organizer[1:-1]+r*(organizer[:-2]-2*organizer[1:-1]+organizer[2:])
        base[0] += r*boundary[0]; base[-1] += r*boundary[1]

        def forcing(value, policy, q, time):
            grad = np.gradient(value, dx, axis=0, edge_order=2)
            drift = policy[:,0]-policy[:,1]
            source = (policy[:,0]+policy[:,1])/2
            if decompose:
                slopes=np.gradient(policy,dx,axis=0,edge_order=2)
                h=np.exp(-x*x/(2*q))/np.sqrt(2*np.pi*q)
                feedback=time/2*(policy[:,1]*slopes[:,0]-policy[:,0]*slopes[:,1])
                return drift[:,None]*grad + np.column_stack((source, source,
                    drift*(ndtr(x/np.sqrt(q))-.5),drift*x*h/q,feedback,
                    (1-time)*drift*x*h/q))
            return drift*grad+source

        f_now = forcing(organizer, effort[k], svar[k]+1-tau[k],tau[k])
        predictor = organizer.copy(); predictor[[0,-1]] = boundary
        predictor[1:-1] = solve_banded((1,1), band, base+dt*f_now[1:-1], check_finite=False)
        f_predict = forcing(predictor, effort[k-1], svar[k-1]+1-tau[k-1],tau[k-1])
        organizer[1:-1] = solve_banded((1,1), band,
            base+dt/2*(f_now[1:-1]+f_predict[1:-1]), check_finite=False)
        organizer[[0,-1]] = boundary

    profile = None
    if decompose:
        profile = dict(x=x, B=organizer[:,0], half_effort=organizer[:,1],
            gap_contribution=organizer[:,2], drift_effort_loss=organizer[:,3],
            feedback=organizer[:,4],remaining_drift_loss=organizer[:,5],baseline=beta(x,s0+1))
        organizer = organizer[:,0]
    value = float(np.interp(x0, x, organizer))
    return SolveResult(value, dict(dx=dx, requested_steps=steps, steps=nsteps, nodes=len(x), width=width,
        min_effort=min_effort, max_effort=float(effort.max()), boundary_effort=float(np.max(effort[:,[0,-1],:]))), profile)


def exact_B(theta, horizon, precision, mu, *, dx=.04, steps=400):
    # Benchmark normalization sigma=c=S0=1.
    if theta == 0:
        return float(beta(mu, 1+horizon)), dict(dx=dx, requested_steps=steps,
            steps=0, nodes=0, width=0., min_effort=0., max_effort=0., boundary_effort=0.)
    result = solve_exact(w=theta, ell=precision*horizon*horizon,
        s0=1/horizon, x0=mu/np.sqrt(horizon), dx=dx, steps=steps)
    return np.sqrt(horizon)*result.best_output, result.diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # w<=2 is the largest range that passes the monotonicity check uniformly
    # on the quick grid with the present interior-equilibrium solver.
    theta_grid=np.linspace(0,2,9 if args.quick else 13)
    horizon_grid=np.geomspace(.25,4,7 if args.quick else 11)
    # The zero-signal HJB is first order and requires a separate monotone
    # discretization; it is not needed for the finite-versus-high comparison.
    precision_grid=np.logspace(-3,4,15 if args.quick else 22)
    mus=[0.,.5,1.,2.]
    rows=[]
    cache={}
    for mu in mus:
        for horizon in horizon_grid:
            for theta in theta_grid:
                for precision in precision_grid:
                    key=(float(mu),float(theta),float(horizon),float(precision))
                    value,diag=exact_B(theta,horizon,precision,mu,
                        dx=.06 if args.quick else .04,steps=240 if args.quick else 400)
                    cache[key]=value
                    rows.append(dict(mu=key[0],theta=key[1],T=key[2],lambda_=key[3],B=value,
                                     boundary_effort=diag['boundary_effort']))
        print(f"completed mu={mu}",flush=True)

    objectives=[]
    for delay_power in [1,2]:
      for A in [1.,2.,5.,10.,20.]:
        for delay in [.01,.05,.1,.25,.5,1.]:
            for mu in mus:
                candidates=[]
                for row in rows:
                    if row['mu']==mu:
                        delay_cost=delay*row['T']**delay_power/delay_power
                        candidates.append((A*np.log(row['B'])-row['theta']-delay_cost,row))
                score,row=max(candidates,key=lambda item:item[0])
                high=max((item for item in candidates if item[1]['lambda_']==precision_grid[-1]),key=lambda item:item[0])
                objectives.append(dict(A=A,delay=delay,delay_power=delay_power,mu=mu,score=score,
                    best={k:row[k] for k in ('theta','T','lambda_','B')},
                    high_precision_score=high[0],high_precision={k:high[1][k] for k in ('theta','T','lambda_','B')},
                    finite_beats_high=bool(row['theta']>0 and row['lambda_']<precision_grid[-1] and score>high[0]+1e-5)))
    result=dict(status="coarse search",normalization=dict(sigma=1,c=1,S0=1),
        grids=dict(theta=theta_grid.tolist(),T=horizon_grid.tolist(),lambda_=precision_grid.tolist(),mu=mus),
        objectives=objectives,max_boundary_effort=max(row['boundary_effort'] for row in rows))
    (OUT/'coarse.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'grid.npz', rows=np.array([
        (r['mu'],r['theta'],r['T'],r['lambda_'],r['B']) for r in rows]),
        columns=np.array(['mu','theta','T','lambda','B']))
    print(json.dumps(dict(finite_cases=sum(v['finite_beats_high'] for v in objectives),
        objectives=len(objectives),max_boundary_effort=result['max_boundary_effort']),indent=2))


if __name__=='__main__': main()
