"""Diagnostic of the exact comparison residual; not a sign certificate.

Reads existing HJB routines. Does not write manuscript, figures, or data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import solve_banded
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.synthetic.nonlinear import Primitives, solve_equilibrium, variance


def full_profile(w, dz):
    z = np.linspace(-10, 10, round(20 / dz) + 1)
    dz = z[1] - z[0]
    G = np.exp(-z*z/2) / np.sqrt(2*np.pi)
    g = 2*ndtr(z)-1
    k = (1+np.sinh(w)/w)/2
    d = g.copy()
    for _ in range(12):
        d -= ((d+np.sinh(w*d)/w)/2-k*g)/((1+np.cosh(w*d))/2)
    H = 2*k*G/((1+np.cosh(w*d))/2)
    A = np.tanh(w*d/2)*H
    lo = .5/dz**2-(z[1:-1]/2+w*A[1:-1])/(2*dz)
    up = .5/dz**2+(z[1:-1]/2+w*A[1:-1])/(2*dz)
    band = np.zeros((3, len(z)-2))
    band[0, 1:], band[1], band[2, :-1] = up[:-1], -.5-1/dz**2, lo[1:]
    rhs = -w*H[1:-1]/2
    rhs[0] -= 5*lo[0]
    rhs[-1] -= 5*up[-1]
    F = np.r_[5, solve_banded((1, 1), band, rhs), 5]
    Fp = np.gradient(F, dz)
    return z, Fp, (A*Fp+H/2)/G


def audit(s, precision, w, spacing, dt, profile):
    p = Primitives(T=1, theta=w, c_i=1, c_j=1, sigma=1,
                   lamb=precision, S0=s, mu0=0)
    eq = solve_equilibrium(p, dy=spacing*np.sqrt(s+1), dt=dt, extent=8)
    xp, bp, source_ratio = profile
    nodes, weights = hermgauss(120)
    nodes, weights = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
    for t in (0, .25, .5, .75, .95):
        tau = 1-t
        S = float(variance(t, p))
        Q = S+tau
        k = tau/Q
        zs = np.linspace(-4, 4, 401)
        ys = zs*np.sqrt(Q)
        index = round(t/dt)
        A = np.interp(ys, eq.y, (eq.effort[index, :, 0]-eq.effort[index, :, 1])/w)
        H = np.interp(ys, eq.y, (eq.effort[index, :, 0]+eq.effort[index, :, 1])/w)
        G = np.exp(-zs*zs/2)/np.sqrt(2*np.pi*Q)
        C = G*(np.interp(np.sqrt(k)*zs[:, None]+np.sqrt(1-k)*nodes,
                         xp, source_ratio)@weights)
        Fp = np.interp((ys[:, None]+np.sqrt(S)*nodes)/np.sqrt(tau),
                       xp, bp, left=-.5, right=.5)@weights
        gap = (C-A*Fp-H/2)/G
        j = np.argmin(gap)
        print(f's={s:g} lambda={precision:g} w={w:g} t={t:g} '
              f'negative-residual/G min={gap[j]:.9g} z={zs[j]:g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=float, nargs='+', default=[.01, 1, 100])
    parser.add_argument('--precisions', type=float, nargs='+', default=[.01, 1, 100])
    parser.add_argument('--w', type=float, default=.1)
    parser.add_argument('--spacing', type=float, default=.005)
    parser.add_argument('--dt', type=float, default=.0005)
    args = parser.parse_args()
    profile = full_profile(args.w, .005)
    for s in args.s:
        for precision in args.precisions:
            audit(s, precision, args.w, args.spacing, args.dt, profile)
