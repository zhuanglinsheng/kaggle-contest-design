"""Diagnostic only: exact nonlinear auxiliary and same-terminal bridge sources.

Finite-difference and Gaussian quadrature output is not a proof or a
joint-design classification. No paper, empirical data, or figures are written.
"""

import argparse
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss


def audit(w, spacing, dt):
    z = np.linspace(-8, 8, round(16 / spacing) + 1)
    dz = z[1] - z[0]
    G = np.exp(-z*z/2) / np.sqrt(2*np.pi)
    g = 2*ndtr(z)-1
    k = (1+np.sinh(w)/w)/2
    d_full = g.copy()
    for _ in range(12):
        d_full -= ((d_full+np.sinh(w*d_full)/w)/2-k*g) / (
            (1+np.cosh(w*d_full))/2
        )
    dp = 2*k*G / ((1+np.cosh(w*d_full))/2)
    rp = np.tanh(w*d_full/2)*dp
    lo = .5/dz**2-(z[1:-1]/2+w*rp[1:-1])/(2*dz)
    up = .5/dz**2+(z[1:-1]/2+w*rp[1:-1])/(2*dz)
    ab = np.zeros((3, len(z)-2))
    ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], -.5-1/dz**2, lo[1:]
    rhs = -w*dp[1:-1]/2
    rhs[0] -= 4*lo[0]
    rhs[-1] -= 4*up[-1]
    beta = np.r_[4, solve_banded((1, 1), ab, rhs), 4]
    bp = np.gradient(beta, dz)
    ratio = (rp*bp+dp/2)/G
    lo = .5/dz**2-z[1:-1]/(4*dz)
    up = .5/dz**2+z[1:-1]/(4*dz)
    mat = np.zeros_like(ab)
    mat[0, 1:], mat[1], mat[2, :-1] = -dt*up[:-1], 1+dt/dz**2, -dt*lo[1:]
    R, d = np.ones_like(z), g.copy()
    x, weights = hermgauss(100)
    x, weights = np.sqrt(2)*x, weights/np.sqrt(np.pi)
    selected = np.flatnonzero(abs(z) <= 4)
    # Coarser diagnostic state grid; evolution still uses every state.
    selected = selected[::max(1, round(.02/dz))]
    checkpoints = {round(t/dt): t for t in [.25, 1, 3, 5]}
    for n in range(round(5/dt)+1):
        Rp, Dp = np.gradient(R, dz), np.gradient(d, dz)
        if n in checkpoints:
            time = n*dt
            eps, a = np.exp(-time), 1-np.exp(-time)
            zz = z[selected]
            # Product of the Gaussian convolution kernel and G_a is G_1
            # times a conditional Gaussian of mean sqrt(a)*z, variance eps.
            C = G[selected] * (
                np.interp(np.sqrt(a)*zz[:, None]+np.sqrt(eps)*x, z, ratio) @ weights
            )
            Fp = np.interp(
                (zz[:, None]+np.sqrt(eps)*x)/np.sqrt(a), z, bp,
                left=-.5, right=.5,
            ) @ weights
            bridge = C-Rp[selected]*Fp-Dp[selected]/2
            original = w*((Rp[selected]-rp[selected])*bp[selected]+(Dp[selected]-dp[selected])/2)
            imin, imax = np.argmin(bridge/G[selected]), np.argmax(bridge/G[selected])
            print(f'w={w:g} v={time:g} bridge/G min={bridge[imin]/G[selected][imin]:.8g}'
                  f' at z={zz[imin]:g}; max={bridge[imax]/G[selected][imax]:.8g};'
                  f' original max={original.max():.8g}')
            positive = (zz >= .2) & (zz <= 4)
            derivative_gap = (Rp[selected][positive]-rp[selected][positive]) / G[selected][positive]
            values = R[selected][positive] - (
                1 + 2/w*np.log(np.cosh(w*d_full[selected][positive]/2)/np.cosh(w/2))
            )
            print(f'  player comparison: min(R-R_full)={values.min():.8g};'
                  f' max((R_z-R_full_z)/G)={derivative_gap.max():.8g}'
                  f' at z={zz[positive][derivative_gap.argmax()]:g}')
            conditional = np.sqrt(a)*zz[:, None]+np.sqrt(eps)*x
            bar_r = G[selected]*(np.interp(conditional, z, rp/G) @ weights)
            bar_d = G[selected]*(np.interp(conditional, z, dp/G) @ weights)
            bridge_r_gap = (bar_r[positive]-Rp[selected][positive])/G[selected][positive]
            bridge_d_gap = (bar_d-Dp[selected])/G[selected]
            print(f'  same-terminal gradients: min((bar_R_z-R_z)/G)='
                  f'{bridge_r_gap.min():.8g}; (bar_d_z-d_z)/G range='
                  f'[{bridge_d_gap.min():.8g}, {bridge_d_gap.max():.8g}]')
        rr = R[1:-1]+dt*w*(3*Rp[1:-1]**2-Dp[1:-1]**2)/4
        dd = d[1:-1]+dt*w*Rp[1:-1]*Dp[1:-1]/2
        rr[0] += dt*lo[0]
        rr[-1] += dt*up[-1]
        dd[0] -= dt*lo[0]
        dd[-1] += dt*up[-1]
        R[1:-1] = solve_banded((1, 1), mat, rr)
        d[1:-1] = solve_banded((1, 1), mat, dd)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spacing', type=float, default=.01)
    p.add_argument('--dt', type=float, default=.002)
    p.add_argument('--incentives', type=float, nargs='+', default=[.1, .5, .8, 1.5])
    args = p.parse_args()
    for incentive in args.incentives:
        audit(incentive, args.spacing, args.dt)
