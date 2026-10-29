"""Full-disclosure output in the original asymmetric nonlinear game.

The analytic value-gradient transformation and independent-coordinate
discretization were cross-checked in the JK9 review. Initial uncertainty is
revealed to players and averaged over by the organizer, rather than discarded.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, ndtr, ndtri, roots_hermitenorm
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded


def full_output(theta, T, sigma, ci, cj, S0, mu, *, nodes=8001,
                cutoff=26., quadrature=160):
    """Return nonlinear output and numerical diagnostics, not a prize slope."""
    if min(theta, T, sigma, ci, cj, S0) <= 0 or nodes < 5:
        raise ValueError("Positive primitives and at least five nodes required")
    wi, wj = theta/(ci*sigma*sigma), theta/(cj*sigma*sigma)
    b = math.log(-math.expm1(-wi))-math.log(-math.expm1(-wj))
    d = -wj+(wi+wj)*expit(np.linspace(-cutoff, cutoff, nodes))
    a, c = d+wj, wi-d
    lower = .5*(a+2*np.cosh((d-wj+2*b)/2)*np.sinh(a/2))
    upper = .5*(c+2*np.cosh((wi+d+2*b)/2)*np.sinh(c/2))
    span = ((wi+math.sinh(wi+b))-(-wj+math.sinh(b-wj)))/2
    z = np.where(lower <= upper, ndtri(lower/span), -ndtri(upper/span))
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise RuntimeError("Unresolved full-disclosure profile coordinates")
    density = span*np.exp(-z*z/2)/math.sqrt(2*math.pi)/((1+np.cosh(d+b))/2)
    fp = np.tanh((d+b)/2)
    hm, hp = np.diff(d)[:-1], np.diff(d)[1:]
    left = 2/(hm*(hm+hp))-fp[1:-1]/(hm+hp)
    right = 2/(hp*(hm+hp))+fp[1:-1]/(hm+hp)
    if min(left.min(), right.min()) <= 0:
        raise RuntimeError("Non-monotone full-disclosure output operator")
    diagonal = left+right+1/density[1:-1]**2
    source = (1+2*fp[1:-1]*(ndtr(z[1:-1])-.5))/density[1:-1]
    band = np.zeros((3, nodes-2))
    band[1] = 1
    band[0, 1:] = -right[:-1]/diagonal[:-1]
    band[2, :-1] = -left[1:]/diagonal[1:]
    h = np.r_[0., solve_banded((1, 1), band, source/diagonal,
                              check_finite=False), 0.]
    residual = h[1:-1]-left/diagonal*h[:-2]-right/diagonal*h[2:]-source/diagonal
    if not np.isfinite(h).all() or h.min() < -1e-10 or np.max(abs(residual)) > 1e-8:
        raise RuntimeError("Invalid full-disclosure output solution")
    gh, weights = roots_hermitenorm(quadrature)
    scale = sigma*math.sqrt(T)
    xx = (mu+math.sqrt(S0)*gh)/scale
    inside = (xx >= z[0]) & (xx <= z[-1])
    hh = np.zeros_like(xx)
    hh[inside] = PchipInterpolator(z, h)(xx[inside])
    gain = scale*float(weights@hh)/math.sqrt(2*math.pi)
    prior_sd, prior_mean = math.sqrt(S0)/scale, mu/scale
    tail = float(ndtr((z[0]-prior_mean)/prior_sd)+ndtr((prior_mean-z[-1])/prior_sd))
    lc = lambda x: np.logaddexp(x, -x)-math.log(2)
    ff = wj+2*(lc((d[[0, -1]]+b)/2)-lc((b-wj)/2))
    value_sum = (ff+d[[0, -1]])/(2*wi)+(ff-d[[0, -1]])/(2*wj)
    boundary = scale*math.sqrt(2*max(wi, wj)*(max(0., float(1-value_sum.min()))+1e-12))
    total = S0+sigma*sigma*T
    baseline = math.sqrt(total/(2*math.pi))*math.exp(-mu*mu/(2*total))
    baseline += mu*(float(ndtr(mu/math.sqrt(total)))-.5)
    return dict(B=baseline+gain, baseline=baseline, gain=gain,
                tail_budget=scale*math.sqrt(2*max(wi, wj))*tail,
                boundary_budget=boundary, operator_residual=float(np.max(abs(residual))))
