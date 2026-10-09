"""Symmetric-state cubic coefficient and finite-precision sufficient window.

The inequalities concern the cubic approximation, not the exact strong-prize
equilibrium. Endpoint coefficients are analytic limits, not high-lambda proxies.
"""
import json

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from archive.experiments.second_order_precision_no_cost import Coefficient
from archive.experiments.exact_joint_design import solve_exact
from pipeline.config import ROOT


def cubic_coefficient(s0, ell, order=100):
    node, weight=leggauss(order)
    a=(node+1)/2; weight=weight/2
    remaining=a[:,None]**2
    future_remaining=remaining*a[None,:]**2
    t=1-remaining; s=1-future_remaining
    weights=weight[:,None]*weight[None,:]*4*a[:,None]**3*a[None,:]
    if np.isinf(ell):
        qt=remaining; qs=future_remaining
    else:
        qt=Coefficient.q(t,s0,ell); qs=Coefficient.q(s,s0,ell)
    variance=s0+1-qt
    denominator=qt*(2*qt-qs)+variance*(4*qt-qs)
    integrand=(t*qt-2*remaining*variance)/(
        (2*np.pi)**1.5*np.sqrt(qs)*denominator**1.5)
    return float(np.sum(weights*integrand))


def endpoints(s0):
    q=s0+1; k=(2*np.pi)**1.5
    rinf=quad(lambda t:2/(k*np.sqrt(q))*np.arctan(
        (s0+t)/np.sqrt(q*(q+2*(s0+t)))),0,1,epsabs=1e-12)[0]
    cinf=-quad(lambda t:(2*s0+t)/(k*(2*q-(1-t))*
        np.sqrt(3*q-2*(1-t))),0,1,epsabs=1e-12)[0]
    c0=1/(6*k*q**2.5)
    l1=s0*s0/6+s0/12+1/60
    k1=3*s0*s0/8+13*s0/60+1/20
    return dict(s0=s0,R_infinity=rinf,C_infinity=cinf,C_zero=c0,
        alpha_lower=rinf/(c0-cinf),alpha_upper=2*q*l1/k1,
        R_derivative_zero=2*l1/(k*q**2.5),C_derivative_zero=-k1/(k*q**3.5))


def main():
    endpoint_checks=[]
    for s0 in [.1,.5,1.,2.,4.,10.]:
        row=endpoints(s0)
        row['C_infinity_quadrature']=cubic_coefficient(s0,np.inf,160)
        row['C_zero_quadrature']=cubic_coefficient(s0,0,160)
        h=1e-5
        row['C_derivative_check']=(cubic_coefficient(s0,h)-cubic_coefficient(s0,0))/h
        assert abs(row['C_infinity_quadrature']-row['C_infinity'])<1e-9
        assert abs(row['C_zero_quadrature']-row['C_zero'])<1e-10
        assert abs(row['C_derivative_check']-row['C_derivative_zero'])<2e-6
        endpoint_checks.append(row)
    checks=[]
    for ell in [1.,30.]:
        for w in [.15,.25,.4]:
            result=solve_exact(w=w,ell=ell,s0=2.,x0=0,dx=.012,steps=1800,decompose=True)
            p=result.profile
            half=float(np.interp(0,p['x'],p['half_effort']))
            leading=w/np.sqrt(6*np.pi)
            c=cubic_coefficient(2.,ell)
            checks.append(dict(ell=ell,w=w,half_effort=half,cubic_prediction=leading+w**3*c,
                rescaled_coefficient=(half-leading)/w**3,C=c))
    out=ROOT/'results/theory/precision_conditions/third_order.json'
    out.write_text(json.dumps(dict(endpoints=endpoint_checks,effort_checks=checks,
        qualification='Exact coefficient calculations; strong-prize window is for the cubic approximation only.'),indent=2)+'\n')
    print(json.dumps(dict(endpoints=endpoint_checks,effort_checks=checks),indent=2))


if __name__=='__main__': main()
