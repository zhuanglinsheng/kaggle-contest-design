"""Exact elementary certificates for ledger section 164's existence point.

This verifies constants, not the underlying equilibrium/compactness lemmas.
No HJB grid or numerical global optimization is used in the certificate.
"""

import math
from fractions import Fraction as F
from verify_low_precision_payoff_envelope import psi_upper


def main():
    # Weak Gaussian Nash branch: accumulated exposure 1/8.
    assert psi_upper(F(7, 4)*F(1, 8)*F(43, 100)) < F(43, 100)
    assert F(43, 100)-F(3991, 10000) < F(199, 500)/10
    # b=sqrt(36/(2*pi)) > 7/3, using pi < 22/7.
    assert F(36)/(2*F(22, 7)) > F(7, 3)**2
    x = F(21, 40)
    assert 1+x+x*x/2+x*x*x/6 > F(5, 3)
    # kappa=2/5-2^(-4/3) > 1/400.
    assert 16*F(159, 400)**3 > 1
    # R=3 exp(-2b) < 1/32.
    assert sum(F(14, 3)**j/math.factorial(j) for j in range(11)) > 96
    # At sigma=1/4, the full-disclosure payoff upper gain is < kappa R.
    assert F(484)/(3*36*32*256) < F(1, 400)
    print('PASS: exact candidate, weak-exposure, and full-disclosure constants.')
    b = math.sqrt(36/(2*math.pi))
    remaining = 3*math.exp(-2*b)
    cost = 2/36*(2*remaining/3)**1.5
    deadline = (cost*36/8)**(1/3)
    prize = cost*36/(8*deadline)
    print('Approximate display only: S0=36, sigma=1/4, c=', cost)
    print('Candidate theta=', prize, 'T=', deadline, 'lambda=0.')
    print('Candidate is a comparison design, NOT asserted to be the joint optimum.')


if __name__ == '__main__':
    main()
