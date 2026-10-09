"""Scalar certificates for a uniform initial-layer bridge.

The analytic identities and comparison arguments are stated separately
in UNIFIED_INITIAL_BRIDGE.txt. No equilibrium grid is used here.
"""
from fractions import Fraction as F
from verify_gaussian_jet_cap import exp_upper
from verify_direct_deadline_buffer import verify as verify_deadline
from verify_mature_root_certificate import main as verify_mature
from verify_broad_gaussian_strip import verify as verify_strip


def verify():
    verify_deadline()
    verify_mature()
    verify_strip()
    radius, prize, eta = F(9, 20), F(63, 250), F(7, 4)

    # Gaussian component's coefficientwise majorant, plus the residual jet.
    tail = F(29, 125)
    r, m = F(13, 50), F(23, 25)
    exposure = prize*F(16, 25)
    gamma_half = F(3991, 10000)*exp_upper(r*r/2)
    gamma_half += F(50, 157)*r/(1-r*r/3)-F(3, 40)*r
    h0 = F(3991, 10000)+(F(1, 25)+eta*F(11, 100))/2
    psi = gamma_half+(F(1, 25)+eta*F(11, 100))/2
    psi += tail*r/(radius-r)
    assert psi < m
    assert eta*exposure*m < r

    delta = m-h0
    e_inc = 4*delta/eta**2
    j_inc = 2*(1+3/eta**2)*delta/eta
    e_bar = F(1, 25)+e_inc/2
    j_bar = F(11, 100)+j_inc/2
    i_bar = F(11, 200)+j_inc/3
    # L=sqrt(2*pi*Q0/Qdelta)<2.8; |G'|<.243/Q.
    L = F(14, 5)
    assert 2*F(22, 7)*F(49, 40) < L*L
    beta_cap = F(19, 125)
    assert F(9, 40) < (beta_cap*F(157, 50))**2
    g = F(16, 25)
    c_g = F(243, 1000)*L
    early = L*(e_bar/(2*prize)+j_bar*beta_cap/prize
               +F(2, 5)*g*j_bar*j_bar/2)+c_g*g*i_bar
    mature = F(32, 25)+c_g*g*j_bar*(1+F(32, 25)*prize)
    f = F(10, 19)
    combined = f*early+(1-f)*mature
    assert early > mature
    assert combined < F(23, 10)
    print("PASS: full Gaussian spatial strip and initial analytic cap.")
    print("PASS: unified output excess coefficient:", float(combined), "<2.3.")
    print("Scope: the bridge uses no mature derivative estimates.")


if __name__ == "__main__":
    verify()
