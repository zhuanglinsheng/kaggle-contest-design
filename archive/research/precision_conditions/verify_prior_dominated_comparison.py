"""Exact constants for the new minimal-scope global comparison."""
from fractions import Fraction as F
from math import factorial
from verify_gaussian_jet_cap import exp_upper


def verify():
    k, wmax, mu = F(1, 100), F(63, 250), F(18, 25)
    # Entire low-diffusion contests have e<=wmax*2/99<.01.
    assert 2*wmax/99 < F(1, 100)
    e, eta, h = F(1,100), F(7,4), F(2,5)+F(1,400)
    radius = eta*e*h
    # Psi=.4 exp(r^2/2)+.32r/(1-r^2/3).
    increment = F(2,5)*(radius**2/2)*exp_upper(radius**2/2)/e
    increment += F(8,25)*eta*h/(1-radius**2/3)
    assert increment < F(1,4)
    # The robust supersolution absorbs both centered-gradient errors;
    # this doubles the old correction but keeps it below .02 downstream.
    assert F(1,2)*(F(1,2)+F(243,1000)*e) < F(63,250)
    assert F(101,99) < F(51,50)**2
    assert F(251,100)*F(51,50)*F(2,99)*F(63,250) < F(1,50)

    # Diffusion >=mu sigma^2: an elementary Gaussian integral gives
    # ||A1||_G <= alpha/a_min; the simpler ball is E<=6w^2, A<=3w.
    K, eh, imbalance = F(7,20), 6*wmax*wmax, 3*wmax
    assert K*imbalance*(2+eh)/2 < eh
    assert K+K*(3*imbalance**2+eh*(4+eh))/4 < imbalance
    assert K*imbalance/2+K*(2+eh)/2 < 1
    assert K*(4+2*eh)/4+3*K*imbalance/2 < 1
    assert (2+6*wmax*wmax)/(4*mu) < F(83,100)
    assert (1+wmax*wmax*(51+36*wmax*wmax)/4)/(3*mu) < F(86,100)
    assert K*(imbalance+2+eh)/2 < F(55,100)
    assert K*(4+2*eh+6*imbalance)/4 < F(82,100)
    source = 3*wmax+F(3,2)+F(729,1000)*wmax
    assert source < F(5,2)
    cosh = sum(K**(2*j)/factorial(2*j) for j in range(5))
    cosh += K**10/factorial(10)/(1-K*K/(11*12))
    factor = (1+cosh)/2  # cosh(K/2)^2
    assert factor < F(26,25)
    assert F(5,2)*factor < F(13,5) < F(29,10)

    # The ideal-budget output comparison is an exact factorization:
    # (.5-(1-x)/40)-x[(1+x)/4+29(1-x)(1+x)^2/160]
    # =(1-x)^2(29x^2+87x+76)/160.
    def multiply(p, q):
        out = [F(0)]*(len(p)+len(q)-1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                out[i+j] += a*b
        return out
    assert multiply([1,-2,1],[76,87,29]) == [76,-65,-69,29,29]
    assert k/(2*(1+F(99,100))**2) < F(1,40)
    # A fixed local neighborhood exists, e.g. gamma0=10000, C=2.9.
    gamma = F(10000)
    determinant = gamma*k*(F(1,20)-k/4)-2*k/5-k*k/2-F(39,10)
    f_xx = -((gamma+2)*k+2)/4
    f_xw = 1-(gamma+2)*k/2
    f_ww = F(29,5)-(gamma+1)*k
    assert determinant == f_xx*f_ww-f_xw*f_xw
    assert determinant > 0
    assert -((gamma+2)*k+2)/4 < 0
    # The competitive budget fits strictly within w+x^2/4<.252.
    assert F(100001,100000)*(F(1,4)+1/(F(1000000)*k)) < wmax
    print("PASS: all low-diffusion output excess coefficients <.02.")
    print("PASS: simple Gaussian ball E<=6w^2, A<=3w is invariant and contracting.")
    print("PASS: all high-diffusion output excess coefficients <2.6.")
    print("PASS: common coefficient 2.9; ideal-budget factorization and local Hessian.")
    print("Scope: scalar certificates; compact payoff and equilibrium arguments are in the proof.")


if __name__ == "__main__":
    verify()
