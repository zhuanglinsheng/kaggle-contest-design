"""Full-Q analytic strip without differentiated Nash equations."""
from fractions import Fraction as F
from math import factorial
from verify_gaussian_jet_cap import exp_upper


def verify():
    x = F(1633, 2000)
    assert x*x > F(2, 3)
    atanh_upper = sum(x**(2*j+1)/F(2*j+1) for j in range(24))
    atanh_upper += x**49/(49*(1-x*x))
    assert atanh_upper < F(1147, 1000)
    assert sum(F(1, 2)**j/factorial(j) for j in range(8)) > F(1000, 607)
    assert F(607, 1000)*F(1147, 1000)/F(157, 50) < F(111, 500)
    # The forcing constant is e^(-1/2)atanh(sqrt(2/3))/pi <.222.
    k, radius = F(7, 20), F(9, 20)
    phase = exp_upper(radius*radius/4)
    bilinear = k*phase*(F(221, 100)+2*radius)
    h0 = F(4, 5)*phase
    forcing = F(111, 500)*k*exp_upper(3*radius*radius/2)
    eh, imbalance = F(21, 200), F(24, 125)
    assert bilinear*imbalance*(h0+eh)/2 < eh
    assert forcing+bilinear*(3*imbalance**2+eh*(2*h0+eh))/4 < imbalance
    assert bilinear*imbalance/2+bilinear*(h0+eh)/2 < 1
    assert bilinear*(2*h0+2*eh)/4+3*bilinear*imbalance/2 < 1
    # The real restriction also has the tighter zero-order bounds.
    real_k = k*F(221, 100)
    assert real_k*F(11, 100)*(F(4, 5)+F(1, 25))/2 < F(1, 25)
    assert F(111, 500)*k+real_k*(3*F(11, 100)**2+F(1, 25)*(F(8, 5)+F(1, 25)))/4 < F(11, 100)
    # One Cauchy tail for all positive-order derivatives.
    r, eta = F(13, 50), F(7, 4)
    gamma_half = F(3991, 10000)*exp_upper(r*r/2)
    gamma_half += F(50, 157)*r/(1-r*r/3)-F(3, 40)*r
    psi = gamma_half+(F(1, 25)+eta*F(11, 100))/2
    tail = phase*(eh+eta*imbalance)/2
    assert tail < F(29, 125)
    psi += F(29, 125)*r/(radius-r)
    assert psi < F(23, 25)
    assert eta*F(63, 250)*F(16, 25)*F(23, 25) < r
    print("PASS: forcing constant <.222; full-Q strip radius .45.")
    print("PASS: centered caps .105z^2,.192z, contraction, real caps .04z^2,.11z.")
    print("PASS: one Cauchy tail gives Psi(.26)<.92, without derivative inputs.")


if __name__ == "__main__":
    verify()
