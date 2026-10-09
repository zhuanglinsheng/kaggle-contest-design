"""One Gaussian cap and one endpoint exclude the short initial layer."""
from fractions import Fraction as F
from math import factorial


def verify():
    s = F(61, 100)**2
    # F(s)=-log(1-s)-.49s/(1-s)^2 has a single interior maximum.
    assert sum(s**j/F(j) for j in range(1, 10)) > F(49, 100)*s/(1-s)**2
    assert 2*s < F(249, 100)
    assert F(2, 5)*(1+s/(3*(1-s))) < F(12, 25)
    # atanh(sqrt(.72))<1.26; an elementary exponential lower bound suffices.
    root = F(849, 1000)
    assert root*root > F(18, 25)
    assert sum(F(63, 25)**j/factorial(j) for j in range(18)) > (1+root)/(1-root)
    factor, z, xmax, ratio = F(19, 10), F(1197, 500), F(523, 1000), F(49, 20)
    assert z == factor*F(63, 50)
    assert z*z/21 < xmax*xmax
    assert sum((2*z)**j/factorial(j) for j in range(18)) > (ratio+z)/(ratio-z)
    exposure = ratio*(F(63, 250)-F(1, 8)**2/4)
    assert exposure < F(61, 100)
    # The uniform time average cap is .48; the output bound increases in x.
    root_ratio, root_x = F(783, 500), F(181, 250)
    assert root_ratio*root_ratio > ratio
    assert root_x*root_x > F(1, 1000)+xmax
    coefficient = F(251, 100)*root_ratio*F(12, 25)
    assert 1-coefficient*(5*xmax*xmax/4+F(1, 1000)*xmax) > 0
    output = root_x*(1+coefficient*(F(63, 250)-xmax*xmax/4))
    assert output < F(49, 50)
    # S_delta/(sigma^2 delta)>2/3 follows from sqrt(.72)>.84 and A<1.26.
    assert F(21, 25)**2 < F(18, 25)
    assert F(21, 25)/F(63, 50) == F(2, 3)
    f = F(10, 19)
    assert f/(1-f/3) < F(16, 25)
    assert F(126, 125)/(1-f/3) < F(49, 40)
    print("PASS: analytic Gaussian cap .4/(1-e^2), e<=.61; time average <.48.")
    print("PASS: T<=1.9delta implies output/reference <", float(output), "<.98.")
    print("PASS: f<10/19, e_delta/w<.64, Q0/Qdelta<1.225.")


if __name__ == "__main__":
    verify()
