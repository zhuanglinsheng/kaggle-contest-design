"""Rational constants in the exact finite-incentive initial bridge gain."""

from fractions import Fraction as F


def verify():
    assert F(32, 31)**2 < F(16, 15)
    assert F(47, 46) < F(21, 20)
    assert F(47, 48)*F(15, 8)/4/3 == F(235, 1536)
    assert F(235, 1536)-F(7, 80) > F(1, 16)
    assert F(441, 1200)+F(21, 80) < 1
    assert F(1, 16)/F(5, 4) == F(1, 20)
    assert F(5, 8)/(1-F(5, 8)**2/6) == F(240, 359) < F(3, 4)
    assert (1+1/(1-F(5, 8)**2/6))/2 == F(743, 718) < F(21, 20)
    assert F(512, 487)**2 < F(10, 9)
    assert 1-F(5, 16)**2/3 > F(29, 30)
    assert F(29, 30)*F(9, 5)/12 == F(29, 200)
    assert F(21, 40)*F(5, 8)/F(25, 8) == F(21, 200)
    assert F(29, 200)-F(21, 200) == F(1, 25)
    assert F(1, 25)/(1+F(5, 8)**2) == F(64, 2225) > F(1, 35)
    assert F(125, 234)/(F(775, 936)-F(5, 8)) == F(50, 19)
    print("Verified: initial margin w^2*G/20 for w <= 1/2; w^2*G/35 for w <= 5/8.")


if __name__ == '__main__':
    verify()
