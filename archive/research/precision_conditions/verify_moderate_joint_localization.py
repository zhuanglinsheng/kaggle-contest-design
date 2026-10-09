"""Rational constants in the moderate-curvature global localization theorem."""

from fractions import Fraction as F


def verify():
    assert F(4, 5)**5 < F(19, 20)
    assert F(9, 5)/2 == F(9, 10)
    assert F(2, 8)+F(1, 2) == F(3, 4)
    assert F(3, 4)/F(9, 10) == F(5, 6)
    assert F(3, 2)**2 == F(9, 4)
    assert F(3, 20)-F(21, 40)**2/2 == F(39, 3200) > 0
    assert F(50, 51)**8 < F(19, 20)**2
    q = F(51, 26)
    assert (F(11, 40)+q/6)**2-F(9, 20)*(q-1) == -F(19031, 270400)
    assert (F(11, 40)+F(2, 6))/3-F(9, 20) < 0
    assert F(5, 6)/F(9, 10) == F(25, 27)
    assert F(4, 3)*F(26, 25)+F(1, 3) == F(43, 25)
    q = F(5, 3)
    assert (F(3, 10)+q/8)**2-F(2, 5)*(q-1) == -F(119, 14400)
    assert (F(3, 10)+F(2, 8))/4-F(2, 5) < 0
    assert (F(1, 2)+F(51, 26)/8)/F(9, 10) == F(775, 936)
    assert ((F(51, 26)-1)/2)/F(9, 10) == F(125, 234)
    assert F(775, 936)-F(125, 234)/4 == F(5, 6)-F(5, 9)/4
    assert F(125, 234)/F(775, 936) == F(20, 31)
    assert F(125, 234)/(F(775, 936)-F(1, 2)) == F(500, 307)
    print("Verified: curvature 7 attainment; curvature 9 exposure < 2/9 and coupled sorting domain.")


if __name__ == '__main__':
    verify()
