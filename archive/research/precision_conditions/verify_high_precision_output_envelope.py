"""Check the rational constants in research-note Section 17.

The continuous HJB comparisons are analytic arguments in the note;
these checks verify their constant arithmetic, not an equilibrium or optimum.
"""

from fractions import Fraction as F


def main():
    # e > 65/24 and 25/8 < pi < 22/7 give these elementary bounds.
    assert F(48, 65) < F(43, 50) ** 2
    assert 2 * (F(43, 50) + F(4, 5)) < F(10, 3)
    assert F(88, 7) < F(71, 20) ** 2

    barrier = F(51, 50)
    crossing_bound = F(4, 5) + F(5, 24) * barrier**2
    assert crossing_bound == F(4067, 4000) < barrier

    cosh_coefficient = F(12, 47)
    source_bound = barrier / 4 + F(5, 24) * barrier**2
    output_coefficient = (
        cosh_coefficient / 2
        + (1 + cosh_coefficient / 4) * F(71, 20) * source_bound
    )
    assert output_coefficient == F(143577, 75200) < 2
    print("All exact constant checks passed.")


if __name__ == "__main__":
    main()
