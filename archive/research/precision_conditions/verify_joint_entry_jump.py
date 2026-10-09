"""Exact rational certificate for the moderate-curvature entry example.

No equilibrium approximation or floating-point comparison is used.
The exact zero-signal output lower bound is proved in the paper.
"""

from fractions import Fraction as F


def verify():
    alpha = F(1, 50)
    deadline = F(27, 100)
    initial_deadline = F(1, 4)
    exposure = alpha * deadline / (1 + deadline)
    assert exposure == F(27, 6350) < F(1, 4)

    # d*b = 2/5 and c_entry = d/2. Normalize payoffs by 1/b.
    normalized_expense = F(2, 5) * (
        alpha / 2 + deadline - initial_deadline
    )
    assert normalized_expense == F(3, 250)
    a = 1 - normalized_expense
    inverse_output_ratio_squared = F(125, 127) / (1 + exposure) ** 2
    square_gap = a * a - inverse_output_ratio_squared
    assert square_gap == F(531114161, 2541633062500)
    assert 0 < inverse_output_ratio_squared < 1 and 0 < a < 1
    # a - sqrt(z) = (a*a-z)/(a+sqrt(z)) > square_gap/2.
    assert square_gap / 2 > F(1, 10000)

    # A cost increase <= 1/(400*b) costs <= 1/(20000*b).
    assert alpha / 400 == F(1, 20000)
    assert F(1, 10000) - alpha / 400 == F(1, 20000)
    print("Verified: positive prize beats every zero-prize design at and above entry.")
    print("Exposure:", exposure)
    print("Exact squared comparison gap:", square_gap)
    print("Uniform normalized payoff margin: strictly above 1/20000.")


if __name__ == "__main__":
    verify()
