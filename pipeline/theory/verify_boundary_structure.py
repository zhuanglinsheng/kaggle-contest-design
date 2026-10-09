"""Exact optimization audit; this is not a counterexample to the contest HJB.

Concavity of the reduced gross frontier and monotone effective incentives alone
do not imply a single switch between arbitrary precision labels.
"""
from fractions import Fraction as Q


def verify():
    # The nonlinear no-signal candidate in ledger 142 has exposure 1/5.
    # Curvature/delay elasticity >=45 is sufficient for both risk cases.
    assert Q(25, 27)**22 < Q(9, 46)
    assert Q(16, 44)+Q(2, 45) < 1
    assert Q(4, 5)+Q(1, 230)+Q(25, 27)**22 < 1
    # Ledger 145: lower-cost certification with exact no-signal exposure 6/25.
    assert Q(1, 4)-Q(6, 25) == Q(1, 100)
    assert Q(6, 25)/Q(1, 4) == Q(24, 25)
    assert Q(45, 46)*(1-Q(24, 25)) == Q(9, 230)
    assert Q(230, 9)*Q(25, 31)**44 < 1
    # (effective incentive, gross payoff, precision type)
    actions = [(Q(0), Q(0), "zero"), (Q(1), Q(3, 2), "finite"),
               (Q(2), Q(5, 2), "full"), (Q(3), Q(3), "finite")]
    slopes = [(b[1]-a[1])/(b[0]-a[0])
              for a, b in zip(actions, actions[1:])]
    assert slopes == [Q(3, 2), Q(1), Q(1, 2)]
    assert all(x > y for x, y in zip(slopes, slopes[1:]))
    probes = [(Q(1, 4), "finite", Q(3)),
              (Q(3, 4), "full", Q(2)),
              (Q(5, 4), "finite", Q(1)),
              (Q(7, 4), "zero", Q(0))]
    incentives = []
    for cost, label, incentive in probes:
        values = [gross-cost*alpha for alpha, gross, _ in actions]
        winners = [a for a, v in zip(actions, values) if v == max(values)]
        assert len(winners) == 1
        assert winners[0][0] == incentive and winners[0][2] == label
        incentives.append(incentive)
    assert all(a > b for a, b in zip(incentives, incentives[1:]))
    for cost, labels in [(Q(1, 2), {"finite", "full"}),
                        (Q(1), {"finite", "full"}),
                        (Q(3, 2), {"finite", "zero"})]:
        values = [gross-cost*alpha for alpha, gross, _ in actions]
        assert {a[2] for a, v in zip(actions, values) if v == max(values)} == labels
    # At a crossing, d/dc of the two optimized payoffs differs by
    # alpha_full-alpha_finite. The reverse crossing fails the tie-order test.
    for cost, derivative, ordered in [(Q(1, 2), Q(-1), True),
                                     (Q(1), Q(1), False)]:
        families = {}
        for label in ("finite", "full"):
            candidates = [a for a in actions if a[2] == label]
            maximum = max(gross-cost*alpha for alpha, gross, _ in candidates)
            families[label] = [alpha for alpha, gross, _ in candidates
                               if gross-cost*alpha == maximum]
        assert len(families["finite"]) == len(families["full"]) == 1
        af, ai = families["finite"][0], families["full"][0]
        assert ai-af == derivative
        assert (af > ai) == ordered
    print("Exact fractions verified: concave gross frontier, falling incentives, "
          "but finite -> full -> finite -> zero precision labels.")
    print("This refutes an optimization-only inference, not the original HJB model.")
    print("Ledger 142: both curvature-45 inequalities verified exactly.")
    print("Ledger 144: crossing derivatives and the reverse-crossing obstruction "
          "verified exactly.")
    print("Ledger 145: certification scaling and margins verified exactly.")


if __name__ == "__main__":
    verify()
