"""Verify the exact signed-incentive symmetry of the original clock PDE.

Polynomial identities are checked by integer arithmetic. This does not prove
uniqueness, analytical regularity, or the sign of any precision comparison.
"""

from itertools import product


def player_flux(a, h):
    return 3*a*a-h*h, 2*a*h  # common denominator four


def verify():
    for w, a, h, c in product(range(-3, 4), repeat=4):
        r, d = player_flux(a, h)
        transformed_r, transformed_d = player_flux(-a, h)
        assert (-w)*transformed_r == -w*r
        assert (-w)*transformed_d == w*d
        assert (-w)*(-a) == w*a  # exact organizer drift is even
        assert (-w)*h == -w*h   # average-effort source is odd
        assert (-w)*(-2*a*c+h) == 2*w*a*c-w*h
    print("Verified exact player symmetry and organizer even/odd channel identities.")
    print("No precision ordering or external joint-design threshold is asserted.")


if __name__ == "__main__":
    verify()
