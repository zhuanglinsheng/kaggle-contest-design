"""Exact rational chain-rule cancellation in the original-clock sensitivity.

This checks algebra only, not the sign of the precision derivative.
"""
from fractions import Fraction as F


def mv(matrix, vector):
    return [sum(x*y for x, y in zip(row, vector)) for row in matrix]


def curvature(v):
    a, b, c = v
    return [(3*a*a-b*b)/2, a*b, 2*a*c]


def verify():
    for n in range(1, 40):
        a, b, c = F(n, 43), F(2*n+1, 47), F(3-n, 53)
        v = [F(n-5, 17), F(7-n, 19), F(2*n, 23)]
        h = [F(n, 11), F(3-n, 13), F(9-n, 29)]
        k = F(n, 31)
        jac = [[3*a/2, -b/2, 0], [b/2, a/2, 0], [c, F(1, 2), a]]
        av, ah, aa_v = mv(jac, v), mv(jac, h), mv(jac, mv(jac, v))
        f_q = mv(jac, [x/2+k*y for x, y in zip(h, av)])
        cv = curvature(v)
        l_f = [x/2+y/2+k*z for x, y, z in zip(ah, cv, aa_v)]
        assert [x-y for x, y in zip(f_q, l_f)] == [-x/2 for x in cv]
    for n in range(1, 40):
        q, root = F(n+2), F(2*n+3, 2)
        h = root*(q-root)/(2*q-root)
        assert 0 < h <= root/3
        for j in range(1, 20):
            k_squared = F(j, 23)
            ratio = 3*h-k_squared*h*h/(1+k_squared*h)
            assert ratio < 3*h <= root
    print("Verified nonlinear cancellation and obstruction constants; full precision-derivative signs remain open.")


if __name__ == "__main__":
    verify()
