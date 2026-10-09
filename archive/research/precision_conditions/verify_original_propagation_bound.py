"""Exact chain-rule algebra for ledger 118; not an output/regime certificate."""
from fractions import Fraction as F


def add(*polys):
    result = {}
    for p in polys:
        for key, val in p.items():
            result[key] = result.get(key, F(0))+val
    return {k: v for k, v in result.items() if v}


def mul(p, q):
    result = {}
    for key, val in p.items():
        for other, weight in q.items():
            new = tuple(a+b for a, b in zip(key, other))
            result[new] = result.get(new, F(0))+val*weight
    return {k: v for k, v in result.items() if v}


def scale(p, weight):
    return {k: v*weight for k, v in p.items() if v*weight}


def verify():
    zero = (0,)*6
    one = {zero: F(1)}
    variables = []
    for i in range(6):
        key = list(zero)
        key[i] = 1
        variables.append({tuple(key): F(1)})
    k, kmax, fp, ey, H, A = variables
    ry = add(ey, mul(fp, H))
    fpp = scale(mul(kmax, add(one, scale(mul(fp, fp), -1))), F(1, 2))
    barrier = add(scale(mul(k, add(scale(mul(ry, ry), 3), scale(mul(H, H), -1))), F(1, 4)),
                  scale(mul(mul(k, fp), mul(ry, H)), F(-1, 2)),
                  scale(mul(fpp, mul(H, H)), F(1, 2)))
    expected = add(scale(mul(k, mul(ey, ey)), F(3, 4)),
                   mul(mul(k, fp), mul(ey, H)),
                   scale(mul(add(kmax, scale(k, -1)),
                             mul(add(one, scale(mul(fp, fp), -1)), mul(H, H))), F(1, 4)))
    assert barrier == expected
    transform = add(scale(mul(mul(k, k),
                              add(scale(mul(A, A), 3), scale(mul(H, H), -1))), F(1, 4)),
                    scale(mul(mul(k, k), mul(A, A)), F(-1, 2)))
    assert transform == scale(mul(mul(k, k),
                                   add(mul(H, H), scale(mul(A, A), -1))), F(-1, 4))
    print("PASS: constant-maximum barrier and variable-kappa exponential transform.")
    print("Kernel domination follows from monotone Riccati kappa and positive propagation.")
    print("No finite-precision output envelope or optimal-design regime certified here.")


if __name__ == "__main__":
    verify()
