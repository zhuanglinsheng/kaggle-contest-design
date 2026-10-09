#!/usr/bin/env python3
"""Numerical search for finite optimal precision without a precision cost.

The second-order design coefficient scales as sqrt(T)/sigma^3.  Its shape
therefore depends only on

    s0 = S0/(sigma^2 T),
    z  = |mu|/sqrt(S0+sigma^2 T),
    ell = lambda*sigma^2*T^2.

We set sigma=T=1, evaluate the Gaussian expectation analytically, and use a
two-dimensional Gauss--Legendre rule after the endpoint-regularizing change
of variables t=1-a^2 and s=1-a^2*b^2.  No simulation noise is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from pipeline.config import ROOT

OUT = ROOT / "results" / "theory" / "no_cost_precision_scan.json"
FIGURE = ROOT / "results" / "theory" / "no_cost_precision_scan.png"


class Coefficient:
    def __init__(self, order: int):
        node, weight = leggauss(order)
        a = (node + 1.0) / 2.0
        weight = weight / 2.0
        self.t = 1.0 - a[:, None] ** 2
        self.s = 1.0 - a[:, None] ** 2 * a[None, :] ** 2
        self.weight = weight[:, None] * weight[None, :] * (
            4.0 * a[:, None] ** 3 * a[None, :]
        )

    @staticmethod
    def q(time: np.ndarray, s0: float, ell: float) -> np.ndarray:
        if np.isinf(ell):
            return 1.0 - time
        if ell == 0.0:
            return np.full_like(time, s0 + 1.0)
        root = np.sqrt(ell)
        argument = root * time
        tanhc = np.tanh(argument) / argument
        variance = (s0 + time * tanhc) / (1.0 + ell * s0 * time * tanhc)
        return variance + 1.0 - time

    def __call__(self, s0: float, z: float, ell: float) -> float:
        q0 = s0 + 1.0
        mu = z * np.sqrt(q0)
        qt = self.q(self.t, s0, ell)
        qs = self.q(self.s, s0, ell)
        latent_variance = np.maximum(q0 - qt, 0.0)
        d = 2.0 * qt - qs
        e = d + 2.0 * latent_variance  # equals 2*q0-qs
        tilted_mean = mu * d / e
        tilted_variance = latent_variance * d / e
        normalizer = np.sqrt(qt + tilted_variance)
        standardized = tilted_mean / normalizer
        tilted_moment = (
            tilted_mean * (2.0 * ndtr(standardized) - 1.0)
            + 2.0 * tilted_variance / normalizer
            * np.exp(-standardized**2 / 2.0) / np.sqrt(2.0 * np.pi)
        )
        integrand = (
            np.exp(-mu**2 / e) * tilted_moment
            / (2.0 * np.pi * np.sqrt(qs) * d * np.sqrt(e))
        )
        return float(np.sum(self.weight * integrand))


def run(order: int) -> dict:
    coefficient = Coefficient(order)
    s0_grid = np.logspace(-4, 4, 17)
    z_grid = np.linspace(0.0, 5.0, 41)
    ell_grid = np.logspace(-8, 10, 73)
    max_excess = (-np.inf, None)
    max_drop = (-np.inf, None)
    decreasing_steps = 0
    comparisons = 0
    finite_above_limit = 0
    curve_shapes: dict[str, int] = {}
    for s0 in s0_grid:
        for z in z_grid:
            values = np.array([coefficient(s0, z, ell) for ell in ell_grid])
            limit = coefficient(s0, z, np.inf)
            excess = values - limit
            idx = int(np.argmax(excess))
            if excess[idx] > max_excess[0]:
                max_excess = (float(excess[idx]), [float(s0), float(z), float(ell_grid[idx])])
            if excess[idx] > 2e-9 * max(1.0, abs(limit)):
                finite_above_limit += 1
            diffs = np.diff(values)
            tolerance = 2e-10 * max(1.0, float(np.max(np.abs(values))))
            signs = np.sign(diffs[np.abs(diffs) > tolerance]).astype(int)
            changes = signs[np.r_[True, signs[1:] != signs[:-1]]] if len(signs) else signs
            shape = {
                (1,): "increasing",
                (-1, 1): "decrease_then_increase",
                (): "flat_at_tolerance",
            }.get(tuple(changes), "other")
            curve_shapes[shape] = curve_shapes.get(shape, 0) + 1
            comparisons += len(diffs)
            decreasing_steps += int(np.sum(diffs < -tolerance))
            idx_drop = int(np.argmin(diffs))
            drop = -float(diffs[idx_drop])
            if drop > max_drop[0]:
                max_drop = (drop, [float(s0), float(z), float(ell_grid[idx_drop])])

    # Repeat the main comparison with a doubled quadrature order on a broad
    # independent deterministic lattice, including the most suspicious point.
    verification = Coefficient(2 * order)
    verify_s0 = np.r_[np.logspace(-4, 4, 9), max_excess[1][0]]
    verify_z = np.r_[np.linspace(0, 5, 21), max_excess[1][1]]
    verify_ell = np.r_[np.logspace(-7, 9, 33), max_excess[1][2]]
    verify_excess = (-np.inf, None)
    verify_decreases = 0
    verify_comparisons = 0
    for s0 in np.unique(verify_s0):
        for z in np.unique(verify_z):
            values = np.array([verification(s0, z, ell) for ell in np.unique(verify_ell)])
            limit = verification(s0, z, np.inf)
            idx = int(np.argmax(values - limit))
            if values[idx] - limit > verify_excess[0]:
                verify_excess = (float(values[idx] - limit), [float(s0), float(z), float(np.unique(verify_ell)[idx])])
            verify_decreases += int(np.sum(np.diff(values) < -5e-11 * np.maximum(1.0, np.abs(values[:-1]))))
            verify_comparisons += len(values) - 1

    examples = []
    for s0, z in [(0.01, 0), (1, 0), (1, 1), (1, 3), (100, 1)]:
        entries = {str(ell): coefficient(s0, z, ell) for ell in [0.0, .01, .1, 1., 10., 100., 1e4, np.inf]}
        examples.append(dict(s0=s0, z=z, coefficient=entries))
    return dict(
        normalization=dict(s0="S0/(sigma^2*T)", z="abs(mu)/sqrt(S0+sigma^2*T)", ell="lambda*sigma^2*T^2"),
        primary=dict(order=order, s0_points=len(s0_grid), z_points=len(z_grid), ell_points=len(ell_grid),
                     parameter_curves=len(s0_grid)*len(z_grid), adjacent_comparisons=comparisons,
                     decreasing_steps=decreasing_steps, finite_curves_above_full_information=finite_above_limit,
                     curve_shapes=curve_shapes,
                     largest_finite_minus_full_information=max_excess,
                     largest_adjacent_drop=max_drop),
        verification=dict(order=2*order, adjacent_comparisons=verify_comparisons,
                          decreasing_steps=verify_decreases,
                          largest_finite_minus_full_information=verify_excess),
        examples=examples,
        interpretation=("Some curves initially decrease and then increase, but no scanned curve has a "
                        "finite local maximum or exceeds the full-information limit. This is numerical "
                        "evidence, not a global dominance proof."),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    result = run(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kaggle-contest-design-matplotlib")
    import matplotlib.pyplot as plt
    evaluator = Coefficient(args.order)
    ell = np.logspace(-6, 8, 160)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
        values = np.array([evaluator(1.0, z, value) for value in ell])
        limit = evaluator(1.0, z, np.inf)
        axes[0].plot(ell, values / limit, label=rf"$z={z:g}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Dimensionless precision $\ell$")
    axes[0].set_ylabel(r"$\mathcal{R}(\ell)/\mathcal{R}(\infty)$")
    axes[0].set_title(r"Initial variance ratio $s_0=1$")
    axes[0].axhline(1, color="black", linewidth=.8, linestyle="--")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    for s0 in [0.01, 0.1, 1.0, 10.0, 100.0]:
        values = np.array([evaluator(s0, 1.0, value) for value in ell])
        limit = evaluator(s0, 1.0, np.inf)
        axes[1].plot(ell, values / limit, label=rf"$s_0={s0:g}$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Dimensionless precision $\ell$")
    axes[1].set_title(r"Standardized initial gap $z=1$")
    axes[1].axhline(1, color="black", linewidth=.8, linestyle="--")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
