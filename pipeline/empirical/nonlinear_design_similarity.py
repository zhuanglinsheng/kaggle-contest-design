"""Original-game evaluator in similarity coordinates for thin terminal layers.

This is a numerical HJB solver, not the leading-order effort formula. Its
conservative gradient equations and output PDE were checked against the
independent value-HJB implementation. Screening outputs require refinement.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from pipeline.empirical.nonlinear_design import no_effort
from pipeline.theory.asymmetric_full_output import full_output

SOURCE = Path(__file__).parents[1]/'theory/nonlinear_similarity_kernel.cpp'


class SimilarityOutput:
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(SOURCE.read_bytes()+platform.machine().encode()).hexdigest()
        path = cache_dir/f'nonlinear_design_{identity}.so'
        # Never replace a loaded library. A source/architecture change gets a
        # new filename; concurrent workers serialize first-time compilation.
        with (cache_dir/'build.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                temp = path.with_suffix(f'.{os.getpid()}.tmp.so')
                subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC',
                                str(SOURCE), '-o', str(temp)], check=True, capture_output=True)
                temp.replace(path)
        self.library = ctypes.CDLL(str(path.resolve()))
        da = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        ip = ctypes.POINTER(ctypes.c_int)
        self.library.solve_hjb_similarity.argtypes = [ctypes.c_double]*5+[ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, da, ip]
        self.library.solve_hjb_similarity.restype = ctypes.c_int
        self.source_sha256 = hashlib.sha256(SOURCE.read_bytes()+Path(__file__).read_bytes()).hexdigest()

    def evaluate(self, draw, design, *, spacing=.04, steps=256,
                 full_nodes=2001, quadrature=64, max_cells=30_000_000):
        baseline = no_effort(draw, design.horizon)
        if design.prize == 0:
            return dict(B=baseline, budget=0., method='analytic_zero_effort')
        precision = draw.observed_precision if design.precision is None else design.precision
        if math.isinf(precision):
            r = full_output(design.prize, design.horizon, draw.sigma,
                            draw.c_i, draw.c_j, draw.S0, draw.mu0,
                            nodes=full_nodes, cutoff=24., quadrature=quadrature)
            return dict(B=r['B'], budget=r['tail_budget']+r['boundary_budget'], method='nonlinear_full_disclosure')
        scale = draw.sigma*math.sqrt(design.horizon)
        h = draw.S0/(scale*scale)
        nu = precision*draw.sigma**2*design.horizon**2
        steady = 1/math.sqrt(nu)
        z = math.tanh(math.sqrt(nu))
        terminal = steady*(h+steady*z)/(steady+h*z)
        wi = design.prize/(draw.c_i*draw.sigma**2)
        wj = design.prize/(draw.c_j*draw.sigma**2)
        extent = max(8., math.sqrt(2*max(wi,wj))+6,
                     abs(draw.mu0/scale)/math.sqrt(1+terminal)+6*math.sqrt(h/(1+terminal))+1,
                     8*math.sqrt((1+h)/(1+terminal)))
        if 2*math.ceil(extent/spacing)+1 > 200000:
            raise RuntimeError('Similarity state grid exceeds explicit budget')
        values = np.empty(3)
        work = ctypes.c_int()
        code = self.library.solve_hjb_similarity(wi, wj, h, nu, spacing, steps,
                   extent, max(4000,steps*20), draw.mu0/scale, values, ctypes.byref(work))
        if code < 1:
            raise RuntimeError(f'Similarity nonlinear HJB failure {code}; steps={work.value}')
        if max(abs(values[1]-1), abs(values[2]-1)) > 1e-4:
            raise RuntimeError('Nonlinear gradient loses probability mass')
        value = baseline+scale*values[0]
        if not math.isfinite(value) or value < baseline-1e-8*max(1., baseline):
            raise RuntimeError('Invalid nonlinear output')
        return dict(B=value, budget=0., method='nonlinear_similarity_HJB',
                    dx=spacing, steps=steps, work_steps=work.value,
                    probability_mass_i=float(values[1]), probability_mass_j=float(values[2]))
