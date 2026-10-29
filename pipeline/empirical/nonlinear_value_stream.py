"""Compiled streaming version of the existing adaptive value-HJB/output PDE.

The Python reference's grid, CN/Heun steps, float32 policies and constraints
are retained. This backend is cross-checked before use in design production.
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
import json
from dataclasses import asdict
from functools import lru_cache
import numpy as np
from scipy.special import ndtr
from pipeline.synthetic.nonlinear import Primitives, variance
from pipeline.empirical.nonlinear_design import no_effort
from pipeline.empirical import nonlinear_design as reference

from pipeline import solver_cache

SOURCE=Path(__file__).parents[1]/'theory/nonlinear_value_stream.cpp'


class ValueStream:
    def __init__(self, build_dir):
        folder=Path(build_dir);folder.mkdir(parents=True,exist_ok=True)
        identity=hashlib.sha256(SOURCE.read_bytes()+platform.machine().encode()).hexdigest()
        path=folder/f'nonlinear_value_{identity}.so'
        with (folder/'build.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not path.exists():
                temp=path.with_suffix(f'.{os.getpid()}.tmp.so')
                subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(SOURCE),'-o',str(temp)],check=True,capture_output=True)
                temp.replace(path)
        self.library=ctypes.CDLL(str(path.resolve()))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.library.original_value_stream.argtypes=[ctypes.c_int]*2+[ctypes.c_double]*5+[array]*7
        self.library.original_value_stream.restype=ctypes.c_int
        self.source_sha256=hashlib.sha256(SOURCE.read_bytes()+Path(__file__).read_bytes()).hexdigest()

    def evaluate(self,draw,design,resolution,*,max_work_cells=None):
        precision=draw.observed_precision if design.precision is None else design.precision
        if design.prize<=0 or not math.isfinite(precision):
            raise ValueError('Streaming value solver requires positive prize and finite precision')
        p=Primitives(T=design.horizon,theta=design.prize,c_i=draw.c_i,c_j=draw.c_j,
                     sigma=draw.sigma,lamb=precision,S0=draw.S0,mu0=draw.mu0)
        scale=math.sqrt(draw.S0+draw.sigma**2*design.horizon)
        dy=min(resolution.spacing*scale,.2*math.sqrt(float(variance(p.T,p))))
        uniform=np.linspace(0.,p.T,resolution.steps+1)
        rate=p.lamb*p.S0
        layer_end=min(p.T,30./rate)
        smallest=min(layer_end/5000.,p.T/(resolution.steps*100.))
        time=np.unique(np.r_[uniform,0.,np.geomspace(smallest,layer_end,240)]) if smallest>0 and layer_end>smallest else uniform
        width=np.ceil(resolution.extent*scale/dy)*dy
        y=np.arange(-width,width+dy/2,dy)
        cap=resolution.max_policy_cells if max_work_cells is None else max_work_cells
        reference_cells=(resolution.steps+242)*(2*math.ceil(resolution.extent*scale/dy)+1)
        if (reference_cells if max_work_cells is None else len(time)*len(y))>cap:
            raise RuntimeError('Reference policy-grid memory budget / streaming work limit exceeded')
        if abs(draw.mu0)+6*math.sqrt(draw.S0)>=resolution.extent*scale:
            raise RuntimeError('Initial state too close to the finite state boundary')
        middle=(time[:-1]+time[1:])/2
        smid=variance(middle,p)
        diffusion=np.ascontiguousarray(p.lamb*smid*smid)
        qmid=np.ascontiguousarray(smid+p.sigma*p.sigma*(p.T-middle))
        terminal=np.ascontiguousarray(np.column_stack((ndtr(y/np.sqrt(variance(time[-1],p))),ndtr(-y/np.sqrt(variance(time[-1],p))))).ravel())
        gain=np.empty(len(y));diagnostics=np.zeros(4)
        code=self.library.original_value_stream(len(time),len(y),dy,float(y[1]-y[0]),p.theta,p.c_i,p.c_j,
                                               time,y,diffusion,qmid,terminal,gain,diagnostics)
        if code==-2:
            raise RuntimeError(f'Non-monotone value gradient: {diagnostics[0]}')
        if code!=1:
            raise RuntimeError(f'Streaming nonlinear value HJB failure {code}')
        if diagnostics[2]<-1e-6 or diagnostics[3]>1+1e-6:
            raise RuntimeError('Nonlinear value outside probability bounds')
        baseline=no_effort(draw,design.horizon)
        value=baseline+float(np.interp(draw.mu0,y,gain))
        if not math.isfinite(value) or value<baseline-1e-8*max(1.,baseline):
            raise RuntimeError('Invalid original-model expected output')
        return dict(B=value,boundary_budget=0.,tail_budget=0.,
                    method='streaming_original_value_HJB_and_output_PDE',
                    diagnostics=dict(dy=dy,nodes_y=len(y),nodes_t=len(time),
                                     min_raw_policy=float(diagnostics[0]),max_policy=float(diagnostics[1]),
                                     value_min=float(diagnostics[2]),value_max=float(diagnostics[3]),
                                     scheme='adaptive-time Crank--Nicolson/Heun; streaming compiled'))


@lru_cache(maxsize=8)
def _engine(build_dir):
    return ValueStream(build_dir)


def checked_output(draw,design,*,cache_dir,resolution):
    """Reference-compatible two-grid check, reusing existing reference records.

    New points use the streaming implementation. Old records remain valid
    because both backends use the same adaptive CN/Heun finite-grid equations.
    Failures are retained, rather than averaged away or converted to success.
    """
    sources={str(Path(f).resolve()):reference.digest(f) for f in
             (reference.__file__,reference.joint_design_exact.__file__,
              reference.asymmetric_full_output.__file__,
              Path(reference.joint_design_exact.__file__).parents[1]/'synthetic/nonlinear.py')}
    signature=dict(draw=asdict(draw),design=design.record(),resolution=asdict(resolution),sources=sources)
    cache_dir=Path(cache_dir)
    old_key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    old_path=cache_dir/f'{old_key}.json'
    record=solver_cache.get(old_path)
    if record is not None:
        if record['signature']!=signature:
            raise RuntimeError('Changed original value-HJB cache signature')
        return record
    sources.update({str(Path(f).resolve()):reference.digest(f) for f in [SOURCE,__file__]})
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    path=cache_dir/f'{key}.json'
    record=solver_cache.get(path)
    if record is not None:
        if record['signature']!=signature:
            raise RuntimeError('Changed compiled value-HJB cache signature')
        return record
    trials=[]
    for level in [resolution,resolution.refined()]:
        try:
            precision=draw.observed_precision if design.precision is None else design.precision
            if design.prize==0 or math.isinf(precision):
                value=reference.nonlinear_output(draw,design,level)
            else:
                value=_engine(str(cache_dir.parent/'value_stream_build')).evaluate(draw,design,level)
            trials.append(dict(resolution=asdict(level),**value))
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as error:
            trials.append(dict(resolution=asdict(level),error=str(error)))
    record=dict(signature=signature,trials=trials,resolved=all('B' in t for t in trials))
    if record['resolved']:
        record.update(B=trials[-1]['B'],diagnostic_radius=3*abs(trials[-1]['B']-trials[0]['B'])
                      +trials[-1]['boundary_budget']+trials[-1]['tail_budget'])
    reference.save(path,record)
    return record
