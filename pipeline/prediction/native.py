"""Portable build of the exact recovered full-path prediction evaluator."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
from pipeline.publication import ROOT, digest
from pipeline.storage import BUILD
from pipeline.empirical.estimate import stan_data
from pipeline.prediction.posterior import counts_for


class FullPath:
    def __init__(self, build=None):
        source=Path(__file__).with_name('full_path.cpp')
        build=Path(build) if build else BUILD/'prediction'
        build.mkdir(parents=True,exist_ok=True)
        binary=build/f'full_path_{digest(source)}.so'
        if not binary.exists():
            import os, tempfile
            with tempfile.NamedTemporaryFile(dir=build,suffix='.so',delete=False) as temp: staged=Path(temp.name)
            try:
                subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(staged)],check=True,capture_output=True)
                os.replace(staged,binary)
            finally: staged.unlink(missing_ok=True)
        self.library=ctypes.CDLL(str(binary));self.function=self.library.batch_forecast_path
        D=ctypes.POINTER(ctypes.c_double);I=ctypes.POINTER(ctypes.c_int)
        self.function.argtypes=[ctypes.c_int,ctypes.c_int,D,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                                ctypes.c_double,ctypes.c_double,ctypes.c_double,D,I,I,D,D]
        self.function.restype=None

    def evaluate(self, raw, columns, cutoff, free=False):
        data=stan_data(raw,'full');N=data['N_Delta']
        if not data['count_start_idx']-1<cutoff<N:raise ValueError('Invalid training cutoff')
        names=['c_i','c_j','sigma','lambda','mu_0','log_r']+(['deadline_power','gap_log_scale'] if free else [])
        points=np.column_stack([np.log(columns[k]) if k in names[:4] else columns[k] for k in names])
        points=np.ascontiguousarray(points,dtype=np.float64)
        held=np.ascontiguousarray(data['hat_y'],dtype=np.float64)
        counts=np.ascontiguousarray(counts_for(raw),dtype=np.int32)
        train=np.empty(len(points));future=np.empty((len(points),N-cutoff))
        D=ctypes.POINTER(ctypes.c_double);I=ctypes.POINTER(ctypes.c_int)
        self.function(len(points),len(names),points.ctypes.data_as(D),cutoff,N,data['count_start_idx']-1,
            data['Delta2f'],data['T_contest'],data['theta'],held.ctypes.data_as(D),
            counts[0].ctypes.data_as(I),counts[1].ctypes.data_as(I),train.ctypes.data_as(D),future.ctypes.data_as(D))
        return train,future
