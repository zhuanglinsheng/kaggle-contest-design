# Task 3：已审查非线性策略/产出的调用接口

只定义审查副本接口；原精度稳态核与任意正精度固定初值核分别调用。所有输出必须可积且生成元通过正性检查；不在接口中裁剪、补值或伪造失败输出。科学接受仍需独立方法、网格和最佳回应检查。

```python
from pathlib import Path
import ctypes,math
import numpy as np
p=Path('docs/review/PaperJK9');out=p/'task3_direction_consistency_20261025'/'resources';da=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ip=ctypes.POINTER(ctypes.c_int);steady=ctypes.CDLL(str((out/'nonlinear_output_kernel.so').resolve()));general=ctypes.CDLL(str((out/'nonlinear_variable_diffusion_kernel.so').resolve()));steady.solve_hjb.argtypes=[ctypes.c_double]*4+[ctypes.c_int,ctypes.c_double,da,ip];steady.solve_hjb.restype=ctypes.c_int;steady.audit_expected_output.argtypes=[da,ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*3;steady.audit_expected_output.restype=ctypes.c_double;general.solve_hjb_variable.argtypes=[ctypes.c_double]*5+[ctypes.c_int,ctypes.c_double,da,ip];general.solve_hjb_variable.restype=ctypes.c_int;general.audit_expected_output_variable.argtypes=[da,ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*4;general.audit_expected_output_variable.restype=ctypes.c_double

def native_output(z,S0,T,prize,precision,dy=.0125,nt=1024,keep=False):
 ci,cj,sig,lam=np.exp(np.array(z)[:4]);h=S0/(sig*sig*T);nu=precision*sig*sig*T*T;wi=prize/(ci*sig*sig);wj=prize/(cj*sig*sig);ny=2*math.ceil(8*math.sqrt(1+h)/dy)+1;policy=np.empty((nt+1,ny,2));steps=ctypes.c_int()
 if abs(precision*S0*S0/(sig*sig)-1)<1e-10:
  status=steady.solve_hjb(wi,wj,h,dy,nt,8.,policy,ctypes.byref(steps));assert status==ny,('steady HJB',status);value=steady.audit_expected_output(policy,ny,nt,dy,h,z[4]/(sig*math.sqrt(T)))
 else:
  status=general.solve_hjb_variable(wi,wj,h,nu,dy,nt,8.,policy,ctypes.byref(steps));assert status==ny,('variable HJB',status);value=general.audit_expected_output_variable(policy,ny,nt,dy,h,nu,z[4]/(sig*math.sqrt(T)))
 assert np.isfinite(value) and value>=0,('output positivity',float(value));B=float(sig*math.sqrt(T)*value)
 if keep:return B,policy,h
 return B
```
