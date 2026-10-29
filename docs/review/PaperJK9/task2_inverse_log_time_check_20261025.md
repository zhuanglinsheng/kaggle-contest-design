# Task 2：既有逆坐标方法的时间步核查

日期：2026-10-25。新直接逆坐标在弱例上收敛，但强例仅粗层通过，细层非单调，不能接受。回查旧 log 逆坐标源未发现可立即确认的代数因子错误，其旧输出的时间推进使用一阶隐式步，按 Newton 收敛调整步长，而没有独立输出截断误差控制。

这里保持旧离散方程，分别减小最大时间步并对照同一 sigma=.06, alpha=2.164248520857008, T=.7656981839261894, lambda=.01 已修复候选 B≈1.1286924，检验时间误差是否解释旧结果差异。仅将每次试验墙钟上限从 90 秒改为 45 秒，不改模型或判定门槛；耗尽记录为失败。

```python
"""Positive log-coordinate implicit conservative inverse HJB prototype."""
import math,time
import numpy as np
from scipy.special import ndtr,ndtri,logsumexp
from scipy.linalg import solve_banded

def logB(v):
 v=np.asarray(v);r=np.empty_like(v);small=abs(v)<1e-5;pos=v>40;neg=v < -40
 r[small]=-v[small]/2-v[small]**2/24
 r[pos]=np.log(v[pos])-v[pos];r[neg]=np.log(-v[neg])
 mid=~(small|pos|neg);r[mid]=np.log(v[mid]/np.expm1(v[mid]));return r

def logsinh(v):return v+np.log(-np.expm1(-2*v))-math.log(2)

def solve(sigma,alpha,T,precision,nodes=401,stretch=5,max_dt=.002,verbose=False):
 q0=1+sigma*sigma*T;e=alpha*T/q0;scale=math.sqrt(q0)
 rate=sigma*math.sqrt(precision)*T;steady=sigma/math.sqrt(precision)
 def S(t):
  z=math.tanh(rate*t);return steady*(1+steady*z)/(steady+z)/q0
 def Q(t):return S(t)+sigma*sigma*T/q0*(1-t)
 d=(1-1e-7)*np.sinh(stretch*np.linspace(-1,1,nodes))/math.sinh(stretch)
 h=np.diff(d);hm=h[:-1];hp=h[1:];width=(hm+hp)/2
 zg=ndtri((d+1)/2);phi=np.exp(-zg*zg/2)/math.sqrt(2*math.pi)
 R=np.ones(nodes);Z=.5*math.log(S(1))-np.log(2*phi)
 J=np.sqrt(S(1))*phi+np.sqrt(S(1))*zg*d/2
 V=(1+d)/2
 tau=0.;dt=max_dt;steps=0;reject=0;start=time.monotonic();n=nodes-2
 def residual(z,r,zo,ro,step,a):
  k=e/(2*a);Mlog=math.log(2)+2*z[1:-1]
  khm=k*hm;khp=k*hp
  # Exponential-fit Helmholtz operator, normalized on a constant function.
  logA=math.log(k)-np.log(width)-logsinh(khm)
  logC=math.log(k)-np.log(width)+np.log(1/np.tanh(khm)+1/np.tanh(khp))
  logD=math.log(k)-np.log(width)-logsinh(khp)
  flat=logsumexp(np.stack([logA,logD]),axis=0)-logC
  logfac=2*math.log(k)-logC-np.log(-np.expm1(flat))
  logA+=math.log(a)+logfac;logC+=math.log(a)+logfac;logD+=math.log(a)+logfac
  mt=Mlog-math.log(step)
  logcoef=mt+math.log(k)-logC
  logden=np.logaddexp(0,logcoef)
  rr=np.exp(logcoef-logden)*(r[1:-1]-ro[1:-1])+np.exp(-logden)-np.exp(np.minimum(logA-logC+k*(r[:-2]-r[1:-1])-logden,600))-np.exp(np.minimum(logD-logC+k*(r[2:]-r[1:-1])-logden,600))
  potential=(e/a)*np.diff(r)
  la=math.log(a/2)-np.log(h)+logB(potential)
  lb=math.log(a/2)-np.log(h)+logB(-potential)
  wt=math.log(step)-np.log(width)
  rz=logsumexp(np.stack([z[1:-1],wt+la[:-1]-z[:-2],wt+lb[1:]-z[2:]]),axis=0)-logsumexp(np.stack([zo[1:-1],wt+np.logaddexp(la[1:],lb[:-1])-z[1:-1]]),axis=0)
  return rz,rr
 while tau<1-1e-13:
  step=min(dt,1-tau);a=precision*(S(1-tau-step/2)*q0)**2*T/q0
  z=Z.copy();r=R.copy();z[[0,-1]]=.5*math.log(Q(1-tau-step))-np.log(2*phi[[0,-1]])
  converged=False
  for it in range(40):
   rz,rr=residual(z,r,Z,R,step,a);f=np.column_stack((rz,rr)).ravel();norm=np.max(abs(f))
   if norm<2e-9:converged=True;break
   band=np.zeros((7,2*n));y=np.column_stack((z[1:-1],r[1:-1])).ravel()
   for color in range(7):
    cols=np.arange(color,2*n,7);delta=1e-7*np.maximum(1,abs(y[cols]));yp=y.copy();yp[cols]+=delta
    zp=z.copy();rp=r.copy();zp[1:-1]=yp[::2];rp[1:-1]=yp[1::2]
    fz,fr=residual(zp,rp,Z,R,step,a);df=np.column_stack((fz,fr)).ravel()-f
    for off in range(-3,4):
     rows=cols+off;valid=(rows>=0)&(rows<2*n);band[3+off,cols[valid]]=df[rows[valid]]/delta[valid]
   try:delta=solve_banded((3,3),band,-f,check_finite=False).reshape(n,2)
   except Exception:break
   weight=1.;accepted=False
   for _ in range(18):
    zn=z.copy();rn=r.copy();zn[1:-1]+=weight*delta[:,0];rn[1:-1]+=weight*delta[:,1]
    zn=(zn+zn[::-1])/2;rn=(rn+rn[::-1])/2
    if rn.min()<0 or rn.max()>1+1e-8:weight*=.5;continue
    fz,fr=residual(zn,rn,Z,R,step,a);nn=max(np.max(abs(fz)),np.max(abs(fr)))
    if nn<norm:z,r=zn,rn;accepted=True;break
    weight*=.5
   if not accepted:break
  if not converged:
   dt=step/2;reject+=1
   if dt<1e-9:raise RuntimeError(f'Newton failure tau={tau}, norm={norm}, zmin={z.min()}')
   continue
  # Physical value/output equations in inverse coordinates, exponentially fitted.
  gm=-hp/(hm*(hm+hp));gp=hm/(hp*(hm+hp));gc=(hp-hm)/(hm*hp)
  gr=gm*r[:-2]+gc*r[1:-1]+gp*r[2:]
  lm=2/(hm*(hm+hp));lp=2/(hp*(hm+hp))
  l=a*lm*np.exp(logB((e/a)*(r[1:-1]-r[:-2])))
  right=a*lp*np.exp(logB(-(e/a)*(r[2:]-r[1:-1])))
  M=2*np.exp(np.maximum(2*z[1:-1],-700));band=np.zeros((3,n));band[1]=M/step+l+right;band[0,1:]=-right[:-1];band[2,:-1]=-l[1:]
  q=Q(1-tau-step);bJ=np.sqrt(q)*phi[[0,-1]]+np.sqrt(q)*zg[[0,-1]]*d[[0,-1]]/2
  for out,source,b in [(J,e*np.exp(z[1:-1]),bJ),(V,-e*(1+gr)**2/4,(1+d[[0,-1]])/2)]:
   rhs=M/step*out[1:-1]+source;rhs[0]+=l[0]*b[0];rhs[-1]+=right[-1]*b[1]
   out[1:-1]=solve_banded((1,1),band,rhs,check_finite=False);out[[0,-1]]=b
  Z,R=z,r;tau+=step;steps+=1;dt=min(max_dt,step*1.2)
  if verbose and steps%100==0:print(tau,Z.min(),R[nodes//2],flush=True)
  if time.monotonic()-start>45:raise RuntimeError(f'Wall budget tau={tau}')
 m=nodes//2
 return dict(B=float(scale*J[m]),value=float(R[m]/2),physical=float(V[m]),residual=float(abs(R[m]/2-V[m])),Zmin=float(Z.min()),steps=steps,rejections=reject,seconds=time.monotonic()-start)
```
