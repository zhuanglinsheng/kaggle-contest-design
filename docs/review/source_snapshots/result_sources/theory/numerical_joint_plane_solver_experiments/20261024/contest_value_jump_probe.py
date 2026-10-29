import math,time
import numpy as np
from scipy.special import ndtr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

def solve(sigma,alpha,T,precision,nodes=801,stretch=18,max_dt=.002,trajectory_callback=None):
 q0=1+sigma*sigma*T;e=alpha*T/q0;scale=math.sqrt(q0);rate=sigma*math.sqrt(precision)*T;steady=sigma/math.sqrt(precision)
 def S(t):
  z=math.tanh(rate*t);return steady*(1+steady*z)/(steady+z)/q0
 def Q(t):return S(t)+sigma*sigma*T/q0*(1-t)
 radius=max(.05,min(.5,5*math.sqrt(precision*T)))
 inner=radius/100
 positive=np.r_[np.geomspace(1e-9,inner,100),np.linspace(inner,radius,nodes)[1:],np.geomspace(radius,7,60)[1:]]
 x=np.r_[-positive[::-1],0.,positive];nodes=len(x);h=np.diff(x);hm=h[:-1];hp=h[1:];n=nodes-2
 gm=-hp/(hm*(hm+hp));gp=hm/(hp*(hm+hp));gc=(hp-hm)/(hm*hp)
 lm=2/(hm*(hm+hp));lp=2/(hp*(hm+hp));lc=-2/(hm*hp)
 def grad(v):return gm*(v[:-2]-v[1:-1])+gp*(v[2:]-v[1:-1])
 def lap(v):return lm*(v[:-2]-v[1:-1])+lp*(v[2:]-v[1:-1])
 v=np.zeros(nodes);tau=0.;dt=max_dt;stepcount=0;reject=0;start=time.monotonic()
 term=S(1);phi=np.exp(-x*x/(2*term))/math.sqrt(2*math.pi*term)
 beta=term*phi+x*(ndtr(x/math.sqrt(term))-.5)
 out=np.column_stack((ndtr(x/math.sqrt(term)),beta,beta));bd=out[[0,-1]].copy()
 times=[0.];policies=[e*phi]
 while tau<1-1e-13:
  step=min(dt,1-tau);a=precision*(S(1-tau-step)*q0)**2*T/q0;q=Q(1-tau-step)
  g=np.exp(-x*x/(2*q))/math.sqrt(2*math.pi*q);base=ndtr(x/math.sqrt(q));w=v.copy();converged=False
  face_x=(x[:-1]+x[1:])/2
  face_g=np.exp(-face_x**2/(2*q))/math.sqrt(2*math.pi*q)
  dl=a/(hm*(hm+hp));dr=a/(hp*(hm+hp))
  def res(w):
   f=face_g[1:]+np.diff(w)[1:]/hp
   back=face_g[:-1]+np.diff(w)[:-1]/hm
   own=e*np.maximum(f,0);opp=own[::-1]
   ld=opp/hm;rd=own/hp;l=dl+ld;right=dr+rd
   force=a/2*lap(w)-ld*hm*back+rd*hp*f-own*own/(2*e)
   r=w[1:-1]-v[1:-1]-step*force
   B=-e*back*(f[::-1]>0)
   row=1+step*(l+right)+step*abs(B)/hp[::-1]*2
   return r,row,None,f,l,right,B
  for it in range(40):
   r,row,z,p,l,right,B=res(w);norm=np.max(abs(r)/row)
   if norm<1e-9:converged=True;break
   rows=[];cols=[];data=[];index=np.arange(n)
   for off,local,gd in [(-1,l,gm),(0,-l-right,gc),(1,right,gp)]:
    j=index+off;ok=(j>=0)&(j<n)
    rows.extend(index[ok]);cols.extend(j[ok]);data.extend((-step*local+(1 if off==0 else 0))[ok]/row[ok])
    mir=n-1-index;j=mir+off;ok=(j>=0)&(j<n)
    rows.extend(index[ok]);cols.extend(j[ok]);mirror_gd=np.zeros(n) if off==-1 else ((-1/hp)[::-1] if off==0 else (1/hp)[::-1])
    data.extend((-step*B*mirror_gd)[ok]/row[ok])
   jac=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsc()
   try:delta=spsolve(jac,-r/row)
   except Exception:break
   weight=1.;accepted=False
   for _ in range(20):
    wn=w.copy();wn[1:-1]+=weight*delta;rn,_,_,pn,*_=res(wn);u=base+wn
    if u.min()>-1e-6 and u.max()<1+1e-6 and (u+u[::-1]).max()<=1+1e-6 and np.max(abs(rn)/row)<norm:
     w=wn;accepted=True;break
    weight*=.5
   if not accepted:break
  if not converged:
   dt=step/2;reject+=1
   if dt<1e-9:raise RuntimeError(f'Newton failed tau={tau}, norm={norm}, pmin={p.min()}')
   continue
  f=face_g+np.diff(w)/h
  p=np.maximum(np.r_[f,0.],0);drift=e*(p-p[::-1]);D=a/2
  l=2*D/(hm*(hm+hp))+np.maximum(-drift[1:-1],0)/hm
  right=2*D/(hp*(hm+hp))+np.maximum(drift[1:-1],0)/hp
  source=np.column_stack((-.5*e*p*p,.5*e*(p+p[::-1]),np.zeros(nodes)))
  for k in range(3):
   lk,rk=(2*D/(hm*(hm+hp)),2*D/(hp*(hm+hp))) if k==2 else(l,right)
   band=np.zeros((3,n));band[1]=1+step*(lk+rk);band[0,1:]=-step*rk[:-1];band[2,:-1]=-step*lk[1:]
   rhs=out[1:-1,k]+step*source[1:-1,k];rhs[0]+=step*lk[0]*bd[0,k];rhs[-1]+=step*rk[-1]*bd[1,k]
   out[1:-1,k]=solve_banded((1,1),band,rhs,check_finite=False)
  v=w;tau+=step;stepcount+=1;dt=min(max_dt,step*1.2)
  times.append(tau);policies.append(e*p)
  if time.monotonic()-start>150:raise RuntimeError(f'Wall budget tau={tau}')
 if trajectory_callback is not None:trajectory_callback(dict(x=x,times=np.array(times),policies=np.array(policies).T))
 m=nodes//2;U=ndtr(x)+v
 return dict(B=float(scale*out[m,1]),value=float(U[m]),physical=float(out[m,0]),global_residual=float(np.max(abs(U-out[:,0]))),peak=float(p.max()*e),pmin=float(p.min()),nodes=nodes,steps=stepcount,reject=reject,seconds=time.monotonic()-start)
