"""Independent best-response check against a prescribed symmetric policy path.

Evaluates the prescribed policy and its best response on the same monotone
finite-state generator. Mesh/time refinement is still required: a small discrete
deviation gain alone does not bound the continuous-model output error.
"""
import math
import numpy as np
from scipy.special import ndtr
from scipy.linalg import solve_banded
from pipeline.theory.joint_design_conservative_audit import solve


def evaluate_symmetric_policy(sigma,alpha,T,precision,x,times,policies,steps=2000,tolerance=1e-7):
 x=np.asarray(x,dtype=float);times=np.asarray(times,dtype=float);policies=np.asarray(policies,dtype=float)
 nodes=len(x)
 if sigma<=0 or T<=0 or alpha<=0 or precision<=0 or steps<1 or policies.shape!=(nodes,len(times)):
  raise ValueError('Require positive alpha/precision and a matching policy path')
 if not np.all(np.isfinite(policies)) or np.min(policies)<-1e-8:
  raise ValueError('Policy must be finite and nonnegative')
 policies=np.maximum(policies,0.)
 if nodes<5 or nodes%2!=1 or not np.allclose(x,-x[::-1],rtol=1e-12,atol=1e-12):
  raise ValueError('Use an odd, symmetric state grid')
 if not np.all(np.diff(x)>0) or not np.all(np.diff(times)>0) or times[0]>0 or times[-1]<1:
  raise ValueError('Use increasing state/time grids covering [0,1]')
 q0=1+sigma*sigma*T;e=alpha*T/q0;rate=sigma*math.sqrt(precision)*T;steady=sigma/math.sqrt(precision)
 def S(t):
  z=math.tanh(rate*t);return steady*(1+steady*z)/(steady+z)/q0
 h=np.diff(x);hm=h[:-1];hp=h[1:];dt=1/steps
 br=ndtr(x/np.sqrt(S(1)));ownvalue=br.copy();physical=br.copy();worst=0;maximum_hamiltonian_gap=0.;max_subgame_gain=0.;max_subgame_tau=0.;max_subgame_x=0.
 def linear(l,r,rhs):
  band=np.zeros((3,nodes-2));band[1]=1+dt*(l+r);band[0,1:]=-dt*r[:-1];band[2,:-1]=-dt*l[1:]
  rhs=rhs.copy();rhs[-1]+=dt*r[-1]
  return solve_banded((1,1),band,rhs,check_finite=False)
 qbest=np.r_[e*np.maximum(np.diff(br)/h,0.),0.]
 for j in range(steps):
  tau=(j+.5)*dt;k=min(np.searchsorted(times,tau),len(times)-1)
  f=(tau-times[k-1])/(times[k]-times[k-1]);policy=(1-f)*policies[:,k-1]+f*policies[:,k];opp=policy[::-1]
  a=precision*(S(1-tau)*q0)**2*T/q0;dl=a/(hm*(hm+hp));dr=a/(hp*(hm+hp))
  old=br.copy()
  for it in range(100):
   br[1:-1]=linear(dl+opp[1:-1]/hm,dr+qbest[1:-1]/hp,old[1:-1]-.5*dt*qbest[1:-1]**2/e)
   candidate=np.r_[e*np.maximum(np.diff(br)/h,0),0.]
   gap=float(np.max((candidate-qbest)**2/(2*e)))
   qbest=candidate
   if gap<tolerance:break
  else:raise RuntimeError(f'BR Howard failure {j}, gap={gap}')
  worst=max(worst,it);maximum_hamiltonian_gap=max(maximum_hamiltonian_gap,gap)
  ownvalue[1:-1]=linear(dl+opp[1:-1]/hm,dr+policy[1:-1]/hp,ownvalue[1:-1]-.5*dt*policy[1:-1]**2/e)
  drift=policy-opp
  gains=br-ownvalue
  imax=int(np.argmax(gains))
  if gains[imax]>max_subgame_gain:
   max_subgame_gain=float(gains[imax]);max_subgame_tau=(j+1)*dt;max_subgame_x=float(x[imax])
  physical[1:-1]=linear(dl+np.maximum(-drift[1:-1],0)/hm,dr+np.maximum(drift[1:-1],0)/hp,physical[1:-1]-.5*dt*policy[1:-1]**2/e)
 m=nodes//2
 return dict(BR=float(br[m]),prescribed=float(ownvalue[m]),physical=float(physical[m]),deviation_gain=float(br[m]-ownvalue[m]),max_deviation_gain=float(np.max(br-ownvalue)),worst=worst,steps=steps,maximum_hamiltonian_gap=maximum_hamiltonian_gap,max_subgame_deviation_gain=max_subgame_gain,max_subgame_reverse_time=max_subgame_tau,max_subgame_state=max_subgame_x)



def audit(sigma,alpha,T,precision,nodes=1601,stretch=18,steps=2000,max_calls=30000):
 captured=[]
 result=solve(sigma,alpha,T,precision,nodes=nodes,stretch=stretch,
              rtol=2e-7,atol=2e-9,max_calls=max_calls,
              trajectory_callback=captured.append)
 path=captured[0];e=alpha*T/(1+sigma*sigma*T)
 response=evaluate_symmetric_policy(sigma,alpha,T,precision,path['x'],path['times'],
                                    e*path['density'],steps=steps)
 return dict(density=result,**response)
