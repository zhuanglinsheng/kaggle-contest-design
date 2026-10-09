"""Independent L-stable HJB audit for competitive failed plane designs.

Backward Euler diffusion and CFL-controlled Lax--Friedrichs numerical viscosity
replace the centered CN/Heun scheme only in the theory diagnostic. The
viscosity vanishes with spatial mesh size. No negative effort is silently
clipped into a valid policy. Organizer output uses physical diffusion and an
implicit upwind drift, not the HJB's artificial viscosity.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr

from archive.experiments.exact_joint_design import filtering_variance

NO_SIGNAL_VERIFICATION_TOL=2e-3


def stable_output(sigma,alpha,T,precision,level,verify_policy=False):
    q0=1+sigma*sigma*T
    scale=math.sqrt(q0)
    dx={3:.02,4:.01,5:.005,6:.0025}[level]
    steps={3:500,4:1000,5:2000,6:4000}[level]
    ell=precision*sigma*sigma*T*T
    s0=1/(sigma*sigma*T)
    def s(t):
        return float(filtering_variance(t,s0,ell))*sigma*sigma*T/q0
    x=np.arange(-math.ceil(7/dx),math.ceil(7/dx)+1)*dx
    e=alpha*T/q0
    uniform=np.linspace(0,1,steps+1)
    if precision>0:
        dense=np.r_[0.,np.geomspace(1e-12,1.,max(20000,50*steps))]
        ss=filtering_variance(dense,s0,ell)*sigma*sigma*T/q0
        cumulative=1-(ss+sigma*sigma*T/q0*(1-dense))
        equal=np.interp(np.linspace(0,cumulative[-1],steps+1),cumulative,dense)
        grid=np.unique(np.r_[uniform,equal])
    else:
        grid=uniform
    terminal=s(1.)
    values=np.column_stack((ndtr(x/math.sqrt(terminal)),ndtr(-x/math.sqrt(terminal))))
    band=np.zeros((3,len(x)-2))
    now=1.;step_count=0
    min_effort=0.;max_effort=0.;partial_energy=0.;energy_integral=0.
    midpoint=len(x)//2
    beta=math.sqrt(terminal/(2*math.pi))*np.exp(-x*x/(2*terminal))+x*(ndtr(x/math.sqrt(terminal))-.5)
    # The organizer runs backward on exactly the same steps as the players.
    # Only two adjacent policies are needed, not the entire trajectory.
    output=np.column_stack((beta,beta,values)) if verify_policy else np.column_stack((beta,beta))
    edge=np.abs(x[[0,-1]])/2
    boundary=np.column_stack((edge,edge,values[[0,-1]])) if verify_policy else np.column_stack((edge,edge))
    output[[0,-1]]=boundary
    def grad(v):return np.gradient(v,dx,axis=0,edge_order=2)
    def store():
        nonlocal min_effort,max_effort
        g=grad(values)
        policy=np.column_stack((g[:,0],-g[:,1]))
        min_effort=min(min_effort,float(policy.min())*alpha/scale)
        max_effort=max(max_effort,float(policy.max())*alpha/scale)
        if min_effort < -2e-6:
            raise RuntimeError(f"stable audit negative effort {min_effort}")
        return np.maximum(policy,0).astype(np.float32)
    current_policy=store()
    for target in grid[-2::-1]:
        while now-target>1e-14:
            g=grad(values)
            speed=max(1e-12,2*e*float(np.max(np.abs(g).sum(axis=1))))
            dt=min(now-target,.2*dx/speed)
            previous=now-dt
            variance=max(0.,s(previous)+sigma*sigma*T/q0*(1-previous)
                         -s(now)-sigma*sigma*T/q0*(1-now))
            # The added diffusion is O(dx), absent from the physical model.
            ratio=(variance+dt*speed*dx)/(2*dx*dx)
            cross=g[:,0]*g[:,1]
            force=e*np.column_stack((g[:,0]**2/2+cross,g[:,1]**2/2+cross))
            band[0,1:]=-ratio;band[1]=1+2*ratio;band[2,:-1]=-ratio
            rhs=values[1:-1]+dt*force[1:-1]
            rhs[0]+=ratio*values[0];rhs[-1]+=ratio*values[-1]
            values[1:-1]=solve_banded((1,1),band,rhs,check_finite=False)
            if not np.isfinite(values).all() or values.min() < -2e-5 or values.max()>1+2e-5:
                raise RuntimeError("stable audit violates value bounds")
            next_policy=store()
            if precision==0:
                partial_energy+=e*dt*(float(next_policy[midpoint,0])**2
                                      +float(current_policy[midpoint,0])**2)/2
                pair=np.array([current_policy[midpoint,0],next_policy[midpoint,0]])
                energy_integral+=(now-previous)*float((pair**2).sum())/2
                if partial_energy>1+1e-3:
                    raise RuntimeError(f"no-signal policy violates individual value bound: {partial_energy}")
            physical_dt=now-previous
            policy=(current_policy.astype(float)+next_policy.astype(float))/2
            drift=e*(policy[:,0]-policy[:,1])
            reward=e*(policy[:,0]+policy[:,1])/2
            for col in range(output.shape[1]):
                b=drift[1:-1] if col!=1 else np.zeros(len(x)-2)
                left=variance/(2*dx*dx)+physical_dt*np.maximum(-b,0)/dx
                right=variance/(2*dx*dx)+physical_dt*np.maximum(b,0)/dx
                band[0,1:]=-right[:-1];band[1]=1+left+right;band[2,:-1]=-left[1:]
                source=reward if col==0 else (-e*policy[:,col-2]**2/2 if col>=2 else np.zeros(len(x)))
                rhs=output[1:-1,col]+physical_dt*source[1:-1]
                rhs[0]+=left[0]*boundary[0,col];rhs[-1]+=right[-1]*boundary[-1,col]
                output[1:-1,col]=solve_banded((1,1),band,rhs,check_finite=False)
            now=previous;current_policy=next_policy;step_count+=1
            if step_count>=150000:
                raise RuntimeError("stable audit time-step cap")
    gain=scale*(output[midpoint,0]-output[midpoint,1])
    B=scale/math.sqrt(2*math.pi)+gain
    if gain < -1e-6 or gain>math.sqrt(2*alpha*T)+1e-5:
        raise RuntimeError("stable output violates energy bounds")
    if precision==0:
        energy=e*energy_integral
        verification=abs(float(values[midpoint,0])-(.5-energy/2))
        if energy>1+1e-3:
            raise RuntimeError(f"no-signal policy violates individual value bound: {energy}")
        if verification>NO_SIGNAL_VERIFICATION_TOL:
            raise RuntimeError(f"no-signal Nash verification residual: {verification}")
    else:
        energy=math.nan;verification=math.nan
    physical_verification=float(np.max(np.abs(output[midpoint,2:]-values[midpoint]))) if verify_policy else math.nan
    if verify_policy and physical_verification>NO_SIGNAL_VERIFICATION_TOL:
        raise RuntimeError(f"physical-policy value verification residual: {physical_verification}")
    return B,dict(dx=dx,steps=step_count,min_effort=min_effort,max_effort=max_effort,
                  no_signal_energy=energy,no_signal_verification=verification,
                  physical_policy_verification=physical_verification)


def calculate_stable(job):
    sigma,alpha,T,precision,level=job
    row=dict(sigma=sigma,alpha=alpha,T=T,**{"lambda":precision},level=level,
             B=math.nan,error=math.nan,status="")
    try:
        B,diag=stable_output(sigma,alpha,T,precision,int(level))
        row.update(B=B,status="stable_BE_LF_audit")
        # Additional no-signal verification discrepancy cannot be ignored.
        row["error"]=diag["no_signal_verification"]*max(1.,math.sqrt(alpha*T)) if precision==0 else 0.
    except (RuntimeError,ValueError,OverflowError,FloatingPointError) as ex:
        row["status"]="failed: "+str(ex)
    return row


def calculate_verified_stable(job):
    sigma,alpha,T,precision,level=job
    row=dict(sigma=sigma,alpha=alpha,T=T,**{"lambda":precision},level=level,
             B=math.nan,error=math.nan,status="")
    try:
        B,diag=stable_output(sigma,alpha,T,precision,int(level),verify_policy=True)
        row.update(B=B,error=0.,status="verified_physical_policy_audit")
        if precision==0:row["error"]=diag["no_signal_verification"]*max(1.,math.sqrt(alpha*T))
    except (RuntimeError,ValueError,OverflowError,FloatingPointError) as ex:
        row["status"]="failed: "+str(ex)
    return row
