"""Conservative weak-feedback HJB audit for symmetric theory designs.

Fixed S0=1, matching the joint-design figure. Evolves p=U_i,y and n=p(-y)
with conservative exponentially fitted fluxes and fully implicit BDF time
integration. Player payoffs and organizer output use the physical diffusion,
independently of the density transport. A returned solution is not automatically
accepted: root/global payoff residuals and refinement must also be checked.
Existing paper inputs and numerical classifications are never modified.
"""
import math,time
import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

def bernoulli(z):
    out=np.empty_like(z)
    small=np.abs(z)<1e-5
    out[small]=1-z[small]/2+z[small]**2/12
    hi=z>50;lo=z < -50
    out[hi]=z[hi]*np.exp(-z[hi]);out[lo]=-z[lo]
    mid=~(small|hi|lo)
    out[mid]=z[mid]/np.expm1(z[mid])
    return out



def solve(sigma,alpha,T,precision,nodes=601,stretch=8.,rtol=1e-6,atol=1e-8,max_calls=40000):
    if nodes<21 or nodes%2!=1 or stretch<=0:
        raise ValueError("Use an odd node count >=21 and positive stretch")
    q0=1+sigma*sigma*T;e=alpha*T/q0;scale=math.sqrt(q0)
    if sigma<=0 or alpha<0 or T<=0 or precision<=0:
        raise ValueError("Require sigma,T,precision>0 and alpha>=0")
    rate=sigma*math.sqrt(precision)*T;steady=sigma/math.sqrt(precision)
    def S(t):
        z=math.tanh(rate*t)
        return steady*(1+steady*z)/(steady+z)/q0
    x=7*np.sinh(stretch*np.linspace(-1,1,nodes))/math.sinh(stretch)
    faces=np.r_[x[0]-(x[1]-x[0])/2,(x[1:]+x[:-1])/2,x[-1]+(x[-1]-x[-2])/2]
    widths=np.diff(faces);distance=np.diff(x)
    terminal=S(1)
    # Gaussian cell averages avoid subtracting nearly equal CDF values on
    # the very fine central cells used for weak feedback.
    quad_x,quad_w=roots_legendre(8)
    points=(faces[:-1,None]+faces[1:,None])/2+widths[:,None]*quad_x/2
    p=np.exp(-points**2/(2*terminal))@quad_w/(2*math.sqrt(2*math.pi*terminal))
    p=p/np.sum(p*widths)
    val=ndtr(x/math.sqrt(terminal))
    beta=math.sqrt(terminal/(2*math.pi))*np.exp(-x*x/(2*terminal))+x*(ndtr(x/math.sqrt(terminal))-.5)
    state=np.column_stack((np.zeros_like(p),val,beta,beta))
    hm=distance[:-1];hp=distance[1:]
    pattern=lil_matrix((nodes*4,nodes*4))
    for i in range(nodes):
        for j in set([k for k in [i-1,i,i+1,nodes-i,nodes-i-1,nodes-i-2] if 0<=k<nodes]):
            pattern[i*4:(i+1)*4,j*4]=1
        for j in range(max(0,i-1),min(nodes,i+2)):
            for k in range(1,4):pattern[i*4+k,j*4+k]=1
    calls=0;start=time.monotonic()
    def rhs(t,y):
        nonlocal calls
        fields=y.reshape(nodes,4);v=fields[:,0]
        q=S(1-t)+sigma*sigma*T/q0*t
        pg=np.exp(-x*x/(2*q))/math.sqrt(2*math.pi*q)
        pg/=np.sum(pg*widths)
        c=pg*np.expm1(np.clip(v,-700,650))
        p=pg*np.exp(np.clip(v,-700,650))
        cf=(c[:-1]+c[1:])/2
        nf=(c[:0:-1]+c[-2::-1])/2
        gf=(pg[:-1]+pg[1:])/2
        dv=e*(nf-.5*cf)
        vf=e*.5*gf+dv
        D=precision*(S(1-t)*q0)**2*T/q0/2
        pe=vf*distance/D
        ad=np.empty_like(pe);small=np.abs(pe)<1e-3
        ad[small]=pe[small]**2/12-pe[small]**4/720
        ad[~small]=pe[~small]/(2*np.tanh(pe[~small]/2))-1
        logratio=-(x[1:]**2-x[:-1]**2)/(2*q)
        # Exponential differences preserve the nearly flat central Gaussian.
        dg=np.where(logratio<0,pg[:-1]*np.expm1(np.minimum(logratio,0)),
                    -pg[1:]*np.expm1(-np.maximum(logratio,0)))
        flux=e*.5*gf*cf+dv*gf+dv*cf-D/distance*((1+ad)*np.diff(c)+ad*dg)
        out=np.empty_like(fields)
        ct=-np.diff(np.r_[0.,flux,0.])/widths+e*x/q*pg*pg
        gt=D*(x*x/q**2-1/q)
        gt-=np.sum(pg*widths*gt)
        out[:,0]=(ct-c*gt)/np.maximum(p,1e-300)
        drift=e*(p-p[::-1])
        l=2*D/(hm*(hm+hp))+np.maximum(-drift[1:-1],0)/hm
        r=2*D/(hp*(hm+hp))+np.maximum(drift[1:-1],0)/hp
        out[1:-1,1:]=l[:,None]*(fields[:-2,1:]-fields[1:-1,1:])+r[:,None]*(fields[2:,1:]-fields[1:-1,1:])
        out[1:-1,1]-=.5*e*p[1:-1]**2
        out[1:-1,2]+=.5*e*(p[1:-1]+p[-2:0:-1])
        out[1:-1,3]=2*D/(hm*(hm+hp))*(fields[:-2,3]-fields[1:-1,3])+2*D/(hp*(hm+hp))*(fields[2:,3]-fields[1:-1,3])
        out[[0,-1],1:]=0.
        calls+=1
        if calls>max_calls:
            raise RuntimeError("Conservative solver RHS-call budget exhausted")
        return out.ravel()
    sol=solve_ivp(rhs,[0,1],state.ravel(),method='BDF',jac_sparsity=pattern.tocsc(),
                  rtol=rtol,atol=atol,max_step=.02)
    f=sol.y[:,-1].reshape(nodes,4);pg=np.exp(-x*x/2)/math.sqrt(2*math.pi);pg/=np.sum(pg*widths);p=pg*np.exp(np.clip(f[:,0],-700,650))
    m=nodes//2
    cdf=np.cumsum(p*widths)-.5*p*widths
    value=cdf[m]
    global_residual=float(np.max(np.abs(cdf-f[:,1])))
    total_value=cdf+cdf[::-1]
    result=dict(success=sol.success,message=sol.message,B=float(scale*f[m,2]),
                calibrated_B=float(scale*(f[m,2]-f[m,3])+scale/math.sqrt(2*math.pi)),
                baseline_error=float(scale*f[m,3]-scale/math.sqrt(2*math.pi)),
                player_value=float(value),physical_value=float(f[m,1]),residual=float(abs(value-f[m,1])),
                global_residual=global_residual,maximum_total_value=float(total_value.max()),
                minimum_physical_value=float(f[:,1].min()),maximum_physical_value=float(f[:,1].max()),
                mass=float(np.sum(p*widths)),min_density=float(p.min()),peak=float(p.max()*alpha/scale),
                peak_location=float(x[p.argmax()]),nodes=nodes,dxmin=float(distance.min()),
                steps=len(sol.t),calls=calls,seconds=time.monotonic()-start)
    result['mesh_checks_passed']=bool(
        sol.success and abs(result['mass']-1)<1e-7
        and result['min_density']>=-1e-8
        and result['maximum_total_value']<=1+2e-4
        and result['minimum_physical_value']>=-2e-3
        and result['maximum_physical_value']<=1+2e-3
        and result['global_residual']<=2e-3
        and abs(result['baseline_error'])<=1e-5)
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--nodes',type=int,default=601)
    ap.add_argument('--precision',type=float,default=.01)
    ap.add_argument('--stretch',type=float,default=8.)
    ap.add_argument('--sigma',type=float,default=.06)
    ap.add_argument('--alpha',type=float,default=2.164248520857008)
    ap.add_argument('--horizon',type=float,default=.7656981839261894)
    a=ap.parse_args()
    print(solve(a.sigma,a.alpha,a.horizon,a.precision,
                nodes=a.nodes,stretch=a.stretch),flush=True)
