# Task 2：逆坐标 Newton 试算的域外延拓

日期：2026-10-25。直接原型对任意 RHS 试算的 J<=0 立即抛异常，这会把 Newton 域外试步当成整个积分失败。这里仅为求解器内部定义 J_safe=max(J,floor)，使时间积分器有机会缩步；逐次保存的接受状态另行检查严格反解单调、J>10*floor、控制非负以及价值边界。

只有接受状态从未触及延拓区才允许通过物理分支筛查；这仍不是均衡认证。记录内部域外调用次数，并改变 floor 核验是否影响结果。不用裁剪最终负控制制造成功。原直接原型及失败记录保留。

```python
import math,time
import numpy as np
from scipy.special import expit,ndtri,ndtr
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags,kron,csr_matrix

def direct_inverse(sigma,theta,T,precision,ci=1.,cj=1.,S0=1.,mu=0.,nodes=401,extent=16.,rtol=1e-7,atol=1e-11,max_calls=40000,keep_path=False,newton_floor=1e-10):
    if min(sigma,theta,T,precision,ci,cj,S0)<=0 or nodes%2!=1:raise ValueError('Positive parameters and odd grid required')
    q0=S0+sigma*sigma*T;scale=math.sqrt(q0);ei=theta*T/(ci*q0);ej=theta*T/(cj*q0)
    probability=expit(np.linspace(-extent,extent,nodes));d=(ei+ej)*probability-ej
    hm=np.diff(d)[:-1];hp=np.diff(d)[1:];normal=ndtri(probability)
    def S(t):
        rate=sigma*math.sqrt(precision)*T*t;tc=math.tanh(rate)/rate if rate else 1.
        return (S0+sigma*sigma*T*t*tc)/(1+precision*S0*T*t*tc)/q0
    def boundary(tau):
        Q=S(1-tau)+sigma*sigma*T/q0*tau
        return np.array([[math.sqrt(Q)*normal[0],ej+(ei-ej)*probability[0],0.],[math.sqrt(Q)*normal[-1],ej+(ei-ej)*probability[-1],0.]])
    initial=np.column_stack((math.sqrt(S(1))*normal,ej+(ei-ej)*probability,np.zeros(nodes)))
    def unpack(tau,y):
        b=boundary(tau);return np.vstack((b[0],y.reshape(nodes-2,3),b[1]))
    def derivatives(v):
        left=(v[1:-1]-v[:-2])/hm[:,None];right=(v[2:]-v[1:-1])/hp[:,None]
        first=(hp[:,None]*left+hm[:,None]*right)/(hm+hp)[:,None]
        second=2*(right-left)/(hm+hp)[:,None]
        return first,second
    calls=0;smallest_slope=math.inf;floor_calls=0;start=time.monotonic()
    def rhs(tau,y):
        nonlocal calls,smallest_slope,floor_calls
        calls+=1
        if calls>max_calls or time.monotonic()-start>100:raise RuntimeError('Inverse integration budget exhausted at '+str(tau))
        f=unpack(tau,y);first,second=derivatives(f);J=first[:,0];fd=first[:,1];Hd=first[:,2]
        smallest_slope=min(smallest_slope,float(J.min()))
        if not np.isfinite(f).all():raise RuntimeError('Nonfinite inverse state at '+str(tau))
        if np.min(J)<newton_floor:floor_calls+=1
        J=np.maximum(J,newton_floor)
        D=precision*(S(1-tau)*q0)**2*T/q0/2;Q=S(1-tau)+sigma*sigma*T/q0*tau
        out=np.empty_like(first)
        out[:,0]=D*second[:,0]/J**2-.5*fd/J
        out[:,1]=(D*second[:,1]+(fd*fd-1)/4)/J**2
        out[:,2]=(D*second[:,2]+.5*fd*Hd)/J**2+(.5+fd*(ndtr(f[1:-1,0]/math.sqrt(Q))-.5))/J
        return out.ravel()
    pattern=kron(diags([np.ones(nodes-3),np.ones(nodes-2),np.ones(nodes-3)],[-1,0,1],shape=(nodes-2,nodes-2)),csr_matrix(np.ones((3,3))),format='csc')
    sol=solve_ivp(rhs,(0.,1.),initial[1:-1].ravel(),method='BDF',jac_sparsity=pattern,rtol=rtol,atol=atol,max_step=.02)
    final=unpack(sol.t[-1],sol.y[:,-1]);first,_=derivatives(final);J=first[:,0];fd=first[:,1]
    qi=(1+fd)/(2*J);qj=(1-fd)/(2*J)
    if not np.all(np.diff(final[:,0])>0):raise RuntimeError('Final inverse grid not increasing')
    baseline=scale/math.sqrt(2*math.pi)*math.exp(-mu*mu/(2*q0))+mu*(float(ndtr(mu/scale))-.5)
    B=baseline+scale*float(PchipInterpolator(final[:,0],final[:,2])(mu/scale))
    result=dict(success=bool(sol.success),message=sol.message,B=B,baseline=baseline,sigma=sigma,theta=theta,T=T,precision=precision,ci=ci,cj=cj,S0=S0,mu=mu,nodes=nodes,extent=extent,calls=calls,time_nodes=len(sol.t),seconds=time.monotonic()-start,smallest_inverse_slope_all_rhs=smallest_slope,minimum_final_control=float(min(qi.min(),qj.min())),maximum_final_control=float(max(qi.max(),qj.max())),maximum_absolute_value_derivative=float(np.abs(fd).max()),maximum_total_value=float(np.max((final[:,1]+d)/(2*ei)+(final[:,1]-d)/(2*ej))))
    accepted_minimum_slope=math.inf;accepted_minimum_control=math.inf;accepted_monotone=True
    for index,tau in enumerate(sol.t):
        state=unpack(tau,sol.y[:,index]);derivative,_=derivatives(state);jj=derivative[:,0];ff=derivative[:,1]
        accepted_minimum_slope=min(accepted_minimum_slope,float(jj.min()))
        accepted_monotone=accepted_monotone and bool(np.all(np.diff(state[:,0])>0))
        if jj.min()>0:accepted_minimum_control=min(accepted_minimum_control,float(np.min((1-np.abs(ff))/(2*jj))))
    result.update(newton_extension_floor=newton_floor,newton_extension_rhs_calls=floor_calls,
        minimum_accepted_inverse_slope=accepted_minimum_slope,minimum_accepted_control=accepted_minimum_control,
        all_accepted_states_monotone=accepted_monotone,
        physical_branch_screen_passed=bool(sol.success and accepted_monotone and accepted_minimum_slope>10*newton_floor and accepted_minimum_control>=-1e-8 and result['maximum_total_value']<=1+2e-4))
    path=None
    if keep_path:
        snapshots=[]
        for j,tau in enumerate(sol.t):
            fields=unpack(tau,sol.y[:,j]);der,_=derivatives(fields);qqi=(1+der[:,1])/(2*der[:,0]);qqj=(1-der[:,1])/(2*der[:,0]);snapshots.append(dict(x=fields[1:-1,0],qi=qqi,qj=qqj,own_value=(fields[1:-1,1]+d[1:-1])/(2*ei),other_value=(fields[1:-1,1]-d[1:-1])/(2*ej)))
        path=dict(times=sol.t,snapshots=snapshots)
    return result,path
```
