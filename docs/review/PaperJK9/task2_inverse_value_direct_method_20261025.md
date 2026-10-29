# Task 2：直接逆价值坐标的局部离散校验

日期：2026-10-25。只在报告中保存并执行原型，不改生产实现或论文。此前 inverse_logphysical 已尝试同类逆坐标且未收敛，本原型使用不同的直接 Y/f/H、logit 端点网格和稀疏 BDF；不是把旧失败路线改名当作已解决。

方程推导见 Task 2 持续报告。终端 Y 为 Gaussian 分位数，f 为仿射函数，H 为零。有限端点用无努力 Gaussian 尾近似，必须另检查端点范围。局部非均匀价值网格的差分对仿射函数精确，避免用不一致的解析/离散分母制造负策略。

校验顺序：先对照已稳定的弱激励谱结果，再对照 lambda=.01 已修复候选；通过后再试原失败候选。需要同时检查反解单调、控制非负、空间/时间/端点收敛和原状态独立收益/产出评价。同坐标的代数恒等式不计作独立最佳回应证据。

```python
import math,time
import numpy as np
from scipy.special import expit,ndtri,ndtr
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags,kron,csr_matrix

def direct_inverse(sigma,theta,T,precision,ci=1.,cj=1.,S0=1.,mu=0.,nodes=401,extent=16.,rtol=1e-7,atol=1e-11,max_calls=40000,keep_path=False):
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
    calls=0;smallest_slope=math.inf;start=time.monotonic()
    def rhs(tau,y):
        nonlocal calls,smallest_slope
        calls+=1
        if calls>max_calls or time.monotonic()-start>100:raise RuntimeError('Inverse integration budget exhausted at '+str(tau))
        f=unpack(tau,y);first,second=derivatives(f);J=first[:,0];fd=first[:,1];Hd=first[:,2]
        smallest_slope=min(smallest_slope,float(J.min()))
        if np.min(J)<=0 or not np.isfinite(f).all():raise RuntimeError('Non-monotone inverse state at '+str(tau))
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
    path=None
    if keep_path:
        snapshots=[]
        for j,tau in enumerate(sol.t):
            fields=unpack(tau,sol.y[:,j]);der,_=derivatives(fields);qqi=(1+der[:,1])/(2*der[:,0]);qqj=(1-der[:,1])/(2*der[:,0]);snapshots.append(dict(x=fields[1:-1,0],qi=qqi,qj=qqj,own_value=(fields[1:-1,1]+d[1:-1])/(2*ei),other_value=(fields[1:-1,1]-d[1:-1])/(2*ej)))
        path=dict(times=sol.t,snapshots=snapshots)
    return result,path
```
