# Task 3：非对称成本的弱反馈谱试算

日期：2026-10-25。报告原型，不修改生产实现。取 q0=S0+sigma²*T，x=y/sqrt(q0)，ei=theta*T/(ci*q0)、ej 同理。分别推进两位参赛者的正梯度 pi、pj：

`pi_tau=D*pi_xx+partial_x(ei*pi²/2-ej*pi*pj)`；
`pj_tau=D*pj_xx+partial_x(ei*pi*pj-ej*pj²/2)`。

策略为 qi=ei*pi、qj=ej*pj；原模型净漂移为 qi-qj。另独立推进两条原策略收益的线性方程及一条组织者产出方程，与梯度反积分比较。初始方差固定为原估计值，不随候选精度改变；可评价非零初始均值。

RK4、2/3 去混叠与小系数截断均沿用对称基准；全阶段负密度超过 .001 立即拒绝，不截断负策略以制造成功。最终筛查还要求密度>=-1e-8、质量误差<=1e-8、总价值<=1+1e-7、独立收益残差<=2e-6。更强激励是否能通过须看实际结果。

小正精度试算不证明零精度极限，更不直接认证后验平均质量；还需空间、时间、截断参数、独立最佳回应及正精度延拓检查。

初始均值可能不落在网格节点上；输出用 Fourier 系数直接评价该点，避免线性插值的二阶误差。早期插值版本的源代码已保存在原试算 JSON 快照，不混合成同一收敛序列。

```python
import math,time
import numpy as np
from scipy.special import ndtr

def asymmetric_spectral(sigma,theta,T,precision,ci,cj,S0,mu,N=2048,steps=4000,tolerance=1e-14,extent=10.,keep_path=False):
    if min(sigma,theta,T,ci,cj,S0)<=0 or precision<0 or N%2:raise ValueError('Invalid primitives')
    q0=S0+sigma*sigma*T;ei=theta*T/(ci*q0);ej=theta*T/(cj*q0);scale=math.sqrt(q0)
    x=np.arange(N)*2*extent/N-extent;k=np.fft.fftfreq(N,d=2*extent/N)*2*math.pi
    alias=np.abs(np.fft.fftfreq(N)*N)>N/3
    if abs(mu/scale)>extent-2:raise ValueError('Initial mean too close to boundary')
    def S(t):
        rate=sigma*math.sqrt(precision)*T*t
        tc=math.tanh(rate)/rate if rate else 1.
        return (S0+sigma*sigma*T*t*tc)/(1+precision*S0*T*t*tc)/q0
    def clean(v):
        v=v.copy();v[alias]=0;v[np.abs(v)<tolerance*N]=0;return v
    v=np.zeros((N,5),dtype=complex);phase=np.where(np.arange(N)%2,-1.,1.)
    v[:,0]=N/(2*extent)*np.exp(-.5*S(1)*k*k)*phase;v[:,1]=v[:,0]
    minimum=math.inf;masserror=0.;maximum=0.;times=[0.];paths=[]
    if keep_path:
        pp=np.fft.ifft(v,axis=0).real;paths.append(np.column_stack((ei*np.r_[pp[:,0],pp[0,0]],ej*np.r_[pp[:,1],pp[0,1]])))
    def rhs(tau,state):
        nonlocal minimum,masserror,maximum
        state=clean(state);f=np.fft.ifft(state,axis=0).real;pi=f[:,0];pj=f[:,1]
        minimum=min(minimum,float(min(pi.min(),pj.min())));maximum=max(maximum,float(max(pi.max(),pj.max())))
        masserror=max(masserror,float(np.max(np.abs(f[:,:2].sum(axis=0)*2*extent/N-1))))
        if minimum < -1e-3:raise RuntimeError('Nonphysical negative density at reverse time '+str(tau))
        qi=ei*pi;qj=ej*pj;drift=qi-qj;st=S(1-tau);Q=st+sigma*sigma*T/q0*tau;D=precision*(st*q0)**2*T/q0/2
        derivative=np.fft.ifft(1j*k[:,None]*state,axis=0).real
        gaussian=np.exp(-x*x/(2*Q))/math.sqrt(2*math.pi*Q);probability=ndtr(x/math.sqrt(Q))
        r=-D*k[:,None]**2*state
        r[:,0]+=1j*k*np.fft.fft(.5*ei*pi*pi-ej*pi*pj)
        r[:,1]+=1j*k*np.fft.fft(ei*pi*pj-.5*ej*pj*pj)
        r[:,2]+=np.fft.fft(drift*derivative[:,2]+drift*gaussian-.5*qi*qi/ei)
        r[:,3]+=np.fft.fft(drift*derivative[:,3]-drift*gaussian-.5*qj*qj/ej)
        r[:,4]+=np.fft.fft(drift*derivative[:,4]+.5*(qi+qj)+drift*(probability-.5))
        return clean(r)
    start=time.monotonic();dt=1/steps
    for j in range(steps):
        tau=j*dt;a=rhs(tau,v);b=rhs(tau+dt/2,v+dt*a/2);c=rhs(tau+dt/2,v+dt*b/2);d=rhs(tau+dt,v+dt*c)
        v=clean(v+dt*(a+2*b+2*c+d)/6)
        if not np.isfinite(v).all() or np.max(np.abs(v))>1e8:raise RuntimeError('Unstable path at '+str(tau))
        if keep_path and ((j+1)%4==0 or j+1==steps):
            pp=np.fft.ifft(v,axis=0).real;times.append((j+1)*dt)
            paths.append(np.column_stack((ei*np.r_[pp[:,0],pp[0,0]],ej*np.r_[pp[:,1],pp[0,1]])))
    f=np.fft.ifft(v,axis=0).real;cdf=[]
    for col in (0,1):
        vh=v[:,col].copy();vh[0]=0;vh[1:]/=1j*k[1:];primitive=np.fft.ifft(vh).real
        cdf.append(primitive-primitive[0]+(x+extent)/(2*extent))
    ui=cdf[0];uj=1-cdf[1];physicali=ndtr(x)+f[:,2];physicalj=ndtr(-x)+f[:,3]
    baseline=scale/math.sqrt(2*math.pi)*math.exp(-mu*mu/(2*q0))+mu*(float(ndtr(mu/scale))-.5)
    result=dict(B=baseline+scale*float(np.real(np.dot(v[:,4],np.exp(1j*k*(mu/scale+extent))))/N),baseline=baseline,
       sigma=sigma,theta=theta,T=T,precision=precision,ci=ci,cj=cj,S0=S0,mu=mu,
       normalized_incentive_i=ei,normalized_incentive_j=ej,N=N,steps=steps,tolerance=tolerance,extent=extent,
       maximum_player_payoff_residual=float(max(np.max(np.abs(ui-physicali)),np.max(np.abs(uj-physicalj)))),
       minimum_density_all_stages=minimum,maximum_density_all_stages=maximum,maximum_mass_error_all_stages=masserror,
       maximum_total_value=float(np.max((ui+uj)[1:])),minimum_value=float(min(ui.min(),uj.min())),
       seconds=time.monotonic()-start)
    result['physical_screen_passed']=bool(minimum>=-1e-8 and masserror<=1e-8 and result['maximum_total_value']<=1+1e-7 and result['maximum_player_payoff_residual']<=2e-6 and result['B']>=0)
    path=None
    if keep_path:path=dict(x=np.r_[x,extent],times=np.asarray(times),policies=np.asarray(paths).transpose(1,0,2))
    return result,path
```
