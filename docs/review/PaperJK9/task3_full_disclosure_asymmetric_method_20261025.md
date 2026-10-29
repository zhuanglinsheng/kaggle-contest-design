# Task 3：非对称成本下完全披露的解析策略与输出核验

日期：2026-10-25。此推导和试算原型仅保存在审查报告，尚未修改论文或生产实现。

## 从原 HJB 推导

取 `w_i=theta/(c_i*sigma²),w_j=theta/(c_j*sigma²)`，缩放值 `v_i=w_i*U_i,v_j=w_j*U_j`。令 `F=v_i+v_j,d=v_i-v_j`。完全披露下扩散为 sigma²；原两条 HJB 给出

`F_tau=(sigma²/2)F_yy+(sigma²/4)(3F_y²-d_y²)`，
`d_tau=(sigma²/2)d_yy+(sigma²/2)F_y*d_y`。

若 `F=f(d)`，一致性要求 `f''=(1-f'²)/2`。写 `f'=tanh((d+b)/2)`，端点条件为 `f(-w_j)=w_j,f(w_i)=w_i`。其解给出

`b=2*atanh[tanh((w_i-w_j)/4)/tanh((w_i+w_j)/4)]-(w_i-w_j)/2`。

稳定的等价表达式为 `b=log(1-exp(-w_i))-log(1-exp(-w_j))`，实现用 expm1 避免小奖金下的相消。

令 `K(d)=(d+sinh(d+b))/2`，则 `K''=f'*K'`，因此 `K(d)` 满足热方程。终端 d 的两端为 -w_j、w_i，故

`K(d(z))=K(-w_j)+(K(w_i)-K(-w_j))*Phi(z)`，其中 `z=y/(sigma*sqrt(T-t))`。

策略为 `m_i=sigma*(1+f')*d_z/(2*sqrt(T-t))`、`m_j=sigma*(1-f')*d_z/(2*sqrt(T-t))`，均非负。对称成本时 b=0，精确退回论文的完全披露公式。该代数推导不依赖经验先验；仍须核验实现、逆映射和输出计算。

## 避免终端奇异性的线性输出方程

给定上述策略，完全披露的条件输出写成 `sigma*sqrt(T-t)*g(z)`。原输出线性方程化为

`g''+(z+2*f'*d_z)*g'-g+d_z=0`，远端 `g(z)~abs(z)/2`。

取 `g=beta(z,1)+h`，则

`h''+(z+2*f'*d_z)*h'-h+d_z+2*f'*d_z*(Phi(z)-1/2)=0`，远端 h=0。

由线性边值问题计算 h，最后在原初始先验 `N(mu,S0)` 下平均。不能把初始后验方差随候选精度改成零，也不能只评价初始均值处的完全披露输出。

以下是报告原型。计划验证：对称公式一致、零奖金/弱奖金的一阶极限、参与者交换对称性、域/容差/先验求积加密；通过前不作反事实质量认证。

```python
import math
import numpy as np
from scipy.special import ndtr, roots_hermitenorm
from scipy.integrate import solve_bvp

def full_asymmetric_output(theta,T,sigma,ci,cj,S0,mu,extent=10.,tolerance=1e-8,quadrature=80):
    if min(theta,T,sigma,ci,cj,S0)<=0:
        raise ValueError('Positive primitives required; zero prize has its analytic baseline')
    wi=theta/(ci*sigma*sigma);wj=theta/(cj*sigma*sigma)
    shift=math.log(-math.expm1(-wi))-math.log(-math.expm1(-wj))
    def K(d):return (d+np.sinh(d+shift))/2
    lo=float(K(-wj));hi=float(K(wi));span=hi-lo
    def profile(z):
        probability=ndtr(z);target=lo+span*probability
        lower=np.full_like(z,-wj);upper=np.full_like(z,wi)
        d=-wj+(wi+wj)*probability
        for _ in range(65):
            value=K(d)-target
            lower=np.where(value<0,d,lower);upper=np.where(value>=0,d,upper)
            derivative=(1+np.cosh(d+shift))/2
            candidate=d-value/derivative
            candidate=np.where((candidate>lower)&(candidate<upper),candidate,(lower+upper)/2)
            if np.max(np.abs(candidate-d))<1e-13:
                d=candidate;break
            d=candidate
        fp=np.tanh((d+shift)/2)
        derivative=span*np.exp(-z*z/2)/math.sqrt(2*math.pi)/((1+np.cosh(d+shift))/2)
        return d,derivative,fp
    def ode(z,y):
        d,dp,fp=profile(z);drift=fp*dp
        source=dp+2*drift*(ndtr(z)-.5)
        return np.vstack((y[1],y[0]-(z+2*drift)*y[1]-source))
    def bc(a,b):return np.array([a[0],b[0]])
    grid=np.linspace(-extent,extent,301)
    solution=solve_bvp(ode,bc,grid,np.zeros((2,len(grid))),tol=tolerance,max_nodes=100000)
    if not solution.success:raise RuntimeError(solution.message)
    # Initial uncertainty is revealed in the full-disclosure game; average
    # the conditional nonlinear output over the unchanged initial prior.
    nodes,weights=roots_hermitenorm(quadrature)
    scale=sigma*math.sqrt(T);x=(mu+math.sqrt(S0)*nodes)/scale
    if np.max(np.abs(x))>extent:
        raise RuntimeError('Initial quadrature extends beyond the solved state domain')
    gain=scale*float(weights@solution.sol(x)[0])/math.sqrt(2*math.pi)
    q=S0+sigma*sigma*T
    baseline=math.sqrt(q/(2*math.pi))*math.exp(-mu*mu/(2*q))+mu*(float(ndtr(mu/math.sqrt(q)))-.5)
    dcheck,dpcheck,fpcheck=profile(solution.x)
    fpp=.5*(1-fpcheck*fpcheck)
    dpp=-solution.x*dpcheck-fpcheck*dpcheck*dpcheck
    gi=(1+fpcheck)*dpcheck/(2*wi)
    gj=(fpcheck-1)*dpcheck/(2*wj)
    gii=(fpp*dpcheck*dpcheck+(1+fpcheck)*dpp)/(2*wi)
    gjj=(fpp*dpcheck*dpcheck+(fpcheck-1)*dpp)/(2*wj)
    ri=.5*gii+.5*wi*gi*gi+wj*gi*gj+.5*solution.x*gi
    rj=.5*gjj+.5*wj*gj*gj+wi*gi*gj+.5*solution.x*gj
    logcosh=lambda a:np.logaddexp(a,-a)-math.log(2)
    value_sum=wj+2*(logcosh((dcheck+shift)/2)-logcosh((shift-wj)/2))
    ui=(value_sum+dcheck)/(2*wi);uj=(value_sum-dcheck)/(2*wj)
    return dict(B=baseline+gain,baseline=baseline,gain=gain,wi=wi,wj=wj,shift=shift,
       nodes=len(solution.x),maximum_bvp_rms_residual=float(solution.rms_residuals.max()),
       maximum_boundary_residual=float(np.max(np.abs(bc(solution.y[:,0],solution.y[:,-1])))),
       minimum_profile_density=float(dpcheck.min()),maximum_profile_density=float(dpcheck.max()),
       extent=extent,tolerance=tolerance,quadrature=quadrature,
       maximum_analytic_hjb_residual=float(max(np.max(np.abs(ri)),np.max(np.abs(rj)))),
       minimum_player_value=float(min(ui.min(),uj.min())),
       maximum_player_value=float(max(ui.max(),uj.max())))
```
