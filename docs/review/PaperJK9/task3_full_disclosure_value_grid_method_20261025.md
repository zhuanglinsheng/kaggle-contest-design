# Task 3：完全披露输出方程的价值坐标求解

日期：2026-10-25。后验批次部分强激励抽样的 z 坐标边值求解达到节点上限，原失败记录保留，不删除这些抽样。

利用已推导关系 `d_zz=-z*d_z-f'*d_z²`，将 h 的输出方程改写为

`h_dd+f'*h_d-h/(d_z)²+[1+2*f'*(Phi(z)-1/2)]/d_z=0`。

在 d∈(-w_j,w_i) 内用 logit 网格，窄 z 尖峰在 d 坐标中得到展开。K 端点差用双曲函数差的稳定恒等式计算，分别用左右尾概率反解 z，避免减去接近 1 的概率。输出线性算子用非均匀网格求解，检查非负系数与行缩放残差；先验平均采用 PCHIP 与 Gaussian 求积。

这是一条与原 z 坐标自适应 BVP 独立的数值离散路线。必须检查 N、端点截断及求积加密，并与已成功的 z 坐标结果比较。它不改变原策略公式或原模型，仍不提供严格 PDE 误差界。

边界和先验尾部另外记录：由总奖金额与双方非负价值，条件产出增益满足 `h(z)<=sqrt(2*max(w_i,w_j)*(1-U_i-U_j))`。截断域两端可用解析价值计算该预算；线性算子的负零阶项给出最大值原理，从而限制将边界设为零在域内的影响。先验域外部分用统一界 `sqrt(2*max(w_i,w_j))` 乘正态尾概率控制。记录的数值预算含 1e-12 价值舍入保护量，不替代区间算术；有限差分和 Gaussian 求积仍需独立加密。

```python
import math
import numpy as np
from scipy.special import expit,ndtr,ndtri,roots_hermitenorm
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded

def full_asymmetric_value_grid(theta,T,sigma,ci,cj,S0,mu,N=8001,cutoff=24.,quadrature=160):
    wi=theta/(ci*sigma*sigma);wj=theta/(cj*sigma*sigma)
    if min(wi,wj,T,sigma,S0)<=0:raise ValueError('Positive primitives required')
    b=math.log(-math.expm1(-wi))-math.log(-math.expm1(-wj))
    d=-wj+(wi+wj)*expit(np.linspace(-cutoff,cutoff,N))
    a=d+wj;c=wi-d
    lower=.5*(a+2*np.cosh((d-wj+2*b)/2)*np.sinh(a/2))
    upper=.5*(c+2*np.cosh((wi+d+2*b)/2)*np.sinh(c/2))
    span=float(((wi+math.sinh(wi+b))-(-wj+math.sinh(b-wj)))/2)
    z=np.where(lower<=upper,ndtri(lower/span),-ndtri(upper/span))
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z)>0):raise RuntimeError('Unresolved transformed profile coordinates')
    q=span*np.exp(-z*z/2)/math.sqrt(2*math.pi)/((1+np.cosh(d+b))/2)
    fp=np.tanh((d+b)/2);hm=np.diff(d)[:-1];hp=np.diff(d)[1:]
    left=2/(hm*(hm+hp))-fp[1:-1]/(hm+hp)
    right=2/(hp*(hm+hp))+fp[1:-1]/(hm+hp)
    if min(left.min(),right.min())<=0:raise RuntimeError('Non-monotone transformed operator')
    diagonal=left+right+1/q[1:-1]**2
    source=(1+2*fp[1:-1]*(ndtr(z[1:-1])-.5))/q[1:-1]
    band=np.zeros((3,N-2));band[1]=1;band[0,1:]=-right[:-1]/diagonal[:-1];band[2,:-1]=-left[1:]/diagonal[1:]
    h=np.r_[0.,solve_banded((1,1),band,source/diagonal,check_finite=False),0.]
    residual=h[1:-1]-left/diagonal*h[:-2]-right/diagonal*h[2:]-source/diagonal
    gh,weights=roots_hermitenorm(quadrature);scale=sigma*math.sqrt(T);xx=(mu+math.sqrt(S0)*gh)/scale
    inside=(xx>=z[0])&(xx<=z[-1]);hh=np.zeros_like(xx)
    hh[inside]=PchipInterpolator(z,h)(xx[inside])
    gain=scale*float(weights@hh)/math.sqrt(2*math.pi)
    prior_sd=math.sqrt(S0)/scale;prior_mean=mu/scale
    tail_probability=float(ndtr((z[0]-prior_mean)/prior_sd)+ndtr((prior_mean-z[-1])/prior_sd))
    prior_tail_bound=scale*math.sqrt(2*max(wi,wj))*tail_probability
    lc=lambda a:np.logaddexp(a,-a)-math.log(2)
    ff=wj+2*(lc((d[[0,-1]]+b)/2)-lc((b-wj)/2))
    value_sum=(ff+d[[0,-1]])/(2*wi)+(ff-d[[0,-1]])/(2*wj)
    boundary_budget=math.sqrt(2*max(wi,wj)*(max(0.,float(1-value_sum.min()))+1e-12))
    total=S0+sigma*sigma*T;baseline=math.sqrt(total/(2*math.pi))*math.exp(-mu*mu/(2*total))+mu*(float(ndtr(mu/math.sqrt(total)))-.5)
    return dict(B=baseline+gain,baseline=baseline,gain=gain,N=N,cutoff=cutoff,quadrature=quadrature,wi=wi,wj=wj,
        domain=[float(z[0]),float(z[-1])],
        prior_tail_probability=tail_probability,prior_tail_output_upper_bound=prior_tail_bound,
        boundary_output_budget=scale*boundary_budget,
        boundary_value_roundoff_guard=1e-12,
        quadrature_weight_outside_domain=float(weights[~inside].sum()/math.sqrt(2*math.pi)),
        minimum_gain_profile=float(h.min()),maximum_scaled_operator_residual=float(np.max(np.abs(residual))))
```
