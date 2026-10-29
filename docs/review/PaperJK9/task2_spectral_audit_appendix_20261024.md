# Task 2：谱方法只读试算附录

日期：2026-10-24。

此文件保留会话内存中执行的数值试算方法，便于复核；它不是生产求解器，尚未用于修改论文或原始分类。代码仅保存在报告代码块中，不新增或修改实现文件。试算结果须同时接受谱网格、时间步、谱截断阈值、空间截断范围及独立最佳回应检查。有限 Fourier 截断和去噪阈值可能改变解；单次成功或收益残差很小不构成认证。

字段 `precision=0` 表示退化方程的数值试算，不自动证明正精度趋零的均衡收敛。零反馈级数仅作独立数值参考；有限项相符不替代收敛或存在性证明。

## 归一化与方程

取 `q0=1+sigma²*T`、`x=y/sqrt(q0)`、反向时间 `tau∈[0,1]`、`e=alpha*T/q0`。`p=∂U_i/∂x`、`n(x)=p(-x)`；`Q=S(t)+sigma²*T/q0*(1-t)`，`t=1-tau`；`D=lambda*(S(t)*q0)²*T/(2*q0)`。计算：

- `p_tau=D*p_xx+e*∂x(p²/2-p*n)`；
- `W_tau=D*W_xx+e*(p-n)*W_x+e*(p-n)*G_Q-e*p²/2`，终端 `W=0`；参赛者线性评价为 `Phi(x/sqrt(Q))+W`；
- `H_tau=D*H_xx+e*(p-n)*H_x+e*(p+n)/2+e*(p-n)*(Phi(x/sqrt(Q))-1/2)`，终端 `H=0`；产出为 `sqrt(q0)*(1/sqrt(2*pi)+H(1,0))`。

梯度、线性收益评价和产出贡献分别推进；从密度反积分重建的 `U_i` 与线性评价作比较。频域使用 2/3 去混叠和显式的小系数阈值，因此必须单独检查阈值敏感性。时间推进为 RK4。

## 会话内存试算代码

以下函数只返回内存结果；自身不写文件。执行时使用 `PYTHONDONTWRITEBYTECODE=1`。

```python
import math
import time
import numpy as np
from scipy.special import ndtr

def spectral_audit(sigma, alpha, T, precision, N=1024, steps=2000,
                   tolerance=1e-14, extent=10., keep_path=False):
    if precision < 0 or N % 2 or steps < 1:
        raise ValueError("Nonnegative precision, an even grid, and positive steps required")
    q0 = 1 + sigma*sigma*T
    incentive = alpha*T/q0
    x = np.arange(N)*(2*extent/N)-extent
    k = np.fft.fftfreq(N, d=2*extent/N)*2*math.pi
    mirror = (-np.arange(N)) % N
    alias_mask = np.abs(np.fft.fftfreq(N)*N) > N/3
    if precision:
        rate = sigma*math.sqrt(precision)*T
        steady = sigma/math.sqrt(precision)
        def S(t):
            z = math.tanh(rate*t)
            return steady*(1+steady*z)/(steady+z)/q0
    else:
        def S(t):
            return (1+sigma*sigma*T*t)/q0
    def clean(v):
        v = v.copy()
        v[alias_mask] = 0
        v[np.abs(v) < tolerance*N] = 0
        return v
    v = np.zeros((N, 3), dtype=complex)
    phase = np.where(np.arange(N) % 2, -1., 1.)
    v[:, 0] = N/(2*extent)*np.exp(-.5*S(1)*k*k)*phase
    times = [0.]
    paths = []
    if keep_path:
        p = np.fft.ifft(v[:, 0]).real
        paths.append(incentive*np.r_[p, p[0]])
    path_minimum_density = math.inf
    path_maximum_density = 0.
    path_mass_error = 0.
    def rhs(tau, state):
        nonlocal path_minimum_density, path_maximum_density, path_mass_error
        state = clean(state)
        f = np.fft.ifft(state, axis=0).real
        p = f[:, 0]
        path_minimum_density = min(path_minimum_density, float(p.min()))
        path_maximum_density = max(path_maximum_density, float(p.max()))
        path_mass_error = max(path_mass_error, abs(float(p.sum()*2*extent/N)-1))
        opponent = p[mirror]
        difference = p-opponent
        st = S(1-tau)
        Q = st + sigma*sigma*T/q0*tau
        D = precision*(st*q0)**2*T/q0/2
        derivative = np.fft.ifft(1j*k[:, None]*state, axis=0).real
        gaussian = np.exp(-x*x/(2*Q))/math.sqrt(2*math.pi*Q)
        result = -D*k[:, None]**2*state
        result[:, 0] += incentive*1j*k*np.fft.fft(.5*p*p-p*opponent)
        result[:, 1] += np.fft.fft(
            incentive*difference*derivative[:, 1]
            +incentive*difference*gaussian-incentive*p*p/2)
        result[:, 2] += np.fft.fft(
            incentive*difference*derivative[:, 2]
            +incentive*(p+opponent)/2
            +incentive*difference*(ndtr(x/math.sqrt(Q))-.5))
        return clean(result)
    start = time.monotonic()
    dt = 1/steps
    for j in range(steps):
        tau = j*dt
        a1 = rhs(tau, v)
        a2 = rhs(tau+dt/2, v+dt*a1/2)
        a3 = rhs(tau+dt/2, v+dt*a2/2)
        a4 = rhs(tau+dt, v+dt*a3)
        v = clean(v+dt*(a1+2*a2+2*a3+a4)/6)
        if not np.isfinite(v).all() or np.max(np.abs(v)) > 1e8:
            raise RuntimeError("Spectral path unstable at reverse time "+str(tau))
        if keep_path and ((j+1) % 4 == 0 or j+1 == steps):
            p = np.fft.ifft(v[:, 0]).real
            paths.append(incentive*np.r_[p, p[0]])
            times.append((j+1)*dt)
    f = np.fft.ifft(v, axis=0).real
    p = f[:, 0]
    primitive_hat = v[:, 0].copy()
    primitive_hat[0] = 0
    primitive_hat[1:] /= 1j*k[1:]
    primitive = np.fft.ifft(primitive_hat).real
    value = primitive-primitive[0]+(x+extent)/(2*extent)
    physical_value = ndtr(x)+f[:, 1]
    result = dict(
        sigma=sigma, alpha=alpha, T=T, precision=precision,
        normalized_exposure=incentive, N=N, steps=steps,
        tolerance=tolerance, extent=extent,
        B=float(math.sqrt(q0)*(1/math.sqrt(2*math.pi)+f[N//2, 2])),
        physical=float(physical_value[N//2]), value=float(value[N//2]),
        global_residual=float(np.max(np.abs(value-physical_value))),
        mass=float(p.sum()*2*extent/N),
        minimum_density=float(p.min()), maximum_density=float(p.max()),
        maximum_total_value=float(np.max((value+value[mirror])[1:])),
        minimum_density_all_rk_stages=path_minimum_density,
        maximum_density_all_rk_stages=path_maximum_density,
        maximum_mass_error_all_rk_stages=path_mass_error,
        seconds=time.monotonic()-start)
    path = None
    if keep_path:
        path = dict(x=np.r_[x, extent], times=np.array(times),
                    policies=np.asarray(paths).T)
    return result, path
```

