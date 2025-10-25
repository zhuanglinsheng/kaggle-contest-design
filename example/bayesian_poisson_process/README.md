# Bayesian Inference of Poisson Process: Homogeneous and Inhomogeneous Using Stan

## Requirements (Python packages)

- cmdstanpy
- numpy
- scipy
- matplotlib

For an improved display, consider installing `ipywidgets`.

## Implementations

### Model 1: [homogeneous Poisson Process](./model_1/test_poisson.ipynb)

- Data: Poisson events $(t_1, t_2, ..., t_N)\subset[0, T)$.
- Parameter: Poisson arrival rate $\lambda > 0$.
- Log Likelihood: $\log\mathcal{L}=N\log\lambda-\lambda T$.

### Model 2: [inhomogeneou Poisson Process](./model_2/test_nhpp.ipynb)

- Data: 1) Inhomogeneou Poisson events $(t_1, t_2, ..., t_N)\subset[0, T)$, and 2) a time series of per-hour efforts data $(m_0, m_{\Delta}, m_{2\Delta}, ..., m_T)$.
- Parameter: Intensity to effort rate $r>0$.

  - Suppose the inhomogeneous intensity $\lambda(t) = r\cdot m_t$.
- Likelihood: $\log\mathcal{L}=\sum_{k=1}^{N}\log\lambda(t_k) - \int^T_0\lambda(s)ds$.

