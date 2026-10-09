# 联合设计奖金、期限与精度：二阶近似下的可行方案

本方案已纳入 `PaperJK7.tex` 第 2.1 节及附录：Theorem 1 保留一阶形式，附录新增二阶努力 Lemma，Lemma 2 补充二阶最佳产出，联合设计 Proposition 替换旧稳态证明。保留一般初始方差 $S_0>0$，不令其随精度 $\lambda$ 变化。以下联合设计结论针对明确写出的二阶近似目标；不声称已经证明精确非线性均衡目标下的全局设计结论。正式表述和证明以论文为准。

## 1. 为什么一阶目标不够

令 $\mu=\tilde y_0$、$c_i=c_j=c$、$\alpha=\theta/c$，并记

$$
Q_s=S_s+\sigma^2(T-s),\qquad q=Q_0=S_0+\sigma^2T,
$$

$$
h(y,Q)=\frac{1}{\sqrt{2\pi Q}}\exp\!\left(-\frac{y^2}{2Q}\right),
$$

$$
\beta(\mu,q)=\sqrt q\,\phi(\mu/\sqrt q)
+|\mu|\left[\Phi(|\mu|/\sqrt q)-\frac12\right].
$$

原一阶预期最佳产出为 $\beta(\mu,q)+\alpha T h(\mu,q)$，完全不依赖 $\lambda$。因此，仅在一阶目标中加入递增精度成本，会选择最低可行精度，不能产生正的内点精度。

## 2. 二阶产出项

以下展开先固定 $T,\sigma,S_0,\lambda>0$。Theorem 1 的压缩映射估计允许对积分方程再迭代一次。令 $U^k=V^k/\theta$，则

$$
U^i=v+\alpha f+O(\alpha^2),\qquad
U^j=1-v+\alpha f+O(\alpha^2),
$$

其中 $v(y,s)=\Phi(y/\sqrt{Q_s})$，且

$$
f(y,s)=-\frac12\int_s^T P_{s,u}\bigl[h(\cdot,Q_u)^2\bigr](y)\,du.
$$

这里 $P_{s,u}$ 是方差为 $Q_s-Q_u$ 的高斯卷积。原因是：两条归一化 HJB 方程中的一阶非线性项均为 $-h^2/2$，且一阶修正的终端值为零。高斯卷积给出

$$
f_y(y,s)=\int_s^T
\frac{y\exp[-y^2/(2Q_s-Q_u)]}
{2\pi\sqrt{Q_u}(2Q_s-Q_u)^{3/2}}\,du.
$$

因此

$$
m_i=\alpha h+\alpha^2 f_y+O(\alpha^3),\qquad
m_j=\alpha h-\alpha^2 f_y+O(\alpha^3).
$$

二阶修正在总努力中抵消，却在差距漂移中保留下来：领先者与落后者的努力差异会影响最佳产出。

令 $X_s\sim N(\mu,q-Q_s)$ 为零努力下的公共后验均值。定义

$$
\mathcal R(\mu,T,\lambda)
=\int_0^T\!\int_s^T
\mathbb E\!\left[
\frac{X_s[2\Phi(X_s/\sqrt{Q_s})-1]
\exp[-X_s^2/(2Q_s-Q_u)]}
{2\pi\sqrt{Q_u}(2Q_s-Q_u)^{3/2}}
\right]du\,ds.
$$

二阶预期最佳产出为

$$
\mathcal B^{(2)}(\mu,\theta,T,\lambda)
=\beta(\mu,q)+\frac{\theta T}{c}h(\mu,q)
+\frac{\theta^2}{c^2}\mathcal R(\mu,T,\lambda).
$$

对每组固定的正参数，精确产出与上述表达式之差为 $O(\alpha^3)$。这个余项结论不自动一致覆盖 $\lambda\downarrow0$、$\lambda\to\infty$ 或无界的 $T$。

**二阶项的验证。** 对零努力基准 $\beta(y,Q_s)$，有 $\partial_y\beta=\Phi(y/\sqrt{Q_s})-1/2$，且它满足零漂移的后向方程。对有努力的后验过程使用 Itô 公式，并加上平均分数的努力增量，可将预期最佳产出写成基准值加

$$
\mathbb E\int_0^T\left[
\frac{m_i+m_j}{2}
+(m_i-m_j)\left(\Phi(\tilde y_s/\sqrt{Q_s})-\frac12\right)
\right]ds.
$$

代入努力展开后，总努力贡献为 $\alpha T h(\mu,q)+O(\alpha^3)$，差距贡献为 $\alpha^2\mathcal R+O(\alpha^3)$。由于对称成本下的状态漂移为 $O(\alpha^2)$，在这些项中用零努力状态代替有努力状态，不改变二阶系数。

## 3. 精度系数的界与零点导数

因为 $x[2\Phi(x/\sqrt{Q_s})-1]\geq0$，有 $\mathcal R\geq0$。又因为 $Q_u\geq\sigma^2(T-u)$、$2Q_s-Q_u\geq Q_s\geq\sigma^2(T-s)$，利用

$$
\sup_x |x|e^{-x^2/D}=\sqrt{D/(2e)}
$$

可得对所有有限精度均成立的界

$$
0\leq\mathcal R(\mu,T,\lambda)
\leq\frac{2\sqrt T}{\pi\sqrt{2e}\,\sigma^3}.
$$

将 $\lambda=0$ 定义为无观测边界：$S_s=S_0+\sigma^2s$，$Q_s=q$，$X_s=\mu$。于是二阶系数连续延拓到

$$
\mathcal R(\mu,T,0)
=\frac{T^2\mu[2\Phi(\mu/\sqrt q)-1]e^{-\mu^2/q}}
{4\pi q^2}.
$$

为了计算右导数，定义

$$
H(s)=\int_0^s(S_0+\sigma^2r)^2dr,
\qquad Q_s=q-\lambda H(s)+O(\lambda^2).
$$

记 $z=|\mu|/\sqrt q$，以及

$$
L=\int_0^T(T-s)H(s)ds
=\frac{S_0^2T^3}{6}+\frac{S_0\sigma^2T^4}{12}+\frac{\sigma^4T^5}{60},
$$

$$
M=\int_0^T sH(s)ds
=\frac{S_0^2T^3}{3}+\frac{S_0\sigma^2T^4}{4}+\frac{\sigma^4T^5}{15}.
$$

对双重积分中的光滑高斯期望求右导数，得到

$$
\mathcal R_\lambda(\mu,T,0+)
=\frac{e^{-z^2}}{2\pi q^{5/2}}
\left[
2\phi(z)(1-2z^2)L
-z[2\Phi(z)-1](1-z^2)M
\right].
$$

可直接核查这一求导：若双重积分的被积函数记为 $J(x,Q_s,Q_u)$，高斯期望的右导数等于
$H(s)(J_{xx}/2-J_{Q_s})-H(u)J_{Q_u}$，在 $(x,Q_s,Q_u)=(\mu,q,q)$ 处计算，然后积分。

特别地，$\mu=0$ 时

$$
\mathcal R_\lambda(0,T,0+)
=\frac{L}{\pi\sqrt{2\pi}\,q^{5/2}}>0\qquad(T>0).
$$

一个显式但保守的充分条件是

$$
\boxed{\mu^2\leq S_0/6\quad\Longrightarrow\quad
\mathcal R_\lambda(\mu,T,0+)>0\quad\text{对所有 }T>0.}
$$

证明如下。此时 $z^2\leq1/6$，而 $M\leq4L$、$2\Phi(z)-1\leq2z\phi(0)$。方括号中的表达式至少为

$$
L\phi(0)\left[\frac43e^{-1/12}-\frac{10}{9}\right]
\geq\frac{L\phi(0)}9>0.
$$

这个条件不是必要条件，也不表示精度收益随初始差距单调变化。

## 4. 候选联合设计问题及存在性命题

建议采用一个明确的可行设计集合：$\theta\in[0,\theta_{\max}]$、$T\in[0,T_{\max}]$、$\lambda\geq0$，其中两个上界有限且为正。奖金上界应体现小激励近似的用途，而不是允许优化程序任意外推。设 $u$ 连续、可微且 $u'>0$，期限成本 $D$ 连续，精度成本取 $\kappa\lambda^2/2$，$\kappa>0$。目标为

$$
\max_{\theta,T,\lambda}
J^{(2)}=u(\mathcal B^{(2)})-\theta-D(T)-\frac\kappa2\lambda^2.
$$

在 $T=0$ 处将努力贡献设为零。

**候选命题。** 此二阶近似设计问题存在全局最优解，且每个最优解的精度有限。若另外满足：

1. $\mu^2\leq S_0/6$；
2. 在无奖金设计的某个最优期限 $T_0\in\arg\max_{0\leq T\leq T_{\max}}[u(\beta(\mu,S_0+\sigma^2T))-D(T)]$ 处，

$$
u'(\beta(\mu,S_0+\sigma^2T_0))
\frac{T_0}{c}h(\mu,S_0+\sigma^2T_0)>1,
$$

则每个联合最优解都满足 $\theta^*>0$、$T^*>0$ 和 $0<\lambda^*<\infty$。

**证明。** 由上一节的界，$\mathcal B^{(2)}$ 在有限的奖金、期限范围内对所有 $\lambda\geq0$ 一致有界。因此 $J^{(2)}\to-\infty$ 当 $\lambda\to\infty$，可以将寻找最优解的精度范围限制到某个有限区间。目标函数连续，故最优解存在且精度有限。

当 $\theta=0$ 时，最佳精度为零，最大目标值就是上述无奖金问题的值。条件 2 意味着在 $(0,T_0,0)$ 处增加一小笔奖金能够严格提高目标，因此任何联合最优解都必须有 $\theta^*>0$。$T=0$ 时支付奖金没有产出收益，故还必须有 $T^*>0$。

最后，对于任意 $\theta,T>0$，条件 1 给出

$$
\partial_\lambda J^{(2)}(\theta,T,0+)
=u'(\mathcal B^{(2)})\frac{\theta^2}{c^2}
\mathcal R_\lambda(\mu,T,0+)>0.
$$

所以 $\lambda=0$ 不可能是联合最优解。结合有限性得到所述结论。

若三个设计变量均为内点，则必要条件为

$$
u'(\mathcal B^{(2)})\left[
\frac{T}{c}h(\mu,q)+\frac{2\theta}{c^2}\mathcal R
\right]=1,
$$

$$
u'(\mathcal B^{(2)})\partial_T\mathcal B^{(2)}=D'(T),
$$

$$
u'(\mathcal B^{(2)})\frac{\theta^2}{c^2}\mathcal R_\lambda
=\kappa\lambda.
$$

这些条件不保证联合凹性或唯一性；奖金、期限在边界时应使用相应的单边条件。

## 5. 适用边界与后续正文处理

- 有限精度来自成本的无界增长；正精度来自零点的正边际收益与零边际成本。二者需要分别证明。
- 若改用线性精度成本，正精度需要边际收益超过正的成本斜率，不再由差距条件单独保证。
- 正文现已明确增加奖金和期限的有限可行范围。若希望允许它们无界，需另行规定足以控制增长的收益、期限成本及均衡适用范围。
- 以上存在性命题针对二阶近似目标。将其提升为精确均衡下的设计结论，还需要对参数边界附近的均衡、精度导数和余项进行一致控制。Theorem 1 当前的充分条件在 $\lambda\downarrow0$ 或 $\lambda\to\infty$ 时不自动提供这种控制。
- 旧稳态联合设计证明已替换。第 5 章保留原数值结果，明确为独立的无精度成本、稳态校准基准，并用 $\mathcal B^{\mathrm{cal}}$ 区别于本命题的目标；不再声称其精度公式由新命题推出。
- 正文可以只保留二阶产出表达、联合设计目标和命题；$\mathcal R$ 的积分、零点导数、充分条件的估计及存在性证明应置于附录。

计算核对：`tests/test_joint_design.py` 用确定性高斯积分与右侧有限差分核对零精度系数及导数（12 组参数），并在 45 组参数与精度组合下检查非负性和系数上界。这些核对仅用于排查推导错误，不代替解析证明，没有重跑论文的实证或 Synthetic Experiments。
