# 从 HJB 出发的精确充分条件

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

## 1. 范围与未解决的问题

这里给出一个**条件性的精确命题**，不截断奖金或均衡策略。奖金阈值可以写成 theta/(c sigma^2)>2，但另需低披露和完全信息均衡的形状条件。这些形状条件尚未从原始参数推导，也未经过数值认证。因此本结果不能表述成“原模型中 w>2 自动保证有限精度”。

设 c_i=c_j=c，初始共同均值为零，真实初始差距为 N(0,S0)，S0>0 固定。比较同一个 theta,T 下的无信号与完全信息均衡。完全信息下，初始真实差距先被揭示，故初始状态仍需对 N(0,S0) 积分。不是把完全信息初始状态设为零。

## 2. 直接 HJB 恒等式

令 alpha=theta/c，d=m_i-m_j，r=(m_i+m_j)/2，a_t=lambda S_t^2，并令

\[
\mathscr L=\partial_t+d\partial_y+\tfrac12a_t\partial_{yy}.
\]

在对称成本的光滑内部均衡中，HJB 写成

\[
\mathscr L V^i=\tfrac c2m_i^2,
\qquad \mathscr L V^j=\tfrac c2m_j^2,
\qquad V_y^i=cm_i,\quad V_y^j=-cm_j.
\]

对 y 求导时必须保留漂移 d 的导数。因此

\[
\mathscr L m_i=m_i(m_j)_y,
\qquad
\mathscr L m_j=-m_j(m_i)_y.
\tag{1}
\]

例如第一式来自 c m_i(m_i)_y-c m_i d_y=c m_i(m_j)_y。式 (1) 是精确的，不是奖金展开。

对有限精度，令 Q_t=S_t+sigma^2(T-t)，h(y,t)=phi(y/sqrt(Q_t))/sqrt(Q_t)，并令 H=E int_0^T r dt。终端 r(y,T)=alpha h(y,T)，而 h 满足无漂移热方程。由 (1)、Ito 公式和时间换序，

\[
H=\alpha T h(0,0)-\alpha T D+K,
\]
\[
D=E\int_0^T d\frac{\tilde y_t}{Q_t}h(\tilde y_t,t)dt,
\quad
K=\frac12E\int_0^Tt\{m_j(m_i)_y-m_i(m_j)_y\}dt.
\tag{2}
\]

具体地，(1) 给出 H=T E r_T+K；又 E h_T=h(0,0)+E int d h_y dt=h(0,0)-D，代入即得 (2)。它分开了差距扩大造成的努力损失和对手反馈造成的修正。

不能直接假设 D 与 K 分别存在有限的完全信息极限：临近终点可能有抵消性的奇异项。以下先消去这些项，再取完全信息表达。

## 3. 完全信息下的有界组合

现在的 y 是完全观测的真实差距，a_t=sigma^2，Q_t=sigma^2(T-t)，初始 y_0~N(0,S0)。记 p(y,t) 为均衡状态密度，h(y,t)=phi(y/sqrt(Q_t))/sqrt(Q_t)。所有期望均对初始真实状态和后续噪声取平均。

对空间积分分部：

\[
m_j(m_i)_y-m_i(m_j)_y=r d_y-d r_y,
\]
\[
\int(r d_y-d r_y)p\,dy
=-2\int d r_y p\,dy-\int d r p_y\,dy.
\]

在式 (2) 中把 r_y 拆为 (r-alpha h)_y+alpha h_y，即消去临近终点的共同项。得到

\[
H_\infty=\frac{\alpha T}{\sqrt{2\pi(S_0+\sigma^2T)}}-wJ-Z,
\qquad w=\frac{\theta}{c\sigma^2},
\tag{3}
\]
\[
J=E\int_0^T d(y_t,t)y_t h(y_t,t)dt,
\]
\[
Z=\mathbb E\int_0^T t\,d(y_t,t)
\left\{\partial_y[r-\alpha h](y_t,t)
+\frac{r(y_t,t)}2\partial_y\log p(y_t,t)\right\}dt.
\tag{4}
\]

初始 Gaussian 卷积给出 E h(y_0,0)=1/sqrt(2pi(S0+sigma^2T))，因此 (3) 的领先项与无信号基准相同。

严格使用这些恒等式需要：各有限终止时刻 t<T 的均衡为经典解；空间边界项消失；相关组合可积；且终端满足 lim_{t up to T} E[r(y_t,t)-alpha h(y_t,t)]=0。可先在 [0,T-epsilon] 推导，再以这些条件取极限。并未声称已有 Theorem 1 自动保证完全信息的这些条件。

定义

\[
G=E\int_0^T d(y_t,t)
\left[\Phi(y_t/\sqrt{Q_t})-\tfrac12\right]dt,
\]
\[
B_{\rm ref}=\beta(0,S_0+\sigma^2T)
+\frac{\alpha T}{\sqrt{2\pi(S_0+\sigma^2T)}}.
\]

最佳产出恒等式遂给出精确关系

\[
\boxed{\mathcal B_\infty=B_{\rm ref}+G-wJ-Z.}
\tag{5}
\]

这里 B_ref 只是一个显式比较基准；式 (5) 对它的使用不意味着把精确均衡替换成领先阶政策。

## 4. 精确充分条件命题

除上述存在性、对称性和可积性条件外，假设：

**A. 无信号时的对手反馈。** 无信号均衡在 y=0 附近可微，并有 (m_i)_y(0,t)>=0。这表示在完全对称状态附近，稍微领先会提高玩家自己的努力。由对称性，(m_j)_y(0,t)=-(m_i)_y(0,t)。

**B. 完全信息下，努力差集中于接近的比赛状态。** d(y,t)y>=0，并且对每个 t<T，

\[
f_t(y):=\frac{d(y,t)p(y,t)}{y}\geq0,
\quad y>0,
\]

满足 f_t(y)exp(y^2/(2Q_t)) 关于 y>0 非增。此条件约束的是“努力差乘以状态密度”，不是假设真实差距有有界支持，也不是要求状态本身服从 Gaussian。对称性使其延伸为偶函数。

**C. 价值函数曲率修正。** 式 (4) 的 Z>=0，且 0<J<infty。r_y=(V^i_yy-V^j_yy)/(2c)，所以这是对精确价值函数曲率和均衡状态密度的一个加权条件。它没有被证明是原模型的自动性质。

若 w>2，则

\[
\boxed{\mathcal B(0)>\mathcal B_\infty.}
\tag{6}
\]

当 w=2 时，若 Z>0 或下述 G<2J 严格成立，结论也成立。若精确 B(lambda) 在 lambda>=0 上连续，且其高精度极限确为这里的 B_infty，则存在有限精度的全局最大化解。此外，一些严格为正的有限精度也会优于完全信息。但这些条件本身不排除最优精度恰为零。

### 证明 A 导致低披露努力下界

无信号、初始均值为零时，对称均衡路径保持在零。令 m(t)=m_i(0,t)=m_j(0,t)，由式 (1)，

\[
\frac{dm(t)}{dt}=-m(t)(m_i)_y(0,t),
\quad m(T)=\frac{\alpha}{\sqrt{2\pi(S_0+\sigma^2T)}}.
\]

故

\[
m(t)=m(T)\exp\left(\int_t^T(m_i)_y(0,s)ds\right)\geq m(T).
\]

此时 d=0，最佳产出的差距漂移项为零，所以 B(0)>=B_ref。

### 证明 B 导致 G<=2J

固定 t，令 b(y)=Phi(y/sqrt(Q_t))-1/2。对 y>0，

\[
\psi(y):=\frac{b(y)}{yh(y,t)}
=\int_0^1\exp\left[\frac{y^2(1-v^2)}{2Q_t}\right]dv
\]

随 y 单调增加。瞬时 G/J 是 psi 在权重 y^2 h(y,t)f_t(y) 下的平均值。

根据条件 B，此权重等于基准权重 y^2 exp(-y^2/Q_t) 乘以一个非增函数。因此它对增加函数 psi 的平均值不高于基准权重的平均值。这可由 Cov(a(Y),psi(Y))<=0 直接得到，其中 a 非增、psi 非减。

基准平均值恰为 2：令 f0(y)=exp(-y^2/(2Q_t))，分部积分及 Gaussian 二阶矩给出

\[
\int_0^\infty y f_0(y)b(y)dy
=Q_t\int_0^\infty f_0(y)h(y,t)dy,
\]
\[
\int_0^\infty y^2 f_0(y)h(y,t)dy
=\frac{Q_t}{2}\int_0^\infty f_0(y)h(y,t)dy.
\]

两式相除为 2。对称性和时间积分遂给出 G<=2J。

### 完成产出比较与有限最优存在性

由式 (5)、条件 C 及 G<=2J，

\[
\mathcal B_\infty\leq B_{\rm ref}-(w-2)J-Z<B_{\rm ref}\leq\mathcal B(0).
\]

严格性来自 w>2 且 J>0。连续性保证一部分正的有限精度仍严格优于 B_infty。极限定义保证所有充分大的精度都不及 B(0)，在剩余紧区间上取最大值即得有限精度全局最大化解。

若完全信息下最优的 theta,T 满足该命题，则固定该 theta,T 降低精度就改善主办方目标。再加上设计集紧性（或成本的强制性）和对 theta,T 一致的高精度极限，可推出联合最优精度有限。固定 theta,T 的证明不能省略这些联合设计条件。

## 5. 目前能说什么、不能说什么

- 式 (1)–(5) 是直接 HJB 推导；命题在 A–C 等明确假设下是精确结论，不使用三阶截断。
- A–C 是额外的均衡形状条件，尚未从 S0、sigma、c、theta、T 推出。特别是，不能只报告 w>2 而隐去这些条件。
- B 给出了一个可独立检查的尾部集中条件；C 则是一个尚需研究的曲率符号。没有把它们当作已完成证明的模型性质。
- 高精度有限数值代理的形状修正与该路线相容，但数值并未验证完全信息的 B、C，更没有验证所有精度的尾部收敛。不得将它们视为已认证的非空参数区间。
- 若论文需要只依赖原始参数且已经闭合的结论，当前可用的是 PROOF_ATTEMPT.md 中的三阶近似命题；精确模型的条件命题适合作为进一步证明的路线，而非已经完成的参数定理。
