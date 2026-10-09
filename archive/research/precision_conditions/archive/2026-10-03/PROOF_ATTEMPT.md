# 无精度成本的有限披露：三阶机制与证明边界

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

## 一、结论的范围

本文沿用原模型，固定 T、sigma、S0>0，对称成本 c_i=c_j=c，初始后验均值 mu=0。S0 独立于精度。记 alpha=theta/c。以下得到两项不同层次的结论。

1. **精确模型的小激励结论。** 可以选取两个有限正精度 lambda_l<lambda_h，使较低精度在所有充分小的正 alpha 下产生严格更高的期望总努力。这证明了维持努力的机制，但不证明最佳产出在小激励下更高。
2. **三阶近似的强激励设计结论。** 如果 S0>=5 sigma^2 T 且 2<=theta/(c sigma^2)<=3，则三阶产出近似关于 lambda 的每个全局最大化解都严格为正且有限，无须精度成本。

第二项尚不是精确产出的强激励定理。现有 Theorem 1 对每个固定有限精度给出的小激励展开，不能自动控制 w in [2,3] 或 lambda 趋于无穷时的余项。本文最后给出升级所需的定量条件，未假称已经验证。

## 二、证明结构

对均衡再展开一阶，得到总努力的三阶系数 C_lambda。利用 Gaussian 积分直接证明 C_0>0、C_infty<0。最佳产出的二阶系数 R 则满足 R_0=0、R_infty>0。因此降低精度损失二阶产出收益，但获得三阶总努力收益。最后用两个端点的严格比较排除无限精度，用零精度的正右导数排除完全不披露。

## 三、推导与完整证明

### 1. 精确模型的三阶展开

令

\[
q=S_0+\sigma^2T,\qquad Q_t=S_t+\sigma^2(T-t),\qquad A_t=q-Q_t,
\]
\[
h(y,t)=(2\pi Q_t)^{-1/2}e^{-y^2/(2Q_t)},\qquad
v(y,t)=\Phi(y/\sqrt{Q_t}).
\]

P_{t,s} 是方差增量 Q_t-Q_s 的 Gaussian 转移算子。对固定的有限正精度，沿 Theorem 1 构造的小激励均衡分支，把附录的固定点迭代再展开一阶：

\[
U_i=v+\alpha f+\alpha^2 k+O(\alpha^3),\qquad
U_j=1-v+\alpha f-\alpha^2 k+O(\alpha^3),
\]
\[
f=-\tfrac12\int_t^T P_{t,s}h(\cdot,s)^2ds,
\quad g=f_y,
\quad k=\int_t^T P_{t,s}[h(\cdot,s)g(\cdot,s)]ds.
\]

这里 U_k=V^k/theta。因为 HJB 的非线性为梯度的二次多项式，已知 U-U0=O(alpha) 及一阶校正后，将其代入非线性，alpha 的一阶项分别为 hg 和 -hg；固定点的有界卷积估计使上式余项在控制一阶空间导数的范数内仍为 O(alpha^3)。所有常数可依赖固定的 lambda,T,sigma,S0。

于是

\[
m_i=\alpha h+\alpha^2g+\alpha^3k_y+O(\alpha^4),\qquad
m_j=\alpha h-\alpha^2g+\alpha^3k_y+O(\alpha^4).
\tag{1}
\]

特别地，差距漂移为 2alpha^2 g+O(alpha^4)。令 X_t 是零漂移后验均值，则 X_t~N(0,A_t)。定义 H_lambda=E int_0^T(m_i+m_j)/2 dt。Duhamel 展开以及 P_{t,s}h(.,s)=h(.,t) 给出

\[
H_\lambda=\alpha T h(0,0)+\alpha^3 C_\lambda+O(\alpha^4),
\]
\[
C_\lambda=\int_0^T
\left\{tE[(hg)_y(X_t,t)]
+2(T-t)E[g(X_t,t)h_y(X_t,t)]\right\}dt.
\tag{2}
\]

第一项来自对 k_y 的两重时间积分换序；第二项来自努力导致的漂移对领先阶努力 h 的影响。对 A_t>0，Gaussian 分部积分及 h_y=-yh/Q_t 给出

\[
E[(hg)_y]=\frac{E[X_thg]}{A_t},\qquad
E[gh_y]=-\frac{E[X_thg]}{Q_t}.
\]

附录中 g 的核和 Gaussian 二阶矩积分进一步给出

\[
E[X_thg]=\frac{A_tQ_t}{(2\pi)^{3/2}}
\int_t^T\frac{ds}{\sqrt{Q_s}\,D_{t,s}^{3/2}},
\qquad
D_{t,s}=Q_t(2Q_t-Q_s)+A_t(4Q_t-Q_s).
\]

因此

\[
C_\lambda=\frac1{(2\pi)^{3/2}}\int_0^T\int_t^T
\frac{tQ_t-2(T-t)A_t}{\sqrt{Q_s}\,D_{t,s}^{3/2}}\,ds\,dt.
\tag{3}
\]

公式 (3) 没有除以 A_t，可连续延伸到零精度。由 (1)，最佳产出恒等式中的差距贡献没有 alpha^3 项。因此

\[
\mathcal B=\beta(0,q)+\alpha T h(0,0)
+\alpha^2R_\lambda+\alpha^3C_\lambda+O(\alpha^4).
\tag{4}
\]

这说明精确最佳产出的三阶系数恰好就是总努力的一半的三阶系数，而不是另一个未识别的作用。

### 2. 总努力机制的严格符号

lambda=0 时，Q_t=q、A_t=0，式 (3) 化为

\[
C_0=\frac{T^3}{6(2\pi)^{3/2}q^{5/2}}>0.
\tag{5}
\]

当 lambda 趋于无穷时，对每个 t>0，Q_t 趋于 sigma^2(T-t)，A_t 趋于 S0+sigma^2t。式 (3) 分子趋于

\[
t\sigma^2(T-t)-2(T-t)(S_0+\sigma^2t)
=-(T-t)(2S_0+\sigma^2t)<0.
\]

以下估计保证可以交换极限和积分。因为 Q_s<=Q_t、Q_t+A_t=q，有 D_{t,s}>=qQ_t；又 Q_t>=sigma^2(T-t)，且

\[
\int_t^T Q_s^{-1/2}ds\leq 2\sqrt{T-t}/\sigma.
\]

式 (3) 被积函数绝对值先对 s 积分后，除去常数 (2pi)^(-3/2)，被

\[
\frac{2t}{\sigma^2q^{3/2}}+
\frac{4A_t}{\sigma^4q^{3/2}}
\]

控制。对于二维支配收敛，也可保留积分前的界：t/(sigma^2 q^(3/2) sqrt(T-t)sqrt(T-s)) 加上 2/(sigma^4 sqrt(q)sqrt(T-t)sqrt(T-s))，在 0<t<s<T 上可积。故 C_lambda 收敛到严格负的 C_infty。

由系数连续性，存在两个固定的有限正精度 lambda_l<lambda_h，使 C_lambda_l>0>C_lambda_h。两者应用式 (2)，得到

\[
H_{\lambda_l}-H_{\lambda_h}
=\alpha^3(C_{\lambda_l}-C_{\lambda_h})+O(\alpha^4)>0
\]

对所有充分小的 alpha>0 成立。这是精确均衡中的总努力比较，不需要假设完全信息均衡的小激励展开一致成立。

### 3. 三阶近似的有限精度定理

定义 B^[3] 为式 (4) 中舍去余项后的函数。注意：以下把它视为一个独立的近似设计目标，并不声称在强激励区间自动逼近精确 B。

在 mu=0 时 R0=0、R_infty>0。令

\[
\alpha_-:=\frac{R_\infty}{C_0-C_\infty}.
\]

若 alpha>alpha_-，则

\[
\mathcal B^{[3]}(0)-\mathcal B^{[3]}(\infty)
=\alpha^2\{\alpha(C_0-C_\infty)-R_\infty\}>0.
\tag{6}
\]

连续性和端点极限意味着所有充分高精度的值低于零精度的值；在剩余的有限闭区间上，B^[3] 取得最大值。因此每个全局最大化解的精度有限。

为排除零精度，定义

\[
L=\frac{S_0^2T^3}{6}+\frac{S_0\sigma^2T^4}{12}+\frac{\sigma^4T^5}{60},
\qquad
K=\frac{3S_0^2T^4}{8}+\frac{13S_0\sigma^2T^5}{60}+\frac{\sigma^4T^6}{20}.
\]

Riccati 方程对 lambda 在零处求导，得到 partial_lambda Q_t=-J_t，其中 J_t=S0^2t+S0 sigma^2t^2+sigma^4t^3/3。对系数核求导可得

\[
R'_0=\frac{2L}{(2\pi)^{3/2}q^{5/2}}>0,
\qquad
C'_0=-\frac{K}{(2\pi)^{3/2}q^{7/2}}<0.
\tag{7}
\]

核的分母在 lambda=0 附近有正下界，故求导换序成立。具体地，C 的分子导数为 -(2T-t)J_t；D 的导数为 qJ_s，连同 sqrt(Q_s) 的导数，给出

\[
-C'_0=\frac1{(2\pi)^{3/2}q^{7/2}}
\int_0^T\int_t^T[(2T-t)J_t+tJ_s]dsdt.
\]

换序后时间权重为 2T^2-3Tt+3t^2/2，逐项积分即为 K。R'_0 则由 2 int_0^T(T-t)J_tdt=2L 得到。

令 alpha_+=2qL/K。若 alpha<alpha_+，则 (B^[3])'_+(0)=alpha^2(R'_0+alpha C'_0)>0。因此，当

\[
\boxed{\alpha_-<\theta/c<\alpha_+}
\tag{8}
\]

时，每个三阶近似的全局最优精度满足 0<lambda*<infty。

### 4. 一个完全由原始参数给出的强充分条件

式 (8) 可以进一步由简单条件保证：

\[
\boxed{S_0\geq5\sigma^2T,\qquad
2\leq\frac{\theta}{c\sigma^2}\leq3,\qquad\tilde y_0=0.}
\tag{9}
\]

证明需要验证 alpha_-<2sigma^2 和 alpha_+>3sigma^2。令 A=S0+sigma^2t，r=T-t。完全信息系数中，置 u=T-s，有 Q_s=sigma^2u，D_{t,s}=sigma^2[K_r-qu]，其中 K_r=2r(2q-sigma^2r)。内层积分为

\[
\int_t^T\frac{ds}{\sqrt{Q_s}D_{t,s}^{3/2}}
=\frac{2}{\sigma^4K_r\sqrt{3q-2\sigma^2r}}.
\]

mu=0 时 R 的核为 2A/[(2pi)^(3/2)sqrt(Q_s)(2q-Q_s)sqrt(D_{t,s})]。对其内层积分作 u=(K_r/q)sin^2(v) 的代换，使用 2q-sigma^2K_r/q=2A^2/q，得到反正切项。因此

\[
R_\infty=\frac{2}{(2\pi)^{3/2}\sigma^2\sqrt q}
\int_0^T\arctan\frac{A}{\sqrt{q(q+2A)}}dt,
\]
\[
-C_\infty=\frac1{(2\pi)^{3/2}\sigma^4}
\int_0^T\frac{S_0+A}{(q+A)\sqrt{q+2A}}dt.
\tag{10}
\]

设 a=S0/q，v=A/q，则 a<=v<=1。令

\[
f(v)=(1+v)\sqrt{1+2v}\arctan\frac{v}{\sqrt{1+2v}}-v.
\]

直接求导，f'(v)=(2+3v)arctan(v/sqrt(1+2v))/sqrt(1+2v)>0，故

\[
f(v)\leq f(1)=\pi/\sqrt3-1<5/6\leq a.
\]

这等价于式 (10) 中 R_infty 的被积函数严格小于 -2sigma^2 C_infty 的对应被积函数。因此 R_infty<-2sigma^2C_infty<2sigma^2(C0-C_infty)，即 alpha_-<2sigma^2。

再令 s=S0/(sigma^2T)>=5。代入 L、K，alpha_+>3sigma^2 等价于

\[
40s^3-75s^2-54s-14>0.
\]

该多项式在 s=5 时为 2841，且其导数 120s^2-150s-54 在 s>=5 时严格为正。因此 alpha_+>3sigma^2。由式 (8)，条件 (9) 得证。

条件的经济含义：初始状态仍有大量未揭示信息；奖金足以使动态努力反馈产生三阶影响；但激励没有强到使三阶目标在零披露处也开始下降。上下界均为保守充分条件，不是必要阈值。

### 5. 升级到精确模型的定量桥梁

固定满足式 (8) 的参数，记 Delta3=B^[3](0)-B^[3](infty)>0。假设精确均衡目标在 lambda>=0 上连续并存在极限 B_infty，且能够另外证明

\[
\mathcal B(0)\geq\mathcal B^{[3]}(0)-\varepsilon_0,
\qquad
\mathcal B_\infty\leq\mathcal B^{[3]}(\infty)+\varepsilon_\infty,
\quad \varepsilon_0+\varepsilon_\infty<\Delta_3.
\tag{11}
\]

则精确模型也存在有限的全局最优精度。这只要求两个端点的一侧误差界，不要求全部精度上的均匀绝对误差。若还能控制零精度处导数误差严格小于 alpha^2(R'_0+alpha C'_0)，则最优精度严格为正。

证明是将 (11) 代入 (6)，得到 B(0)>B_infty，再应用尾部截断与紧集最大值定理；正右导数排除零端点。若零精度均衡难以建立，也可使用一个有限正精度的严格候选下界替代，但须重新验证其收益差。

对于联合设计，需要在完全信息最优的 theta、T 上验证上述精度改善，并控制相关 theta、T 集合上的高精度尾部。固定 theta、T 的定理本身不能直接称为联合设计定理。

## 四、尚未完成的工作

- 式 (9) 是**三阶近似**的严格参数条件，不能以接受强假设为由，把未证实的 (11) 当作原模型已经满足的事实。
- 更新：Theorem 1 已补充任意有限激励、固定有限正精度下的有界光滑内点均衡存在性。零精度端点及完全信息极限仍需单独证明；新的存在性结论不提供这些边界上的一致估计。
- small-alpha 的 O(alpha^4) 余项不提供 w in [2,3] 的数值界，也不保证对 lambda 无穷极限一致。
- 上述总努力机制已有精确模型的局部证明；原模型强激励下有限最优精度的无条件证明仍未完成。

数值核对保存在 third_order.json：零端点、无穷端点、零端点导数分别用独立积分计算复核，并与小激励非线性均衡的总努力作比较。这些数值核对不替代上述解析证明，也不认证余项界。
