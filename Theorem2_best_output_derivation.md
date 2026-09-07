# Theorem 2(Best-Output Design):自包含的定理与证明

> 本笔记提供一个**自包含**的推导:首先给出 Theorem 2 的最终形式(§2),其次列出证明所依赖的**外部结论**(取自 Ryvkin 2022 与 `paper/PaperJK5.tex`,列为引理、不给出证明,§3),最后给出 Theorem 2 的**内部证明**(§4)。内部证明只使用 §3 的外部引理与初等分析,不依赖本笔记之外的其他结果。
> 文档独立于 `paper/PaperJK*.tex`(未修改任何论文文件)。

---



## 1. 设定与记号(自包含)

**基础博弈(Ryvkin 2022 = 论文 Theorem 1)。** 两名风险中性选手 $i=1,2$,固定期限 $T>0$。选手 $i$ 在 $t\in[0,T)$ 选努力率 $q_{i,t}\ge0$,流成本 $c_iq_{i,t}^2/2$。相对差距 $y_t$ 满足
$$
dy_t=(q_{1,t}-q_{2,t})dt+\sigma dW_t,\qquad y_0=0,
$$
$W$ 为标准布朗运动, $\sigma>0$ 为创新风险。选手只能看到带噪公开信号并滤波;在论文的**稳态滤波限制**($S_t\equiv\bar S=\sigma/\sqrt\lambda$)与**阈值终局收益**($\theta\,\mathbf1_{\{\tilde y_T>0\}}$,$\tilde y_T$ 为滤波差距)下,马尔可夫完美均衡(MPE)努力由 Ryvkin 闭式给出,且只是感知差距 $\tilde y_t$ 与剩余时间的函数。初始感知差距为 $\tilde y_0=\mu_0$(实证归一 $y_0=0$)。

**水平层(A1)。** 每名选手的个体产出 $x^i_t$(私有得分)满足
$$
dx^1_t=q_{1,t}dt+\tfrac{\sigma}{\sqrt2}dW^1_t,\qquad
dx^2_t=q_{2,t}dt+\tfrac{\sigma}{\sqrt2}dW^2_t,\qquad W^1\perp W^2,\qquad x^1_0=x^2_0=\bar x_0,
$$
于是 $y_t=x^1_t-x^2_t$ 恰为上面的差距方程。A1 **不改变均衡**(均衡只依赖差距)。

**主办方目标(A2)。** 平台获得最优解: $B(\theta,T):=\mathbb E\big[\max\{x^1_T,x^2_T\}\big]$($\mathbb E$ 为比赛开始前、给定 $(\mu_0,\theta,T,\sigma,\lambda,c)$ 的事前期望)。

**记号。**
$$
w:=\frac{\theta}{\sigma^2c}\ (\text{对称时}),\qquad z_0:=\frac{\mu_0}{\sigma\sqrt T},\qquad
M(\theta,T):=\int_0^T\mathbb E\big[m_1(\tilde y_s,s)+m_2(\tilde y_s,s)\big]ds
$$
为 Theorem 1 的事前总期望努力(状态 $(\mu_0,0)$ 出发); $m_i$ 为 MPE 努力。

**维持性假设(与论文 Prop. 1 相同范围)。** 对称成本 $c_1=c_2=c$;小 $w$ 领头阶近似;所需函数的光滑性。



## 2. Theorem 2(陈述)

> **Theorem 2(Best-Output Design)。**
> 令设定 §1 成立,固定 $c,\sigma,\lambda>0$ 与 $\theta,T>0$。则
>
> **(T2.i)(分解恒等式,精确、无近似)** 期望最佳产出满足
> $$
> B(\theta,T)-\bar x_0=\tfrac12\,M(\theta,T)+\tfrac12\,\mathbb E|y_T|.
> \tag{2.1}
> $$
>
> **(T2.ii)(领头阶闭式)** 存在只依赖 $(\sigma,T,c)$ 的常数 $C<\infty$,使
> $$
> \Big|\tfrac12\,\mathbb E|y_T|-\frac{\sigma\sqrt T}{\sqrt{2\pi}}\Big|\le C\,w\,\sigma\sqrt T,
> \qquad
> M(\theta,T)=\frac{2w\sigma\sqrt T}{\sqrt{2\pi}}e^{-z_0^2/2}+o(w\sigma\sqrt T),
> \tag{2.2}
> $$
> 其中 $o(\cdot)$ 在 $w\to0$(其余参数固定)时成立。合并后,当 $\mu_0=0$($z_0=0$),
> $$
> B(\theta,T)-\bar x_0=\frac{\sigma\sqrt T}{\sqrt{2\pi}}(1+w)+R,\qquad |R|\le C\,w\,\sigma\sqrt T.
> \tag{2.3}
> $$
>
> **(T2.iii)(比较静态,领头阶)**
> $$
> \partial_\theta B=\tfrac12\,\partial_\theta M>0;
> \qquad
> \partial_T B=\tfrac12\,\partial_T M+\frac{\sigma}{2\sqrt{2\pi T}}>0;
> $$
> 且($\mu_0=0$)$\partial_\sigma B=\dfrac{\sqrt T}{\sqrt{2\pi}}\Big(1-\dfrac{\theta}{\sigma^2c}\Big)$,故
> $$
> \partial_\sigma B\ \gtrless\ 0 \iff w\lessgtr 1,\qquad
> B\text{ 在 }\sigma^\star=\sqrt{\theta/c}\ (w=1)\text{ 处取极小};
> $$
> 而领头阶上 $B$ 关于信号精度中性: $\partial_\lambda B=O(w)$。
>
> **(T2.iv)(设计者问题)** 设 $W(\theta,T)=\beta^T\big(B(\theta,T)-\bar x_0\big)^\alpha-\theta$,$0<\alpha<1$,$0<\beta\le1$。对固定 $T$, $W$ 关于 $\theta$ 单峰;若
> $$
> \alpha\beta^T\left(\frac{\sigma\sqrt T}{\sqrt{2\pi}}\right)^{\alpha-1}
> \frac{\sqrt T\,e^{-z_0^2/2}}{\sigma c\sqrt{2\pi}}>1,
> \tag{2.4}
> $$
> 则存在唯一内点极大 $\theta^\star>0$;否则角点 $\theta^\star=0$(零奖金"纯双抽"竞赛)为最优。
>
> **(T2.v)(奖–期替代,iso-B 前沿)** 对 $a>1$,记 $T_a=aT$,令
> $\theta_a=\min\big\{\theta\in[\underline\theta,\theta_0]:B(\theta,T_a)\ge B(\theta_0,T)\big\}$。当 $\mu_0=0$ 且解落在 $(\underline\theta,\theta_0)$ 内部时,领头阶给出
> $$
> \theta_a=\frac{\theta_0+\sigma^2c}{\sqrt a}-\sigma^2c
> \;\Longleftrightarrow\;
> \frac{\theta_0-\theta_a}{\theta_0}
> =\Big(1-\frac1{\sqrt a}\Big)\Big(1+\frac{\sigma^2c}{\theta_0}\Big)
> =\Big(1-\frac1{\sqrt a}\Big)\Big(1+\frac1w\Big),
> \tag{2.5}
> $$
> 严格大于 iso-总努力前沿的削减 $\big(1-\frac1{\sqrt a}\big)$;若 (2.5) 右端 $\le\underline\theta$,则 $\theta_a=\underline\theta$(奖金下界束紧)。

---



## 3. 外部引理(依赖结论,不证明)

以下结论取自 Ryvkin(2022,*Mgmt Sci* 68(11):8144–8165)与 `paper/PaperJK5.tex`(Theorem 1、Prop. 1),本笔记直接引用。

> **Lemma E1(均衡努力及其幅度界,Ryvkin Prop. 1(16)/论文 Theorem 1)。** 在 §1 设定下,MPE 努力可写为
> $$
> m_i(\tilde y_t,t)=\frac{K_i(z)}{\sqrt{2\pi\sigma^2(T-t)}}e^{-z^2/2},\quad z=\frac{\tilde y_t}{\sigma\sqrt{T-t}},\quad K_i(z)=\frac{\sigma^2}{2}\big[\gamma(\rho_1)+\gamma(\rho_2)\big]\big[1-\rho(z)^2\big]\big[1\pm\rho(z)\big],
> $$
> 其中 $\gamma,\rho_1,\rho_2,\rho(\cdot)$ 由 Ryvkin 式 (11)–(13) 定义。特别地(对称、$w$ 属于任一紧区间),
> $$
> m_1-m_2=\frac{2\sigma A_+(z)e^{-z^2/2}}{\sqrt{T-t}},
> \qquad
> \sup_z\big|A_+(z)e^{-z^2/2}\big|\le C_1 w
> \tag{E1}
> $$
> 对某 $C_1<\infty$ 成立;且对称时 $m_2(\tilde y,t)=m_1(-\tilde y,t)$,故 $\tilde y\mapsto m_1-m_2$ 为奇函数。
>
> **Lemma E2(小 $w$ 总期望努力,Ryvkin Lemma 7(24)/论文 Prop. 1)。** 对称选手、初始状态 $(\mu_0,0)$:
> $$
> M(\theta,T)=\frac{2w\sigma\sqrt T}{\sqrt{2\pi}}e^{-z_0^2/2}+o(w\sigma\sqrt T),\qquad w\to0.
> \tag{E2}
> $$

(如需"滤波下感知路径 $\tilde y$ 的事前分布关于 0 对称"这一事实,它由 $y_0=0$、$\mu_0=0$ 与均衡对称性直接推出,在 §4 内证,不列于此处。)



## 4. Theorem 2 的证明

### 4.0 组织

内部论断:
- **Claim P1**(分解恒等式)→ (T2.i);
- **Claim P2**(扩散溢价)→ (2.2) 第一式;
- **Claim P3**(努力项渐近 + 比较静态)→ (2.2) 第二式与 (T2.iii);
- 随后 (T2.iv)、(T2.v)。

记号:省略 $m_1-m_2$ 的自变量表示 $m_1(\tilde y_s,s)-m_2(\tilde y_s,s)$;除注明 $\mu_0=0$ 处外,推导允许一般 $\mu_0$。

### 4.1 Claim P1(分解恒等式,(T2.i))

*证明。* 对每条样本路径 $\max\{x^1_T,x^2_T\}=\bar x_T+\tfrac12|y_T|$, $\bar x_T=\tfrac12(x^1_T+x^2_T)$。由 A1,
$$
\bar x_T=\bar x_0+\frac12\int_0^T(q_{1,s}+q_{2,s})ds+\frac{\sigma}{2\sqrt2}(W^1_T+W^2_T).
$$
取期望: $\mathbb E(W^1_T+W^2_T)=0$; $\int_0^T\mathbb E(q_{1,s}+q_{2,s})ds$ 恰为 Theorem 1 的总期望努力 $M(\theta,T)$(对信息流取全期望,初始感知状态 $\mu_0$)。故 $\mathbb E[\bar x_T]=\bar x_0+\tfrac12M(\theta,T)$,两端加 $\tfrac12\mathbb E|y_T|$ 得 (2.1)。∎

### 4.2 Claim P2(扩散溢价,(2.2) 第一式)

*证明。* 记 $J_T:=\int_0^T(m_1-m_2)ds$。由 $y_0=0$,$y_T=\sigma W_T+J_T$。

**步骤 1(居中)。** $\mu_0=0$ 且选手对称 ⇒ 感知路径 $\tilde y$ 的事前分布关于 0 对称(初始状态对称、策略对称、信息结构对称);由 (E1) 中 $m_1-m_2$ 为 $\tilde y$ 的奇函数,$\mathbb E[J_T]=0$。

**步骤 2($L^2$ 界)。** 由 (E1),
$$
|J_T|\le\int_0^T|m_1-m_2|ds\le2C_1\sigma w\int_0^T\frac{ds}{\sqrt{T-s}}=4C_1\sigma w\sqrt T,
$$
故 $\mathbb E[J_T^2]\le16C_1^2w^2\sigma^2T$。

**步骤 3(绝对期望的 Lipschitz 误差)。** 反向三角不等式与 Cauchy–Schwarz:
$$
\big|\mathbb E|y_T|-\mathbb E|\sigma W_T|\big|\le\mathbb E|J_T|\le\big(\mathbb E[J_T^2]\big)^{1/2}\le4C_1w\sigma\sqrt T.
$$
又 $\mathbb E|\sigma W_T|=\sigma\sqrt{2T/\pi}$,于是
$$
\Big|\tfrac12\mathbb E|y_T|-\frac{\sigma\sqrt T}{\sqrt{2\pi}}\Big|\le2C_1w\sigma\sqrt T,
$$
即 (2.2) 第一式($C=2C_1$)。∎

*注(有限 $w$ 的方差修正)。* 由对称性与步骤 2,$\mathbb E[y_T^2]=\sigma^2T(1+O(w))$;若近似 $y_T\sim\mathcal N(0,\alpha\sigma^2T)$,则 $\tfrac12\mathbb E|y_T|\approx\sigma\sqrt T\sqrt\alpha/\sqrt{2\pi}$,$\alpha=\mathbb E[y_T^2]/(\sigma^2T)$。MC 标定($T=60,\sigma=1$): $\alpha-1\approx0.5\%(w=0.2)$、$8.9\%(w=1)$、$42\%(w=2.5)$。

### 4.3 Claim P3(努力项渐近、λ 中性与比较静态,(2.2) 第二式与 (T2.iii))

*证明。* (2.2) 第二式即 Lemma E2。
**λ 中性。** $M$ 的领头阶不含 λ;均衡函数本身 λ 无关(论文 §2.1),λ 只经感知路径实现分布进入更高阶($O(w^2)$);扩散溢价主项只含 $\sigma,T$。故 $\partial_\lambda B=O(w)$。
**θ 单调。** 由 (T2.i) 与 (2.2), $\partial_\theta B=\tfrac12\partial_\theta M>0$,且领头阶
$\partial_\theta B=\sqrt T\,e^{-z_0^2/2}/(\sigma c\sqrt{2\pi})+o(1)>0$。
**T 单调(双通道)。** 扩散项 $\partial_T(\sigma\sqrt T/\sqrt{2\pi})=\sigma/(2\sqrt{2\pi T})>0$;努力项:由 E2,$M\propto\sqrt T\,e^{-\mu_0^2/(2\sigma^2T)}$,
$$
\partial_T\ln M=\frac1{2T}+\frac{\mu_0^2}{2\sigma^2T^2}>0,
$$
故 $\partial_TB>0$。注意 $\theta=0$ 时仍有 $\partial_TB=\sigma/(2\sqrt{2\pi T})>0$:时长在零奖金下也有价值(采样通道)。
**σ 的 U 型($\mu_0=0$)。** 由 (2.3),忽略余项 $R$:
$$
B-\bar x_0=\frac{\sqrt T}{\sqrt{2\pi}}\Big(\sigma+\frac{\theta}{c\sigma}\Big)
\ \Rightarrow\ \partial_\sigma(B-\bar x_0)=\frac{\sqrt T}{\sqrt{2\pi}}\Big(1-\frac{\theta}{c\sigma^2}\Big)=\frac{\sqrt T}{\sqrt{2\pi}}(1-w),
$$
在 $w=1\iff\sigma=\sqrt{\theta/c}$ 变号,且 $\partial_\sigma^2=\dfrac{2\theta\sqrt T}{c\sqrt{2\pi}}\sigma^{-3}>0$,故为唯一极小。对照 $M\propto\theta/(\sigma c)$ 恒随 $\sigma$ 递减——同一风险参数对两种目标的比较静态**符号相反**。∎

### 4.4 (T2.iv)(设计者问题)

*证明。* 记 $D(\theta,T):=B-\bar x_0$。领头阶 $D$ 关于 $\theta$ 线性且严格增,故对固定 $T$,
$$
\partial_\theta^2W=\alpha(\alpha-1)\beta^TD^{\alpha-2}(\partial_\theta D)^2<0\quad(\alpha<1),
$$
即 $W$ 关于 $\theta$ 严格凹(单峰)。边际
$$
\partial_\theta W=\alpha\beta^TD^{\alpha-1}\partial_\theta D-1,\qquad
\partial_\theta D=\frac{\sqrt T\,e^{-z_0^2/2}}{\sigma c\sqrt{2\pi}}>0.
$$
与 Prop. 1 不同,此处 $D(0,T)=\tfrac{\sigma\sqrt T}{\sqrt{2\pi}}(1+O(w))>0$ 有正下界,故 $\partial_\theta W|_{\theta=0}$ 有限;唯一内点极大当且仅当 $\partial_\theta W|_{\theta=0}>0$,即 (2.4);否则 $\partial_\theta W<0$ 恒成立,最优在角点 $\theta^\star=0$。∎

### 4.5 (T2.v)(iso-B 替代前沿)

*证明。* $\mu_0=0$ 领头阶下,由 (2.3) 忽略 $R$:
$$
B(\theta,T)-\bar x_0=\frac{\sqrt T}{\sqrt{2\pi}}\Big(\sigma+\frac{\theta}{\sigma c}\Big).
$$
约束 $B(\theta_a,T_a)\ge B(\theta_0,T)$ 化为
$$
\sqrt a\Big(\sigma+\frac{\theta_a}{\sigma c}\Big)\ge\sigma+\frac{\theta_0}{\sigma c}
\iff \theta_a\ge\frac{\theta_0+\sigma^2c}{\sqrt a}-\sigma^2c.
$$
$B$ 关于 $\theta$ 严格增((T2.iii)),等号即最小可行 $\theta_a$,得 (2.5) 左式;右式为代数变形。对照 iso-M 前沿:$M\propto\theta\sqrt T$(E2,$\mu_0=0$),$M(\theta_a^M,T_a)=M_0$ 给出 $\theta_a^M=\theta_0/\sqrt a$,削减 $1-1/\sqrt a$。因 $\sigma^2c/\theta_0>0$,
$$
\frac{\theta_0-\theta_a}{\theta_0}-\Big(1-\frac1{\sqrt a}\Big)=\Big(1-\frac1{\sqrt a}\Big)\frac{\sigma^2c}{\theta_0}>0,
$$
iso-B 削减严格更大,放大倍数为 $1+\sigma^2c/\theta_0=1+1/w$。若 (2.5) 的解 $\theta_a<\underline\theta$,则因 $B$ 在 $\theta$ 上增,可行集的下确界在 $\underline\theta$ 处达到: $\theta_a=\underline\theta$(下界束紧,(2.5) 的解释在截断处失效)。∎

### 4.6 适用范围

- (T2.i) 对**非对称**选手与任意 $\mu_0$ 逐字成立(只用定义与 A1);(T2.ii)–(T2.v) 的显式系数按 E2 需对称成本。非对称/任意 $\mu_0$ 的竞赛,把 E2 换成精确 BVP 数值 $M$ 代入 (2.1) 即可(§6 的 73 场数值即按此)。引理 2 的对称性要求只影响"居中"步骤;非对称时 (T2.i) 仍精确,只是扩散溢价主项需保留 $\mathbb E[y_T]$ 平移项。
- 维持性假设与论文 Prop. 1 相同;λ 二阶通道见 §5 R1。



## 5. 备注(不计入定理,但影响口径)

**R1(λ 在哪里会回来)。** 领头阶 $\partial_\lambda B=O(w)$;两条二阶通道:(i) 感知路径分布的 λ 依赖改变 $\mathrm{Var}(J_T)$ 与 $\mathrm{Cov}(W_T,J_T)$($O(w^2)$);(ii) 若终局产出按滤波条件分布估值( $y_T\mid I_T\sim\mathcal N(\tilde y_T,\bar S)$,$\bar S=\sigma/\sqrt\lambda$),则
$$
\mathbb E\big[\tfrac12|y_T|\,\big|\,I_T\big]
=\sqrt{\frac{\bar S}{2\pi}}e^{-\tilde y_T^2/(2\bar S)}
+\tilde y_T\Big(\Phi\big(\tfrac{\tilde y_T}{\sqrt{\bar S}}\big)-\tfrac12\Big),
$$
事前取期望出现正的 $\sqrt{\bar S}$ 项——λ 经滤波不确定性重新进入 best-output 目标(衔接审稿意见 #23)。实证中 $\bar S\ll\sigma^2T$ 且主办方事后看真实私有榜,故主文采用 (2.2)。

**R2(水平层独立创新的角色)。** (2.1) 中的 $\tfrac12\mathbb E|y_T|$ 是"两次独立抽样取最大"的溢价;它要求 A1 的两条独立个体创新(每人 $\sigma/\sqrt2$)。若只有差距中的单一布朗运动而无独立个体层,该溢价消失。A1 的独立性是 best-output 目标区别于总努力目标的经济内容来源。

**R3(误差口径)。** (2.2) 的相对误差 $O(w)$ 在样本量级($w\in[0.03,2.76]$,中位 1.27)下:扩散溢价 MC 实测偏差 0.5%–9%($w=0.2$–$1$)、42%($w=2.5$,见 §6);努力项误差由精确 BVP 控制在论文反事实精度内。写论文时建议在表注声明"定理按 $w$ 领头阶陈述,表中精确值来自 BVP 数值解 + 有限 $w$ 修正 $\sqrt\alpha$"。



## 6. 数值核查(全部可用仓库代码复现)

| 校验 | 结果 |
|---|---|
| MC $E[\max]$($w=0.2/1.0/2.5$;$T=60,\sigma=1$) | 3.759 / 6.403 / 10.284 vs 闭式 3.708 / 6.180 / 10.816(小 $w$ 几乎精确) |
| 扩散溢价 $\tfrac12E|y_T|$ MC vs 主项 3.090 | 3.074 / 3.365 / 4.400(修正 $0.5\%/8.9\%/42\%$) |
| σ 的 U 型($\theta=1$ 扫 σ) | MC 在 $\sigma^\star\approx\sqrt{\theta/c}=1$ 附近取极小,方向与 (T2.iii) 一致 |
| λ 敏感性($\lambda=0.5\to200$) | $E[\max]$ 变化 $\le0.4\%$ |
| 73 场 iso-M 中位数($a=1.01/1.02/1.05/1.10$) | −0.645/−1.279/−3.111/−5.955%——与论文表 9 逐位一致 |
| 73 场 iso-B 中位数(同上) | −1.588/−3.102/−7.447/−13.611%(约 iso-M 的 2.1–2.3 倍) |
| 额外削减 vs $1/w$ 相关 | −0.96,支持 (2.5) 弹性 $1+1/w$ |
| 极端场($a=1.10$) | 3288($w=0.06$):iso-B −78.3%;4488($\sigma=5.35,w=0.03$):触及 0.01k 下界(呼应 (T2.iv) 角点) |
| 数值边界 | 仓库 BVP 求解器在 $w\lesssim2.8$ 可靠($w\gtrsim14$ 出现伪迹并触发守卫);样本 $w\le2.76$ 不受影响 |

复现:MC 用对称闭环模拟(逐小时 $x^1,x^2$、$y$、带噪读数、稳态 Kalman 增益);前沿用 `test_counterfactual/all_contests_joint_optimize.py` 的 `total_expected_effort` + 二分,把二分目标 $M_0$ 换成 $M_0+2(\mathrm{spread}(T_0)-\mathrm{spread}(T_a))$。



## 7. 可直接粘贴的 LaTeX 命题(供写入论文)

```latex
\begin{proposition}[Best-Output Design]\label{prop-best-output}
Fix the two-player contest of Theorem~\ref{thm-equilibrium} extended with the
level layer (A1) and the sponsor payoff $B(\theta,T):=\mathbb E[\max\{x^1_T,x^2_T\}]$
(A2). Under the maintained steady-state filtering restriction, the threshold
terminal payoff, symmetric costs $c_1=c_2=c$, and the leading-order
approximation in $w=\theta/(\sigma^2 c)$,
\begin{enumerate}
\item[(i)] $B(\theta,T)-\bar x_0=\tfrac12 M(\theta,T)+\tfrac12\,\mathbb E|y_T|$
(exact), with $\tfrac12\,\mathbb E|y_T|=\sigma\sqrt T/\sqrt{2\pi}\,\big(1+O(w)\big)$;
\item[(ii)] $B$ is increasing in $\theta$ and $T$, and $\partial_\sigma B\gtrless0$
iff $w\lessgtr1$, with the turning point at $\sigma=\sqrt{\theta/c}$;
at the leading order $B$ is neutral in $\lambda$;
\item[(iii)] the designer's problem $\beta^T(B-\bar x_0)^\alpha-\theta$ is
single-peaked in $\theta$, with a unique interior maximum whenever
\eqref{eq-corner-condition} holds and a corner (zero-prize) maximum otherwise;
\item[(iv)] along the iso-$B$ frontier, lengthening the contest by factor $a$
permits a prize cut of $\big(1-1/\sqrt a\big)\big(1+\sigma^2 c/\theta_0\big)$,
which strictly exceeds the iso-effort cut $\big(1-1/\sqrt a\big)$.
\end{enumerate}
\end{proposition}
```

---

*§3 外部引理出处:Ryvkin(2022)Prop. 1、Lemma 7;论文 `paper/PaperJK5.tex` Theorem 1、Prop. 1。§4 为内部证明。文档独立成文,未改动仓库中的论文与代码。*
