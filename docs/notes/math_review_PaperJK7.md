# PaperJK7.tex 数学证明审查报告（v2：含勘误与修订复核）

- 审查对象：`paper/PaperJK7.tex`，**当前 08:38 修订版**（2725 行；上一版为 01:22 的 2677 行）。
- v1 报告（01:40）包含两条主要问题。本文第 1 节**撤销其中一条（误报）**，第 2 节确认另一条已在修订中正确修正，第 3 节核对修订版的其余改动，第 4 节列出仍未处理的建议。

---

## 0. 结论摘要

| v1 结论 | 复审结果 |
|---|---|
| 问题①：引理 3 的核径向导数公式错误 | **误报，撤销**。论文原式 $ (1+x^3+xz^2-3x^2z^2)/((1-x^2)(1+x-2z^2))$ **正确**；是我复核脚本把核函数写成 $\sqrt{q-B}\sqrt{q+B}$，漏看了原文的 $(q+B)$（无根号） |
| 问题②：引理 4 低扩散段的 α 幂次与 $E$ 项 | **属实，已修正**。08:38 修订把 barrier 改为带 $\alpha$、积分项放大 2 倍，残差计算正确，下游常数同步更新，复核通过 |
| 问题③（低优先级）：逗号笔误 | 已改（`S_t\,d\widehat B_t`） |
| 问题③：范数混用说明 | 已加（新 2300–2306 行），正确 |
| 第四部分第 2 条：$b_0=\beta\notin\mathcal H$ 的空间问题 | 已加说明（新 2033–2039 行），正确 |
| 其余低优先级建议 | 未处理（第 4 节） |

---

## 1. 勘误：v1 问题①（核径向导数）——论文原式正确，特此撤销

### 1.1 我出错的原因
v1 的复核脚本（以及据此得到的"正确公式"）把核函数写成了

$$K^{\text{script}}=\frac{A}{\pi\sqrt{2\pi}\sqrt{q-B}\,\sqrt{q+B}\sqrt{q^2+qB-2A^2}},$$

而论文（第 1953–1955 行）是

$$K_q(A,B)=\frac{A}{\pi\sqrt{2\pi}\sqrt{q-B}\,(q+B)\sqrt{q^2+qB-2A^2}},$$

即 $(q+B)$ **不带根号**。差一个根号导致 $\partial_B\log K$ 中 $-\tfrac{1}{q+B}$ 变成了 $-\tfrac{1}{2(q+B)}$，于是我"验证"出了一个错误的闭式并把论文的正确公式判为错误。这是实质性错误，向作者致歉。

### 1.2 用论文原核函数的三重复核

**(i) 沿射线数值微分**（最直接的检验，不依赖任何手推）：取 $\varphi(\rho)=K_q(\rho A_0,\rho B_0)$，则 $\varphi'(1)/\varphi(1)=(A_0K_A+B_0K_B)/K$。中心差分（$h=10^{-6}$）在 $q\in\{0.3,0.7,1,2.5,7\}$、$x\in\{0.05,0.2,0.5,0.8,0.95,0.999\}$、$z\in\{0,0.3x,0.7x,x\}$ 网格上，与论文公式的最大偏差 $9.97\times10^{-4}$（纯差分误差）。示例：

| $q$ | $x$ | $z$ | 射线导数比 | 论文公式 |
|---|---|---|---|---|
| 1.0 | 0.5 | 0.3 | 1.11363636 | 1.11363636 |
| 1.0 | 0.2 | 0.2 | 0.94047619 | 0.94047619 |
| 0.4 | 0.6 | 0.5 | 1.55681818 | 1.55681818 |
| 2.0 | 0.9 | 0.4 | 4.94403731 | 4.94403731 |

**(ii) 解析偏导**（用原核函数）：
$$\partial_A\log K_q=\frac1A+\frac{2A}{q^2+qB-2A^2},\qquad
\partial_B\log K_q=\frac{1}{2(q-B)}-\frac{1}{q+B}-\frac{q}{2(q^2+qB-2A^2)},$$
代入 $x=B/q,\ z=A/q$ 得
$$\frac{AK_A+BK_B}{K_q}
=1+\frac{4z^2-x}{2(1+x-2z^2)}+\frac{x(3x-1)}{2(1-x^2)},$$
通分后分子恰为 $1+x^3+xz^2-3x^2z^2$（我逐项展开核对），即与论文一致。

**(iii) 正性与后续论证**：论文紧接的"分子对 $z^2$ 仿射，最小值取端 $1+x^3$ 或 $(1-x)(3x^3+x^2+x+1)$"对**正确**分子成立（$z^2=x^2$ 时分子 $=1+2x^3-3x^4=(1-x)(3x^3+x^2+x+1)$，$x<1$ 时为正）。全域网格上该比值最小约 $0.866>0$。故 $K_q$ 沿径向严格递增，配合新的 $\partial_A\log K_q>0$（第 1961–1964 行，正确）与两步链条（第 1975–1981 行，逻辑正确），得到
$K_q(A_t,A_r)<K_q(\rho_r(s+t),\rho_r(s+r))<K_q(s+t,s+r)$，
进而 $b_2(s,\ell)<b_2(s,\infty)$ 的证明成立。

**结论**：v1 问题①作废；论文该处（含修订版新增的 $\partial_A\log K_q$ 与两步链条说明）正确。相应地，v1 关于推导文档 `POSITIVE_PRIZE_FULL_DISCLOSURE_WINDOW.md` 同式"有误"的说法也一并作废。

---

## 2. v1 问题②（引理 4 低扩散超解）：已在修订中修正，复核通过

### 2.1 修订后的文本（第 2308–2349 行）
- 明确 Feynman–Kac 方程：$J_t+\mathscr L_tJ+\tfrac\alpha2\{H+[2\Phi(y/\sqrt{Q_t})-1]A\}=0$，$\mathscr L_t=\tfrac{a_t}2\partial_{yy}+\alpha A\partial_y$（源项现带 $\alpha$，正确）；
- barrier 改为 "unnormalized"：$\overline J=\alpha(T-t)G_{Q_t}(y)+\int_t^T2\alpha d(s)\{1+.486\tfrac{\alpha(T-s)}{Q_s}\}ds$（首项带 $\alpha$，积分项放大 2 倍）；
- 残差：$.486\alpha^2d\tfrac{T-t}{Q_t}+2\alpha d-2\alpha d(1+.486\tfrac{\alpha(T-t)}{Q_t})=-.486\alpha^2d\tfrac{T-t}{Q_t}\le0$；
- 结论：$\mathcal B-b(T)\le\alpha TG_{S_0+\sigma^2T}(0)+\tfrac{\alpha^2T^2}{4S_T^{3/2}}(1+.486\tfrac{\alpha T}{S_T})$，并把竞争集上的系数界从 $<.01$ 调为 $<.02$。

### 2.2 逐项复核
1. $(\partial_t+\mathscr L_t)[\alpha(T-t)G_{Q_t}]=-\alpha G_{Q_t}+\alpha^2A(T-t)G'_{Q_t}$ ✓（用 $\partial_tG_{Q_t}+\tfrac{a_t}2\partial_{yy}G_{Q_t}=0$）。
2. 源项 $=\alpha G_{Q_t}+\alpha E/2+\alpha(2\Phi-1)A/2$ ✓（$H=2G_{Q_t}+E$）。
3. 用 $|A|\le2d$、$|G'_Q|<.243/Q$：$\alpha^2A(T-t)G'_{Q_t}\le.486\alpha^2d(T-t)/Q_t$ ✓；用 $|E|\le2d$、$|2\Phi-1|\le1$：$\alpha E/2+\alpha(2\Phi-1)A/2\le2\alpha d$ ✓。
4. 与 $K'(t)=-2\alpha d(1+.486\alpha(T-t)/Q_t)$ 相加得 $-.486\alpha^2d(T-t)/Q_t\le0$ ✓（我独立重算与文中一致）。
5. $J(0,0)=\alpha TG_{Q_0}(0)+\int_0^T2\alpha d(s)(\cdots)ds=\alpha TG_{Q_0}(0)+\tfrac{\alpha^2T^2}{4S_T^{3/2}}(1+O(\alpha T/S_T))$ ✓ 与 (eq-design-low-diffusion-output) 一致；文中括号内的 $1+.486\alpha T/S_T$ 是保守上界（精确系数约 $0.324$）✓。
6. 下游一致性：竞争集上 $e\le2w/99<.01$、$\sqrt{2\pi Q_0/S_T}\le\sqrt{2\pi\cdot101/99}$，代入新显示的系数得 $\le0.0128<0.02$ ✓；而 (eq-design-competitive-output-screen) 需要的是该量除以 $2.9$ 后 $\le1$，余量充足 ✓。命题 1(iii) 中"线性项与二次项都被 $C\alpha$ / $C\alpha^2$ 控制"的用法不受系数变化影响 ✓。
7. 新增的点态范数说明（第 2300–2306 行）"Fourier-moment 范数支配点态范数" ✓ 成立（解析范数经 Fourier 反演控制 $\sup|\cdot|$），故 $|p-G_{Q_t}|,|n-G_{Q_t}|\le d$、$|E|,|A|\le2d$ 可用 ✓。

**结论**：v1 问题②已正确解决，无需进一步改动。

---

## 3. 修订版其余改动的核对

1. **逗号笔误**（原 1831 行）：现为 `d\tilde y_t=(m_i-m_j)dt+\sqrt\lambda S_t\,d\widehat B_t` ✓ 已改。
2. **加权空间归属**（新 2033–2039 行）：明确"加权空间陈述针对扰动 $E(0)$，而线性增长的背景 $\beta,b_w$ 本身不属于 $\mathcal H$；减去 $w=0$ 剖面后 $1-R_w$、$2\Phi-1-D_w$、$\beta-b_w$ 及其所需导数具有高斯衰减，故 $E(0)$ 属于下文的分式定义域空间"。这与 v1 第四部分第 2 条的疑问一致，且论证方向正确（$1-R_w=w\Phi(1-\Phi)+O(w^2)$ 等确实具有高斯尾衰减）✓。
3. **无其他内容改动**：以行号偏移核对——1905 行以前完全不变（笔误为等长替换）；引理 3 证明内净增 18 行（$\partial_A\log K_q$ 5 行 + 链条改写 + 上述空间说明 7 行）；引理 4 证明内净增 30 行（超解段）；2490 行之后整体平移 +48 行，内容一致。故除上述各处外，论文其余数学内容未变，v1 中被判"核对无误"的部分仍然有效。

---

## 4. 仍未处理的低优先级建议（可不改）

1. 完整披露能量恒等式的推导中，$dS$ 的漂移是 $\alpha(U^i_y+U^j_y)=\alpha S_y$（因 $m_j=-\alpha U^j_y$）；正文第 1831 行只写了 $\tilde y$ 的漂移 $m_i-m_j$，建议在推导该恒等式处点一句，避免读者误用 $U^i_y-U^j_y$。
2. 引理 3 无界精度段：$\eta=\ell^{-1/2}<1/4$ 与有界段 $\ell\le L$ 的衔接（$L=\eta_*^{-2}$）；以及"$11/20$ 留出均匀余量"处给出 $w$ 的显式阈值。
3. 命题 1(i)：明确 $\delta$ 与 $\sigma_F$ 的选取顺序与依赖（$\delta=\delta(\bar\sigma,\bar\lambda,\bar T)$，再取 $\sigma_F\le\min\{\bar\sigma,1\}$）。
4. 定理 1 证明中 $\|f_y\|_{\rho'}\le\|f\|_\rho/(\rho-\rho')$ 可用，但常数非最优（最优为 $1/(e(\rho-\rho'))$），加一句说明更稳妥。
5. 第五部分（外部定理适用性、谱扰动常数 $C$、引理 4 端点连续性与 Berge 论证、数值常数与验证脚本对照表）仍然适用，见下节。

---

## 5. 已逐条复核且未发现问题的部分

* Kalman–Bucy/Riccati：$d\tilde y$、$dS/dt=\sigma^2-\lambda S_t^2$、稳态 $\bar S=\sigma/\sqrt\lambda$、附录 (app-S-equ) 的 $\tanh/\coth$ 闭式解与分支单调性。
* 模型 HJB 与一阶条件（含 $+V^i_yV^j_y/c_j$ 的交叉项符号）。
* 定理 1：微分后系统、终值 $p(y,0)=n(y,0)=G_{S_T}$、守恒形式积分的 $\int p=\int n=1$、零激励解 $U_i^{(0)}=\Phi(y/\sqrt{Q_t})$、领先阶 $m_k=(\theta/c_k)G_{Q_t}+O(w)$ 及其与 Ryvkin 完全披露极限的一致性、Itô 验证恒等式 (eq-equilibrium-verification)。
* Ryvkin 基准：$f(1)=1$、$f'=\tanh(wd/2)$、$K'=\cosh^2(wd/2)$、$K''/K'=wf'$ 与差方程一致、$f\le1$、能量恒等式 (eq-design-full-energy)（用漂移 $\alpha S_y$ 重推成立）。
* (eq-best-output-drift-identity)：$\beta_y$、$\beta_q=\beta_{yy}/2$、$Q_t'=-a_t$ 与 $\langle\tilde y\rangle'=a_t$ 的抵消、$\beta(0,Q_0)=b(T)$。
* (eq-design-gaussian-bilinear)：$y=0$ 处数值等于 $G_q(0)/2$；解析上 $\int_0^q(|G'_{q-v}|*G_v^2)(0)dv=G_q(0)/2$。
* 引理 3：$\rho_t$ 的严格单调与 $<1$；**核比较恒等式（含径向导数与正性论证）现已确认正确**；$b_2(s,\ell)<b_2(s,\infty)$；$r\to1$ 处可积性。
* 引理 4：构造链条（$b(T_0)=b$、$\sigma^2T_0=kS_0/(1-k)$、$c_e\sigma^2=Mk$、$\mathcal B>.99b$、$w+x^2/4<.252\Rightarrow x<2$）、$r<1$ 的代数化简（$29/160=2.9/16$）、$F_\gamma$ 关于 $\gamma$ 单调、有限子覆盖、饱和常数（$K=w/.72$、映射半径比、$\cosh^2(.35/2)\{\cdot\}<2.9$）、**低扩散超解（修订后）**。
* 命题 2：$E_\ell$ 正概率；逐 $k,\ell$ 的零计数条件概率辨识强度；三点方差多项式的系数辨识；均值方程定出 $r,\mu_0$；$c_k$ 回收公式；$c_i=c_j$ 时 $(r,c_i,c_j)$ 共同缩放的不可识别性。
* 命题 1 的整体逻辑：(i) 候选设计与完全披露上界、$\sigma_F$ 的选取；(ii) 紧性+Berge+入口右导数 $c_e/c-1>0$；(iii) $\Gamma$ 预算与两段精度界、$c>u'(b_0)C_\sigma$ 排除正奖金。
* 附录 (appexidx-normal-benchmark) 的 $\hat\sigma^2$ MLE；精度反事实的导数符号条件与端点定义；第 2623 行单位换算 $\lambda\to a^2\lambda_a$、$\sigma\to\sigma_a/a$。

---

## 6. 仍需作者确认（与 v1 相同，仅保留仍适用者）

1. 外部定理的逐条适用性：Nishida (1978) Ch. III Remark 2.3；Jackson (2024) Thm 2.7 的 a-priori 有界性条件与 $F_k$ 分解对应；Kato (1980) Ch. VII §1–2；Lorenzi et al. (2005) Prop. 3.2.2、Thm 7.1.3。
2. 引理 3 无界精度段：$\|(\mathcal A_w-\mathcal A_0)f\|_{\mathcal H}\le C|w|\|(-\mathcal A_0)^{1/2}f\|_{\mathcal H}$ 中 $C$ 的来源与 $X=D((-\mathcal A_0)^{7/8})$ 上 Duhamel 压缩性的常数。
3. 低扩散段：条件 $\lambda S_0^2<.72\sigma^2$ 与 $\alpha T/S_T\le.01$ 在竞争集上是否一致满足（修订后不等式已成立，此处只涉及条件本身的核对）。
4. 引理 4 的端点连续性/Berge 论证（$T\to0$、$\ell\to0$ 的一致性）。
5. 数值常数（.72, 2.9, .486, 7/4, .252, $10^{-5}$, $10^6$, 99/2, 198 等）建议附"常数—验证脚本"对照表（如 `verify_prior_dominated_comparison.py`、`verify_low_precision_direct_budget.py`、`pipeline/theory/joint_design_exact.py`）。
6. 命题 1(i)/(ii) 的 $\sigma$ 区间衔接：$\sigma_F\le\min\{\bar\sigma,1\}$ 与 (ii) 的 $\sigma>1$ 是否要求 $\bar\sigma>1$；确认 $\bar\sigma$ 的选取与 $c$ 无关。
