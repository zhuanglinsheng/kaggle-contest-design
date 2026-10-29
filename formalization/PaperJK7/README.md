# PaperJK7 Lean 验证

本目录形式化验证论文的 25 个代数、微分及不等式命题。它不是整篇论文的形式化证明。源文件的每个分析或概率输入都作为明确假设出现；没有将论文中的均衡存在、滤波或谱稳定性结论设为公理。

## 已验证范围

| 论文中的步骤 | Lean 命题 |
| --- | --- |
| HJB 平方配方及非负努力约束 | `complete_square`, `effort_maximizer` |
| HJB driver 分解及负坐标有界性条件 | `driver_decomposition`, `negative_coordinate_bound` |
| Riccati 方程推出 Q 的导数 | `remaining_variance_derivative` |
| 三个不同时间点识别方差系数 | `quadratic_three_roots`, `variance_identification` |
| 两个不同累计强度差识别均值与 r | `mean_identification` |
| 成本不对称、成本恢复、共同缩放不变性 | `asymmetric_intensities`, `cost_identification`, `common_scale_invariance` |
| 比较核的分子在整个指定区间严格为正 | `kernel_endpoint_factorization`, `kernel_numerator_positive` |
| 低扩散超解残差符号 | `barrier_residual` |
| 全披露 payoff screen 的多项式恒等式与不等式 | `payoff_screen_identity`, `payoff_screen_bound` |
| 二次延迟成本的平方配方上界 | `quadratic_delay_bound` |
| 一阶质量系数的分解、正性与对称项消失 | `first_order_bracket`, `first_order_coefficient_positive`, `symmetric_precision_cancellation` |
| 低扩散段的加权二次通量控制 | `weighted_flux_bound` |
| 高扩散段在 0 ≤ w ≤ 0.252 上的四个压缩映射常数界 | `high_diffusion_radius_one`, `high_diffusion_radius_two`, `high_diffusion_lipschitz_one`, `high_diffusion_lipschitz_two` |

例如，`variance_identification` 证明三点上的方差表达式相等必然推出三个系数相等。它没有形式化证明“联合分布相等推出这些条件方差相等”的概率论步骤。`remaining_variance_derivative` 假设 Riccati 微分方程成立，没有证明 Kalman–Bucy 滤波定理。

## 复现

固定 Lean 4.19.0 和 mathlib v4.19.0。安装相应 Lean 工具链后，在本目录运行：

```sh
lake update
lake exe cache get
lake build
```

本次检查实际使用临时目录中下载的 Lean 4.19.0 和 mathlib v4.19.0，通过 mathlib 的 `lake env lean` 编译本目录的同一源文件。临时环境不作为项目依赖提交。机器上的缓存下载器遇到本机链接兼容问题时，已通过 `lake env lean --run Cache/Main.lean get` 完成下载，证明文件由 Lean 正常编译。

`verification.log` 保存工具链版本、源文件 SHA256、命令、退出状态以及全部 25 个命题的公理依赖。结果仅依赖 Lean/mathlib 的标准基础公理 `propext`, `Classical.choice`, `Quot.sound`，没有 `sorryAx`。源文件末尾保留全部 `#print axioms`，复现时会重新输出。

## 尚未形式化的内容

Kalman–Bucy 滤波、Itô 与 Feynman–Kac、均衡验证中的可积性、Nishida/Jackson 的 PDE 存在定理、加权函数空间上的谱扰动与一致余项、Gaussian 卷积积分恒等式，以及完整的联合设计命题和统计识别命题，均未在本目录中形式化。
