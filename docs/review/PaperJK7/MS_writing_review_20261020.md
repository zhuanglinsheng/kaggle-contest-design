> Historical review / 历史审查记录：本文对应文件名或正文注明的版本与日期。
> 原评论和阶段状态保留用于追溯；其中的问题、Pending 或未核查措辞不自动构成
> 对当前 PaperJK11 或长期结果的状态判断。当前代码、输入输出对应与运行步骤见
> [复现指南](../../REPRODUCIBILITY.md)及 [图表来源索引](../../PAPER_EVIDENCE.csv)。

# PaperJK7 全文写作与证据一致性审查

审查日期：2026 年 10 月 20 日。目标期刊按 Management Science 理解。

> **二次复核说明：本文件保留初审意见，问题定性、优先级及待办状态请以同目录 [逐项复核报告](MS_writing_recheck_20261020.md) 为准。** 二次核查覆盖本文件62个条目（含重复对象）；部分意见已降级为可选编辑建议，部分因论文已经明确说明而撤销。尤其“反事实机制冲突”应收窄为“机制衔接不够准确”，不能当作数学矛盾；时间切分、恢复实验限制和总节约/披露增量的区分均已在现稿说明。以下初审措辞不应直接作为尚待修复的错误清单。

**判断：当前版本还不宜视为达到 MS 投稿所需的写作成熟度。主要障碍不是英文语法，而是理论、经验估计、反事实和数值图形之间的证据层级没有始终对齐。** 论文已有完整结构，主要表格的数字也能对上结果文件；但一处机制解释冲突、一幅缺少数值不确定性说明的核心图，以及实际榜单与重建提交路径的概念混用，会明显影响审稿人对可信度的判断。这些需要先于语言润色处理。

本轮交付审查意见和可复核的数值核对程序，不直接改动正文、参考文献或科学结果。下文区分“已确认的表述或呈现问题”和“需要补充分析才能回答的问题”。涉及数学证明，只评价本轮读到的逻辑表达、依赖关系和符号一致性，不把阅读或编译成功视为证明已获独立认证。

## 审查范围和核对结果

逐段阅读 `paper/PaperJK7.tex` 及其三个实际载入片段 `appendix_100k_rows.tex`、`appendix_joint_design.tex`、`appendix_effort_recovery.tex`，检查全部 56 页 PDF 的页面缩略图，并放大检查关键机制图、数值分区图及近似误差图。`section_joint_design.tex` 没有被主稿载入，其中某些机制说明比当前正文更准确，不能把该文件的文字当作已经进入论文。

在临时副本中进行完整 LaTeX 和 BibTeX 编译，生成 56 页，无未定义引用、未定义交叉引用或 overfull 报错。现有 PDF 与重编译 PDF 的逐页提取文本相同。现有目录的 `.aux`、`.bbl`、`.log` 较旧，不能据此判断 PDF 漏引；PDF 实际已包含 Blum and Hardt。这不是稿件漏引问题。

数值检查使用当前统一先验版本 `results/reestimation/lambda100_sd225/`，不混用历史反事实结果。

| 内容 | 本轮检查 | 结果 |
| --- | --- | --- |
| 两个 benchmark 回归、四个 full/time-split 回归 | 从 73 场逐场数据独立计算 OLS、HC1 标准误、正态参考 p 值、R²、Spearman 相关 | 与保存结果及正文精度一致 |
| 表 1—3 | 后验均值、标准差、RMSD；表 1 的两组区间 | 与各自 summary 一致 |
| 表 4 | 逐场数据重算均值、中位数、10,000 次 bootstrap 区间 | 一致 |
| 表 7 | 从逐场政策结果重算两面板全部 9 行 | 一致 |
| 表 8 | 精度分类、方向分组、各组收益分位数 | 一致 |
| 表 9 | 73 行、13 列逐项核对；参数中位数再与估计汇总核对 | 一致，允许已印出的小数舍入 |
| 表 10 | 从 40 组恢复记录重算误差和覆盖率 | λ 为 17.0%、97.5%；σ 为 12.1%、82.5%，一致 |
| 表 11 | 所有行的系数、标准误、区间和 R² | 一致 |
| 表 12 | 与辅助报告中的尺度和秩相关汇总对照 | 未发现文字与结果文件冲突；未从链重新计算秩相关 |
| 主样本采样汇总 | 146 个 full/early fit 的诊断列 | 最大 rank R̂ 1.008519；最小 bulk ESS 487.14；最小 tail ESS 1005.19；divergence 合计 0 |
| 引用与编号 | 引用键、重复键、label 和 ref/eqref | 无缺失或重复 |
| release 检查 | `python3 -m pipeline.publication --check` | PASS：698 个文件、397 个 accepted-fit 引用 |

配套程序 `audit_numbers_20261020.py` 共通过 80 项分组检查，结果写入同目录 `numerical_audit_20261020.json`。其中部分检查包含整张表或多个统计量，不应把“80 项”理解为只有 80 个数字。release 的哈希核验只证明版本一致，不证明模型设定、近似误差或因果解释成立。本轮未重跑 397 组采样、非线性均衡分区或全部原始数据预处理。

## 必须优先解决的问题

### 1 正文与附录对反事实机制的解释不一致

**优先级 P1。位置：主稿第 1305 行，PDF 第 29—30 页；第 2614—2640 行；`appendix_joint_design.tex` 最后一段。**

正文说 no-feedback choices 与 Proposition 1 的 incentive logic 一致，并用“领先使双方松懈、保留不确定性维持竞争”解释表 7。这段末尾虽然说不直接验证理论机制，但仍把读者引向同一机制。

附录的计算却明确给出另一层含义：一阶质量系数中精度相关项乘以 `(1/c_i − 1/c_j)`，其大小和方向还取决于 μ₀；当成本相同或 μ₀ 为零时，该项消失。Proposition 1 正是在对称成本、零初始均值条件下成立，其优势依靠更高阶或非线性效应。表 7 因而不能展示该对称机制在数据中的大小。这里的关键不是 λ=0 与 λ>0 的边界区别，而是**经验计算所保留的机制不同**。

建议替换该段：

> The first-order design comparisons operate through an asymmetric incentive channel: precision affects expected quality only when effort costs differ and the initial perceived gap is nonzero. This channel vanishes under the symmetric conditions of Proposition 1. The empirical comparisons therefore describe a distinct implication of the maintained approximation; evaluating the proposition's mechanism requires higher-order or nonlinear counterfactuals.

同时将 “interior finite-precision mechanism” 改为 “symmetric finite-precision mechanism”。命题证明存在一个正的有限精度设计严格优于所有 full-disclosure designs，但没有证明最优点在正精度域内部取得。有限设计、内部最优点、边界极限是三个不同概念。

### 2 核心分区图隐藏了重要的未核验分类

**优先级 P1，已确认。位置：Figure 3，主稿第 535—555 行，PDF 第 11 页。**

图形用完整填色、平滑边界和 θ*、T*、λ* 展示三类最优区域。图注只解释非线性搜索和 smoothing-spline，没有说明未核验单元。

实际图源的 `three_class_comparison/metadata.json` 记录：

- 1,179 个此前未解决的点被补作三分类，其中 1,144 为零奖、19 为 full disclosure、16 为 finite precision。
- 926 个点存在求解失败、且解析 payoff screen 尚不能排除其竞争力的候选设计。
- 16 个点在点估计排序与保守 envelope 排序之间分类不同。
- 该轮分类没有新增均衡求解，metadata 明确限定为探索性三分类，并非已核验的全局分类。

来源：`results/theory/numerical_joint_plane_moderate_dense/three_class_comparison/{metadata.json,README.md,unresolved_point_decisions.csv}`。README 关于“图含 unverified note”的描述也不符合当前实际显示的 PDF；放大检查后未见该说明。

这是证据呈现问题，不是图形美观问题。仅写“命题不是完整分区”不能代替“图中一些数值分类尚未通过核验”。建议恢复未解决区域的留白、网纹或散点状态，或只展示可支持的区域；图注明确搜索集合、失败处理、平滑的显示性质。未完成核验时，避免用星号暗示连续设计域中的全局最优。

可用图注核心句：

> Shading identifies the best evaluated design family on the numerical search grid. Some cells remain unverified because potentially competitive candidates failed to solve. Curves are smoothed display envelopes and do not certify global regime boundaries.

若继续保留当前三分类，需要在补充材料中报告上述数量及相应网格总数，并让读者能看出这些点分布在哪里。

### 3 最近一次提交分数与实际 leaderboard 被混用

**优先级 P1。位置：第 253—264、581—584、958—981、2590、2689 行；Figures 1 和 6。**

主稿将 displayed score 定义为“最近一次提交分数”，并称选择依据为 reconstructed final private standings。附录又承认替代配对的 last-score ranking 不一定等于基于最佳成绩的 final leaderboard。表 4 则使用官方 public/private rank。

因此至少有三种对象：逐次提交的分数路径、官方榜单、由最后一次分数重建的排序。它们不应统一称作 public/private leaderboard。最终入选提交的评分也不必等于最后一次提交；当前模型的 y_T 胜负判定需要明确这种映射简化。

建议：

1. 数据部分先明确定义“carried-forward submission-score gap”，将其与官方 leaderboard ranks 分开。
2. 对 baseline 和 alternative pair 分别说明排序算法；除非用官方排名，不称 final private ranks，改为 reconstructed end-of-window private-score ranking。
3. 修改摘要、引言、结论中“final private ranks are used”的对应表述，使其与实际选择算法一致。
4. 明确终端最后分数与实际获奖提交之间的差异；如要声称拟合真实胜负机制，需补相应数据对照。

Kaggle 官方比赛示例说明最终成绩来自选定提交在 private leaderboard 上的表现，可作制度依据：[官方比赛提交说明](https://www.kaggle.com/competitions/mastering-ordinal-regression-with-wine-data/overview/making-a-submission)。这不证明历史 73 场各自规则相同，历史样本仍需逐场规则或数据字段支持。

### 4 近似收益与近似误差尚未建立可比较的证据

**优先级 P1，属于需要补充分析的问题。位置：Figure 5、表 7—8、第 1242 和 2602 行。**

固定奖和期限时，精度调整的中位质量增益只有 0.011%；表 7 在偏好较少披露的子样本中，相对重新优化的 full disclosure 的额外中位成本节约约为 0.56%—1.37%。现有直接努力比较只覆盖合成基准附近六组参数；优化后奖金中位数在 δ=100 时上升 493.08%。恢复实验也不覆盖这些优化后设计。

这不能据以断言收益错误。努力误差百分比和质量收益百分比的分母不同，**不能直接把 2.51% 的努力误差与 0.011% 的质量增益作数值大小比较**。但论文也不能仅凭现有误差图认定小收益的排序可靠。

建议至少选代表性参数和收益接近零、收益较大、奖金外推较大的设计，计算非线性质量、质量约束偏差、与 full-disclosure 的 payoff 差及排序是否改变。没有这一结果时，将表 7—8定位为 approximation-based sensitivity calculations，避免使用可执行推荐的语气。积分和精度网格的细化误差很小，只证明一阶公式算得稳定，不证明公式近似真实均衡足够准确。

### 5 文献综述存在模型归属和结论方向问题

**优先级 P1，部分已确认，部分需要按所引版本复核。位置：第 308—329 行。**

第 312 行把 Aoyagi (2010) 和 Mihm and Schlapp (2019) 放在“初始阶段是可行性不确定性或基础实验”的统一描述下。Aoyagi 的原文明确是各阶段 effort 加 noise 的随机绩效信号，不能将其归入 feasibility-uncertainty 模型。应分开写各自机制，再概括共同的信息披露问题。[Aoyagi 作者接受稿，第 2 节](https://ir.library.osaka-u.ac.jp/repo/ouka/all/3424/Aoyagi_2010-ip.pdf)。

第 327—329 行将 Huang et al. 概括为“更充分的信息让排名更可靠，但使高方差模型激励最强”。其当前 SSRN 摘要强调的是 **intermediate split ratios** 对 high-dispersion approaches 的激励最强，不能据此写成“信息越充分，激励越强”的单调关系。同时 training/test split 也不是 public/private leaderboard split 或本文 λ 的直接同义词。[作者主页](https://p9huang.github.io/)；[SSRN 记录](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4976827)。

该 SSRN 全文此次抓取受限，故应把具体版本的完整命题核对列为必要后续工作，而不是宣称已完整验证。当前摘要已足以说明现有概括过强。更稳妥的综述句为：

> Huang et al. study how data splitting changes contestants' incentives to choose modeling approaches. Their analysis highlights an incentive toward high-dispersion approaches at intermediate split ratios. We instead hold innovation risk fixed and study how public-signal precision changes effort through belief updating.

### 6 理论信息结构和经验信息结构之间的桥接不够集中

**优先级 P1，关系到贡献表述。位置：第 410、608—617、674—684 行。**

理论假设过去努力可观察、双方没有额外私人信息，因而公共滤波是正确 Bayesian posterior。经验实现却用连续回填的公开差距构造 perceived state，而似然另外使用离散 Kalman recursion；二者不是同一后验。正文已经逐项承认这些近似，问题在于这些承认散落多处，而摘要和贡献部分仍很容易被读成“同一个精确 Bayesian game 直接产生并识别经验似然”。

建议在估计部分开头集中用一个段落或映射表说明四层对象：连续公共信号的理论均衡；按提交更新的观测近似；用于行为的 perceived-state recursion；用于积分潜在路径的 likelihood filter。经验部分的 tilde y 统一叫 maintained perceived state。另用一句说明忽略本地验证、个人实验知识等私人信息是公共信息均衡的实质性限制，不能只说“past efforts as observable”是技术便利。

这不是要求删除框架；它是使读者准确知道哪些结论来自定理、哪些来自经验 measurement assumptions。

## 正文组织和论证强度

### 7 摘要与贡献段没有充分呈现主结论的适用条件

**P2。位置：第 170—181、277—289 行。**

摘要说 finite precision 严格优于 every full-disclosure design，应在同一句或紧接一句写出对称 contestants、足够分散的初始不确定性、低成本和低创新风险的条件。否则像一般性最优性结论。贡献二提到低成本和低风险，但仍略过关键初始条件。

摘要还缺少“经验近似”这一会改变解释的重要信息，反而用了较长的 path-wise holdout technical description。建议把条件、两项实证关系、经验设计的范围放入一个更紧的摘要，保留 pair-selection 条件即可。下文给出改写示例。

### 8 贡献主线需要统一

**P2。位置：标题、引言、Section 2.1 与 Section 5.3。**

标题承诺 structural inference，引言也主要从识别困难进入；篇幅很大的联合设计定理随后成为第二条主线，但经验反事实又没有检验该定理的对称机制。建议保留推断为主线：先解释为什么从 public submissions 能区分 innovation 与 information，再说明理论展示了何种管理问题值得识别，最后将一阶设计作为框架的有限应用。若决定以设计为主贡献，就需要更强的非线性经验反事实，不能靠标题调整解决。

### 9 引言重复且进入经济问题较慢

**P2。位置：第 204—289 行，PDF 第 2—4 页。**

平台列表、两类榜单、信息流、Bayesian updating、private validation 在不同段落反复介绍。建议将 Figure 1 前后两次流程讲解合并，平台举例缩为一句，把“管理者为什么不能直接选择最精确信息”作为清楚的问题。用可验证的 sample eligibility 替代 “carefully select 73 high-quality contests”。“high-quality”没有独立定义。

### 10 对其他结构估计工作的比较带有未经支持的评价

**P2。位置：第 370—376 行。**

“their CCP-based approach relies on a coarser behavioral approximation”是对比较方法的总体排序，没有证明或共同基准。本文自己也用了领先阶努力、连续回填和二人近似，不宜用方法标签暗示精度优越。直接写区别：submission/continuation decisions versus continuous effort；score learning versus public-signal filtering。若要使用 coarser，明确它粗化了什么状态，以及为何与本问题相关。

### 11 理论存在性结果的层级让读者误解

**P2。位置：Theorem 1，第 475 行；Lemma 1，第 1863 行。**

正文只声明小激励时均衡存在，附录却声明所有正有限精度及正参数下的 bounded classical equilibrium。这不构成形式矛盾，但正文可能让读者以为后面的低成本设计在存在性上没有覆盖。建议把一般存在性与小激励展开分开：注明存在性由附录 Lemma 1 支持，小激励条件用于误差展开；或解释为何正文刻意只采用较弱存在性陈述。

### 12 λ 的统计一致性与战略作用应分别命名

**P2。位置：Section 5.2 与第 287—289 行贡献三。**

private benchmark 支持 λ 的观测噪声解释，不单独证明 λ 通过 Q 影响努力；正文在第 1035 行已准确指出这一点。贡献三的 “rather than merely fitting ... behavior” 容易把这种 directional diagnostic 写成对完整战略结构的识别验证。建议明确观测解释、行为解释和反事实外推三种证据，避免把前者替代后两者。

### 13 时间切分不等于独立样本或实时预测

**P2。位置：摘要、引言第 269—272 行、Section 5.2。**

early fit 使用前 75% 数据而保持终点 T 的做法已说明清楚；但 focal pairs 仍由事后路径排序确定。应继续坚持 conditional diagnostic 的定位，不在后文或图题中换成完全外样本预测。后期 disagreement 还包含后期 public 数据，并非只含 private outcomes；可称 later public–private discrepancies，避免 “subsequent private outcomes” 让读者误会其输入完全私有。

### 14 回归缺少经济量级解释

**P2。位置：第 1103—1105、1207—1210 行，Tables 5—6。**

目前以 slope 和 p 值为主。λ 的 time-split R² 为 0.029，方向支持与解释能力应区分。log(1+x) 不是标准 log-log elasticity，不能将 −0.015 直接解释为百分比弹性。建议用样本中位数或四分位差给出原尺度 predicted difference，并把它描述为关联而非干预效果。该量级可由现有逐场数据计算，无需新估计。

### 15 二阶段不确定性需要与报告的 p 值同处说明

**P2。位置：Tables 5—6、Figures 7—8；第 2600 行。**

HC1 标准误只处理给定点估计回归的异方差；参数估计、选对手及时间插补的不确定性未传播，附录已经说明。建议将这句话放入统一回归表注，给出 HC1 的全称或简短解释；主图图注本身也应说明使用 HC1，不要依赖读者找到后置附录。若要提高推断强度，再考虑传播后验不确定性和数据构造误差。

### 16 稳健性结论应按参数分别总结

**P2。位置：Section 6、摘要和结论。**

替代对手时 λ benchmark slope 的 p=0.115，σ 仍显著。Section 6.1 当前措辞准确，应保留，不应在摘要或结论升级为“两项结果在所有稳健性分析中成立”。另外，替代对手分析重做的是 benchmark，不是 Table 5 的 disagreement/time-split 全套关系，不能替代这些检查。43 场的有效窗口也变了，不能只把差异归因于换对手。

### 17 先验与近对称弱识别的讨论可以更直接

**P2，补证据问题。位置：Proposition 2、Tables 1—3、第 2602 行。**

结构识别条件 c_i≠c_j 不提供有限样本识别强度。合成结果中 r 与成本后验高度相关，原文对此较为诚实；但 empirical design 使用成本差和 μ₀ 的联合符号作方向判断，正好依赖这些参数。建议报告近对称程度、方向后验概率和 prior sensitivity，区分“模型中识别”与“这 73 场充分识别”。没有这些检验时，75% posterior support 应称维护先验和近似模型下的方向支持，不能称一般可靠推荐。

### 18 数据窗口与真实 contest deadline 的区别需要前置

**P2。位置：第 975 行、表 7 表头 ΔT、结论。**

T 是构造窗口（含 initialization），不是官方比赛开赛到截止的精确时长。表 7 的 deadline change 会被管理者读作真实赛程改变。建议给出映射关系，报告两者差异的汇总，或明确这些是相对于 constructed model horizon 的比例变化。若要提出赛程建议，需先说明该映射对选手入场、实际可用时间的含义。

### 19 经验选择标准没有展示二人近似的集中度依据

**P2，补证据问题。位置：第 565—568、633—637、958—965 行。**

“dominant top rivalry is plausible”“competition concentrated near the frontier”是模型适用动机，当前五次提交、十天跨度和重叠规则主要保证活动量，不能证明第三名不构成战略竞争。建议补一项前列成绩/奖差或 top-2 相对其他选手的集中度描述；否则将 dominant rivalry 改为 focused pairwise approximation，避免把筛选解释成已验证的市场结构。

### 20 恢复实验的范围不能由覆盖率数字掩盖

**P2。位置：Appendix C.2、Table 10。**

λ 的覆盖为 39/40，σ 为 33/40，后者明显低于名义 95%；不能只突出 λ 的表现。当前附录已承认 σ 区间可能过窄，建议在经验估计局限处保留一句。十组 targeted configurations、每组四次模拟也不能充当全样本均匀误差保证，且 simulator/likelihood 的 interval-start convention 与经验 endpoint convention 不同。若不补端到端模拟，将本实验称 targeted recovery diagnostic，避免“验证了整个实证流水线”的表达。

### 21 大幅节约主要来自奖和期限调整 不能都归给披露

**P2。位置：表 7、第 1303—1305 行、结论。**

δ=100 的 89.93% 总成本下降，对应很强的设定性 delay cost 和大幅联合调整；额外的披露节约为偏好减少披露子样本内 1.37% 的中位数，二者分母和样本不同。当前表注已区分，正文应优先说清楚总调整与披露增量的关系，不让读者把 89.93% 读作披露政策收益。建议压缩极端 δ 的正文行，将 0.25、1、5 等代表情景留正文，其余放附录。

### 22 缺少一个清楚的可实施信息政策解释

**P2，管理含义。位置：Section 2.1 和结论。**

λ 是理想化 Gaussian signal precision，不自动对应 public test share、榜单四舍五入、更新频率或隐藏榜单。这些实际政策改变的信息和激励不完全相同。建议解释本文比较的是信号精度这一抽象工具，并列明可能的实施方式需要另建映射，避免直接推荐某个 train/test split 或减少 public data。补充最小政策例子会比再重复一遍“保留不确定性”更有管理意义。

## 图表与排版

| 对象 | 已确认的问题或改进点 | 建议 |
| --- | --- | --- |
| Figure 1，PDF 第 3 页 | 与 Figure 6 大量重复；把重建提交路径称榜单；“观察到”未区分观察者和时间 | 保留一幅主要流程图，标明 contestants/analyst 和 contest/post-contest |
| Figure 2，第 8 页 | 图较小，轴标题依赖图注；主要展示标准 Riccati 收敛 | 补正式轴标签；可移到附录，为经济结果留空间 |
| Figure 3，第 11 页 | 未核验分类被平滑完整区域遮蔽 | 按第 2 项修正，这是最高优先级图形问题 |
| Figure 4，第 18 页 | 合成样本横轴使用 2026 年具体日期，容易被快速浏览者当成实际数据 | 改为 days since start；legend 的 real output gap 改为 latent gap |
| Figure 5，第 18 页 | 标题 “Exact” 比实际 numerically solved nonlinear equilibrium 更强；只展示 player i | 改为 “Numerical Nonlinear Equilibrium and First-Order Approximation”；保留当前 player i 范围限定，若声称双边精度则补 j |
| Figure 6，第 22 页 | 相交箭头、框线和符号较密；估计/验证边界未显示事后选对手的作用；r 未显示 | 简化成估计输入、维持状态、留作诊断输入三部分，并加选择机制注记 |
| Figures 7—8，第 25、27 页 | 每幅右侧像软件输出；Omnibus、JB、condition number 占据注释；精确回归表与图重复 | 主图保留点、拟合线和必要 uncertainty display；合并统计表，常规残差诊断移附录 |
| Figure 9，第 30 页 | 相比正文主样本散点，图注更清楚地说明 HC1，格式不一致 | 统一各散点图的符号、transform、样本和标准误说明 |
| Tables 1—3，第 20—21 页 | 主列里均值、SD、RMSD 大量重复，说明很长；Panel B 不显示被估计的 S₀ | 正文只保留主要参数估计和区间；完整诊断移附录；给出 S₀ 后验或明确指向补表 |
| Tables 5—6 | HC1 正文图注不够独立，残差检验与主结论的关系不清楚 | 留 estimator、样本、区间、关键范围，补一句点估计误差未传播 |
| Table 7，第 29 页 | 列名 λ<∞ 容易被理解成可行的有限精度最优；实际正奖的少披露项全是 λ↓0 | 直接写 No-feedback limit λ↓0；标题或 panel 明示 boundary comparisons |
| Table 8，第 49 页 | 75% 分组不是 credible interval；0.011% 量级值得读者注意 | 保留当前范围限定，并在正文清楚区分收益小与方向支持 |
| Table 9，第 50—51 页 | 13 列数值可读性弱，成本小数舍入到 0.0xx 时相对信息损失较大；无不确定性 | 明确作在线补充表，提供机器可读 csv；考虑重要参数区间或单独 uncertainty 图 |
| 全部参考文献，第 53—56 页 | 期刊名和书名显示下划线而非期刊样式中的强调 | `\usepackage[normalem]{ulem}`，或不需要时移除 ulem；重编译检查 |

`ulem` 在主稿第 29 行未使用 normalem，改变了 `\emph` 的渲染；这一点已经从 PDF 确认。当前没有发现大面积文字越界或图表被裁掉，问题主要是密度、重复和强调样式。

## 参考文献核对

重编译版本共有 36 条被引文献。引用键和正文对应关系已检查，不能将 `.bib` 中所有未用条目当作参考文献清单错误。以下区分书目信息、引用内容和排版问题。

| 文献 | 问题 | 处理 |
| --- | --- | --- |
| Khorasani, Nittala, Krishnan (2023) | `pages={msom.2021.0378}` 把 DOI 后缀当页码，缺 volume/number | 改为 Manufacturing & Service Operations Management 25(6):2249–2267，DOI 不变。[出版社](https://pubsonline.informs.org/doi/abs/10.1287/msom.2021.0378) |
| Shakhgildyan | 当前引作 2022 online-first，缺正式卷页 | 已有正式版本 Review of Economic Design 29:545–583 (2025)。统一采用正式版本或明示所用 online-first 版本，正文年份同步。[出版社](https://link.springer.com/article/10.1007/s10058-022-00309-3) |
| Huang, Zhang, Erat | 未明确 working-paper 状态；摘要归纳可能改变原文结论方向 | 先按所引版本核对结论，再注明 SSRN working paper 与版本日期；2024 的原始写作日期不必机械改为 2026。作者主页仍列为待修订稿，不应自行标成已发表 MS 论文。[作者主页](https://p9huang.github.io/) |
| Aoyagi (2010) | 书目信息未见问题，正文将模型概括为 feasibility uncertainty 不准确 | 按第 5 项重写模型归属。[作者接受稿](https://ir.library.osaka-u.ac.jp/repo/ouka/all/3424/Aoyagi_2010-ip.pdf) |
| Blum and Hardt (2015) | 已被实际 PDF 正确收录；不能再报告“漏引” | 文献支持 adaptive overfitting 和 leaderboard reliability，不支持“保留不确定性维持竞争”的均衡机制；相关两种作用要分开。[PMLR 原文记录](https://proceedings.mlr.press/v37/blum15.html) |
| Barrau and Bonnabel；Frogerais et al. | 分别为 invariant/continuous-discrete extended Kalman filter，与这里的线性 Gaussian filter 不是最直接对应 | 保留 Bensoussan 作主依据；说明其余文献提供哪一项额外依据，或删除装饰性并列引用 |
| Bergemann et al. (2022) | 是 auction information disclosure，不能无说明地归为 static contest design 的同类结果 | 移入广义 information design 的背景句。[作者公开稿](https://economics.mit.edu/sites/default/files/2022-12/optimal%20information%20disclosure%20in%20classic%20auctions.pdf) |
| Moscarini and Smith | key 含 2007，正式打印为 2011，key 本身不是错误 | 核对采用的 working-paper 版本与正文陈述；无需仅因键名旧而改年份 |
| Mihm and Schlapp | journal 字段是 “Management science” | 统一为 Management Science；与其他模型分开概括，不把所有两阶段模型统一成 Poisson feasibility race |
| Hoffman and Gelman；Lewis and Shedler 等 | BibTeX 会把 NUTS、Hamiltonian Monte Carlo、Poisson 等专名小写 | 对必须保留的大写词加 braces，不对整段标题随意强制 title case |
| Meta-Kaggle | 连续更新的数据集，仅有 DOI 年份难以识别实际抽取版本 | 增加 extract/snapshot 日期和数据版本说明，区别于样本截止 2025-04-15；不要用样本截止日冒充下载日 |
| 全部书目 | ISSN、ISBN、URL 展示与 working-paper 状态不统一；大量 HTTP dx.doi.org 链接 | 依期刊样式统一字段，正式期刊补足卷期页；优先规范 DOI 展示 |

其余被引条目已检查本稿的引用对应及打印形式，本轮未逐篇阅读全文，也未对每个历史版本做完整书目数据库认证。尤其证明引用到的 Jackson Theorem 2.7、Nishida、Kato 和 Lorenzi 的具体适用条件仍需要专门数学核验，不能由本次文字审查代替。

## 符号和局部文字问题

1. **已确认的符号错误，P2**：主稿第 2158 行 `\|\partial_yP_v\|_{{\Lambda_{\max}}^\infty\to {\Lambda_{\max}}^\infty}` 将函数空间写成 Λ_max^∞。按周围 heat-semigroup bound 的语义，应核实并恢复为 `L^\infty\to L^\infty`。这很像替换符号时误改，当前编译不会报警。
2. **第 382—387 行路线图不全**：只写到 Section 5，实际还有 Robustness 和 Conclusion。补上 Section 6、7，或删去机械路线图。
3. **第 2590 与 2689 行的 tie-breaking 描述不同**：baseline 用最终更新顺序，alternative 用提交 ID/contestant ID。若是有意差异，明确写；若不是，应对照 producer 修正。当前不能只统称“same selection rule”。
4. **“path-wise holdout”**：容易被理解为整个样本选择都与 private data 独立。首次定义可保留，其后优先用 private paths excluded from the likelihood；具体选择条件集中交代一次。
5. **feedback、signal、disclosure precision**：若都指 λ，应首次指定术语对应，此后主要用 feedback precision；不要与 signal readout frequency 混为一谈。
6. **innovation risk 与 private volatility**：前者是 latent diffusion 参数，后者含 carry-forward 零增量和更新频率。现有正文区别正确，应保留，不能为简练而删掉。
7. **uncertainty、risk、noise**：S 是 filtering uncertainty，σ 是 latent innovation scale，1/(λh) 是 measurement variance；避免都简称 noise 而不指明来源。
8. **“independent Poisson randomness”**：保留 conditional-on-history 的限定，避免改写成两人的整个 submission processes 无条件独立；双方通过共同历史关联。
9. **“Exact”**：数值计算的均衡称 nonlinear numerical equilibrium 更准确；exact 只用于闭式或数学对象，别与数值求解混用。
10. **“Post. RMSD”**：当前脚注明确区别于跨数据集 RMSE，这一点是正确的；仍建议首次正文用完整定义，减少读者把单个后验的距离当作频率学风险。
11. **“Normality robustness”旧标签**：当前标题已改为 score calibration，正文正确承认 rescaling 不检验 Gaussianity。保留这一修正，不因历史脚本名称又写回“正态性得到验证”。
12. **零精度语言**：正文应始终用 λ↓0 limit；“optimal λ=0”只可作为闭包上计算的简称，并立即说明不可行边界。表头比脚注更适合直接表达这一点。

## 可直接采用的英文改写

### 摘要示例

> Public leaderboards provide feedback in crowdsourcing contests, but the performance that determines the winner remains hidden. We develop a dynamic contest model with noisy public signals and a Bayesian estimation framework linking effort to submission activity. For symmetric contestants, sufficiently dispersed initial uncertainty, and low effort costs and innovation risk, a positive finite-precision design strictly outperforms every full-disclosure design, even when prizes and deadlines are jointly chosen. A closed-form leading-order effort policy makes estimation tractable. Using 73 Kaggle contests, we find that estimated innovation risk is positively associated with private-score volatility and estimated feedback precision is negatively associated with public–private score disagreement. These associations also appear in time-split diagnostics. Private-score paths are excluded from the likelihood, although the focal pairs are selected using reconstructed end-of-window private scores. First-order design comparisons illustrate potential cost reductions from joint changes in prizes, deadlines, and precision. Their precision effect operates through contestant asymmetry and does not test the symmetric nonlinear mechanism.

此版本有意保留条件和经验计算边界。采用前应最终核对 baseline pair 的 ranking producer，确保 reconstructed end-of-window 的表述与实际一致。摘要应随主线选择进一步压缩，不必逐字照用。

### 理论贡献示例

> We show that noisy feedback can improve contest design even when precision has no direct cost and the organizer can also adjust the prize and deadline. The result holds for symmetric contestants with sufficiently dispersed initial uncertainty, low effort costs, and low innovation risk. It establishes strict dominance over full disclosure without requiring an attained interior optimum.

### 经验结果示例

> Across contests, the estimated parameters align directionally with diagnostics based on private scores excluded from the likelihood. The evidence is stronger for innovation risk: its benchmark association remains significant under alternative focal pairs, whereas the corresponding precision association does not. These diagnostics assess parameter interpretation conditional on the selected pairs and maintained observation model.

### 表 7 解释示例

> At δ=1, the first-order calculation selects full disclosure in 39 contests, the no-feedback limit in 24, and a zero prize in 10. Among the 24 contests favoring the no-feedback limit, the median additional cost reduction relative to reoptimized full disclosure is 0.84%. These are limiting comparisons under the positive-feedback restriction. Their ranking remains conditional on the output approximation, the duration grid, and the assumed delay cost.

## 达到 MS 写作成熟度的修改顺序

第一轮先解决会改变读者判断的内容：统一 §5.3 与附录的机制解释；修正 Figure 3 的未核验状态；统一实际 leaderboard、提交分数路径及配对排序；改正 Aoyagi/Huang 的文献归纳；将近似设计收益的证据范围写清楚。

第二轮处理论证组织：集中解释理论到经验的映射，给出贡献优先级；压缩重复的信息流讲解；加入必要的经济量级和管理含义；把条件写进结论本身，而不是在数页后用免责声明补救。需要新增的非线性反事实、弱识别检查等，应明确是科学工作，不能假装通过英文润色已经完成。

第三轮处理图表和语言：减少重复图，重绘高密度流程图，合并回归统计，移走软件式诊断输出，修正 ulem、函数空间符号和书目信息。再逐句统一术语、主语、时态和连接逻辑。

最后做投稿版本：当前全文 56 页，正文到第 31 页，附录从第 32 页开始，References 从第 53 页开始。若将约 21 页附录独立为 online appendix，仍需按实际重排后的正文加参考文献页数评估长度。MS 官方要求初投稿没有硬性页数上限，但邀请修改稿在 1.5 倍行距下为 32 页，online appendix 不计入；不能简单说“56 页违反初投稿规则”。建议正文加 references 朝 32 页内组织。[官方投稿规范](https://pubsonline.informs.org/page/mnsc/submission-guidelines)。

若准备双匿名投稿，另生成匿名版本，移除首页作者、机构、页眉作者名和致谢。当前是作者工作稿，保留这些本身不是研究写作错误；只是尚不能作为匿名审稿文件直接提交。官方规范同时强调问题应与管理决策相关，因此结尾应明确模型支持什么决策、何种实施映射尚未建立，而不是只重复技术贡献。

**完成标准**：同一个结果在摘要、正文、图注、附录中具有相同的适用条件；图中的每一种视觉确定性都与数值证据相称；模型对象与平台对象名称一致；收益、近似误差和不确定性各有对应证据；主要结论无需依靠读者读完附录后自行收窄。达到这些标准后，语言精修才会真正提高投稿质量。
