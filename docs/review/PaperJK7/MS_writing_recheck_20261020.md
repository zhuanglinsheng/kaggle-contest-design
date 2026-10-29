# PaperJK7 审查意见逐项复核

复核日期：2026 年 10 月 20 日。本文件是对上一轮审查意见的复核，优先于初审报告中的问题定性和优先级。论文正文未修改。

**复核结论：部分具体问题得到确认，但上一轮把若干编辑建议和论文已经承认的限制定性过重，应当纠正。尤其不能把“机制衔接不够准确”说成数学矛盾，也不能把已经清楚标示的一阶反事实说成未经披露的精确预测。**

下列 62 个条目覆盖原报告的 22 个正文意见、14 行图表意见、12 行文献意见、12 个局部意见，以及 2 项投稿安排。不同部分存在重复，62 不代表 62 个独立错误。结论分五类：

- **确认**：存在可定位的事实、符号、名称或呈现问题，需要修改。
- **收窄**：意见有依据，但原来的范围、严重程度或推论过强，按本轮限定保留。
- **已说明**：论文已经处理或限定，撤销“现稿未说明”的批评。
- **编辑建议**：属于可选的组织、措辞或版式选择，不能当作科学错误。
- **待证实**：缺少足够证据，不能要求作者按已确认错误修改。

P1 表示会明显影响读者对核心证据的判断；P2 是明确应改进的准确性或解释问题；P3 是编辑优化。它们是本次审查的工作优先级，不是 MS 的正式分级。

## 主要意见逐项复核

| 编号 | 原意见 | 本轮结论 | 再核证据和应采取的处理 |
| --- | --- | --- | --- |
| M01 | 正文与附录的反事实机制冲突 | 收窄，P2 | §5.3 第 1305 行已经说“不直接验证”，不能认定数学矛盾。附录 A 中精度项确在成本相等或 μ₀=0 时消失；数值积分复核也支持这一性质。应把“consistent with the incentive logic”收窄为共享直觉，并在正文明确一阶不对称渠道。`interior finite-precision mechanism` 改为 `symmetric finite-precision mechanism` 仍有必要。 |
| M02 | Figure 3 没有披露未核验分类 | 确认，P1 | 论文图与 three_class_comparison 的源图 SHA256 相同。直接统计原始 CSV 得 14,553 个网格点、1,179 个原未解决点，全部位于显示范围内；926 个点有可能竞争的失败候选，16 个点两种排序不一致。图注未报告这些状态。措辞改为“未充分说明”，不推断作者有意隐藏。 |
| M03 | 最后提交分数与官方 leaderboard 混用 | 确认，P2 | `Leaderboard.refresh` 直接替换该人的分数，不取历史最大值；baseline 选择调用重建排序。当前 rebuild 保留旧 focal pairs，alternative 用最后记录排序。附录明确说不一定等于真实 final leaderboard，而摘要、§3.3、结论仍用 `final private ranks`。应统一名称。尚未逐场对照官方名次，不能进一步声称选错了多少对手、样本有多大偏差。 |
| M04 | 近似收益尚无相应误差证据 | 收窄，P2 | §5.2 末、§5.3 标题与表 7 注、附录计算及结论均明确是近似；撤销“未披露限制”的读法。保留科学证据缺口：六组努力比较和 targeted recovery 没有量化优化后质量约束误差。可补非线性反事实以增强结论；不能据此宣布表中收益错误或要求删表。 |
| M05 | Aoyagi/Huang 文献归纳不准确 | 收窄，P2，Huang 有版本限定 | Aoyagi 的模型是各阶段 effort 加随机绩效冲击；原句实际写的是“feasibility uncertainty or foundational experimentation”，并未明确逐篇对应，不能强读成Aoyagi必定被单独归入feasibility。保留分开精确描述的建议。Mihm 作者稿则区分初始方案和改善努力。Huang 当前 SSRN 摘要强调 intermediate split ratios，支持收窄“more informative … strongest incentive”；尚未取得全文及作者实际使用版本，不能宣称已核实其所有命题或该段每个方向。 |
| M06 | 理论与经验信息结构未桥接 | 收窄，P3 | 第 608—617 行集中讨论 continuous versus hybrid filter；第 674 行明确分开 likelihood filter 和 strategic beliefs；第 684 行说明两层近似。这是已披露的结构。可加一张映射表改善阅读，不应作为框架自相矛盾或重大漏报。私人信息假设也已在第 410 行明示；增加现实含义属于扩展。 |
| M07 | 摘要遗漏所有定理条件 | 编辑建议，P3 | 摘要写 `can strictly outperform`，是存在性表述，不是声称对所有参数成立；贡献段也给出低风险、低成本条件。不能要求摘要复制全部定理假设。可加 `under sufficient conditions` 或简述对称设定，提高聚焦度。 |
| M08 | 标题与理论设计主线未统一 | 编辑建议，P3 | 推断方法加设计应用可组成同一篇论文，没有逻辑冲突；主线选择是编辑判断。可加强过渡，不能据此断言标题错或必须改为设计论文。 |
| M09 | 引言重复、high-quality 太主观 | 编辑建议，P3 | 确有平台列表与两次流程讲述；可压缩，`high-quality` 可改为满足具体 eligibility criteria。没有发现该词带来数值或识别错误。 |
| M10 | 用 coarser 评价 Lemus–Marshall 缺少依据 | 收窄，P2 | 再读作者公开稿，CCP 实现用了本人最大分数、前十名分数、时间等状态；不能仅因其为 CCP 就判为更粗糙。本文的两人、leading-order 实现也有简化。保留“应说清具体粗化了什么”的建议，不据此断言对方方法更精确。 |
| M11 | 小激励存在性与全局存在性不一致 | 编辑建议，P3 | Theorem 1 是较弱存在性加展开；附录 Lemma 1 声称较广存在性。强结果蕴含弱结果，不矛盾；附录后置也符合常见组织方式。可在正文交叉引用较强结果，未完成其全部数学假设认证。 |
| M12 | λ 的观测解释与战略解释混淆 | 收窄，P2 | 第 1035 行已明确 benchmark 不检验其通过 Q 和 effort 的作用；第 287—288 行的 `assess whether` 也不是“证明整个结构成立”。撤销后者的强读法。仍建议把这条界限前置到贡献摘要，以免方向一致性被读成战略因果验证。 |
| M13 | 时间切分被当成完全独立样本 | 已说明 | 摘要、引言、§5 开头、§5.2 和结论均保留事后选对手条件；未发现声称完全独立的文字。`later private outcomes` 可更明确涵盖 public–private discrepancy，但不构成数据泄漏被隐瞒的证据。 |
| M14 | 缺少经济量级解释 | 编辑建议，P3 | 表和正文已给出 slope、R²；没有发现把 log(1+x) slope 误读成弹性的句子。四分位差或原尺度差异可提升管理可读性，但上一轮关于“不能读成弹性”是防错提示，不是现存错误。 |
| M15 | HC1 未传播估计和插补不确定性 | 收窄，P2 | 附录 C 第 2600 行明确披露。问题是主图表注对推断口径不够自足，而非没有披露；可在主图/表注补一句。不能把 HC1 本身说成用错。传播多层不确定性属于额外分析。 |
| M16 | 两项关系被说成在所有稳健性中成立 | 已说明 | §6.1 正确给出 λ 的 p=0.115，且明确 σ 证据更强；摘要、结论没有写“all robustness checks”。43 场 count-window 变化已在附录说明，本轮重新比较输入确认。保留现稿，不再作为错误。 |
| M17 | 近对称与先验敏感性需要讨论 | 收窄，P2 | 表 1—3及正文已承认 r 与成本分离有限。完整参数向量在对称点不识别，不等于 λ、σ 各自必然弱识别。应避免后一推论。可针对政策方向所需的联合符号做敏感性分析，不能仅凭成本接近就宣布该场估计无效。 |
| M18 | 模型 horizon 与官方 deadline 不同 | 收窄，P2 | §5.1 第 975 行明确写了窗口与官方赛程不同，因此不是隐藏处理。保留表 7 解释建议：ΔT 是相对模型窗口的变化，转成真实赛程建议需要映射。未重建 73 场官方开赛时间差，不能量化偏差。 |
| M19 | 二人模型缺少 dominant rivalry 依据 | 收窄，P2 | 稿件明示为 pairwise approximation；activity/overlap 只能证明活跃和重叠，不能证明前二人占据全部战略竞争。建议把 `dominant` 的语气收窄，或补集中度描述。没有证据说明73场的二人近似全部不适用。 |
| M20 | 恢复实验范围和 σ 欠覆盖未交代 | 已说明 | Appendix C.2 明确十组配置、四次重复、σ uncertainty can be understated，以及 start/end convention 不同。39/40 与 33/40 再核一致。撤销“覆盖率掩盖范围”的定性；是否增加实验是科学增强项。 |
| M21 | 89.93% 总节约被全归因于披露 | 已说明 | 表 7 明确单列 total saving 与 additional saving，注明不同样本，正文没有把 89.93% 叫披露增益。两数分母/样本差异真实存在，但论文已经区分；撤销误归因指控。情景压缩仅是版式建议。 |
| M22 | 缺少 λ 到实际政策的对应 | 编辑建议，P3 | 文中主要讨论理想化精度，没有直接推荐某个 train/test split。可加实施含义，但不能把没有提供一对一映射本身说成错误，更不能要求在抽象理论中强行设置该映射。 |

### M01 的额外核验

附录写 `A=G_q(μ₀){T(1/c_i+1/c_j)/2+(1/c_i−1/c_j)H}`，其中 H 在 μ₀=0 时为零。这一代数性质不需要依赖数值求解。另以三组 μ₀、三组 Q 对附录 Gaussian identity 作独立数值积分，最大绝对差为 1.74×10⁻¹⁷；在代表性参数下，分别检验对称成本、零均值时 A 对精度不变，以及正 alignment 时降低精度提高 A。这验证了“一阶渠道不同”的依据，不是对完整均衡或高阶机制的认证。

但“一阶渠道不同”和“共享保留不确定性的经济直觉”可以同时成立。因此，上一轮的“正文与附录冲突”措辞过强。更准确的修改是：正文明确同一宽泛直觉下存在不同数学渠道，避免用理论命题替代经验渠道的证据。

### M02 的额外核验

源图 SHA256：`8cc165c617da012a4715975a02729160aea7425d1db95554a366ee775528619b`，与 `paper/numerical_regions_bw.pdf` 一致。源 CSV 的 14,553 点全部在当前坐标范围内，1,179 个原未解决点约占 8.10%。

`pipeline/theory/joint_design_three_class.py` 将这些点改作最佳成功候选的三分类；`joint_design_moderate_plane.py:draw` 在 three-class/smoothing 分支不显示 unresolved 图例或核验提示，使用平滑曲线填色和星号标签。故“图的确定性高于保存的数值核验状态”仍是本轮最需要优先修改的呈现问题。

这不证明任何一个最终分类一定错，也不证明 θ*=0 区域的经济机制错误。应披露核验状态，并考虑显示未解决区域，不能把“未核验”写成“已证伪”。

### M03 的额外核验

`metakaggle/leaderboard.py:refresh` 对已经存在的 team 先移除旧 score，再插入新 score；`select_2_strongest` 从该重建榜单选对手。当前 `rebuild_main.py` 按旧 gap signatures 识别并保留 pair，没有重新用官方 rank 选择。`chapter6_data.py` 明确用 last private score 排 alternative candidates。

因此需要修正的是“数据对象及排序来源的名字”，不是现阶段就替换样本或重新选人。另一个历史脚本 `pipeline/robustness/build_data.py` 确实使用官方 final rank，但它不是当前 alternative-pair 数据的 producer，不能混用它来推断当前样本。

Kaggle 页面在本轮未返回可读正文；上一轮引用的单个官方比赛示例也不能认证历史 73 场规则。关于官方榜单/最终提交与最近一次分数的具体制度差异，应依历史规则和字段说明核对。该外部访问限制不影响上述代码中“最后值覆盖旧值”的直接证据。

## 图表意见逐项复核

| 编号 | 对应对象 | 本轮结论 | 核查后的处理 |
| --- | --- | --- | --- |
| F01 | Figure 1 与 Figure 6 重复 | 编辑建议 | 两幅图的信息有重叠，但分别承担介绍流程、估计边界的功能，不能说必须删除一幅；术语问题按 M03 处理。 |
| F02 | Figure 2 小且缺轴标题 | 编辑建议 | 轴含刻度，图注明确 t、S；不存在无法解释的轴。补轴名或移附录可提升可读性。 |
| F03 | Figure 3 的分类与边界 | 确认 | 详见 M02；原始核验状态、光滑显示和全局最优三者须区分。 |
| F04 | Figure 4 使用 2026 日期和 real gap | 编辑建议 | 图题已经明确 synthetic，不应称伪装成真实数据；相对天数与 latent gap 更统一，属可选改进。 |
| F05 | Figure 5 使用 Exact | 收窄 | Exact 常用于与解析近似区分的数值基准，并不必然错误；但此处明确是数值解，标题改为 nonlinear numerical 更精确。图注已清楚限于 player i，不要求为兑现现有声称而补 j。 |
| F06 | Figure 6 密集且没有 r | 编辑建议 | r 缺席不证明模型遗漏该参数；这是概念图，不必列全部参数。选对手条件加注可帮助理解，但该图也不是完整因果识别 DAG。 |
| F07 | Figures 7—8 软件式统计输出 | 编辑建议 | Omnibus/JB 等不是算错；删减和图表合并是版式选择。HC1 自足性按 M15 处理。 |
| F08 | Figure 9 与主图说明口径不同 | 收窄 | alternative 图注明说 HC1，主图只在后置统一说明。建议统一，以读者可独立理解为由；不声称其估计方法不同。 |
| F09 | Tables 1—3 重复且缺 S₀ 后验 | 收窄 | 多组统计有解释价值，可保留；Panel B 的 S₀ 作为 nuisance 被积分出，已在正文说明，不是漏估计。若讨论 free-S₀ 模型识别，补 S₀ 摘要有帮助，不能要求主表必须报告。 |
| F10 | Tables 5—6 的 HC1 和诊断说明 | 收窄 | 方法和不确定性已经在附录说明；主表增加推断口径是准确性/可读性改进，残差检验本身非错误。 |
| F11 | Table 7 的 λ<∞ 表头 | 收窄 | 表注与正文已明确所有正奖少披露方案都是 no-feedback limit，不能说读者完全没有被告知；直接用 λ↓0 作列名仍更准确。 |
| F12 | Table 8 的75%支持和0.011%收益 | 已说明 | 表注已说 across-contest percentiles 不是 credible intervals，也明确 limiting comparisons；数值再次核对无误。 |
| F13 | Table 9 过密且舍入损失大 | 收窄 | 放大后数字可读，无确认的列重叠。初审把 0.0xx 的舍入说得过重：该数量级三位小数通常保留多个有效位；如需相对误差应逐值计算，不能泛指严重失真。可改善间距并提供 CSV，缺区间是信息选择而非表错。 |
| F14 | References 的下划线 | 确认 | 再渲染第54页，期刊名、书名确有下划线。主稿使用无 normalem 的 ulem，覆盖 emph。若按通常期刊强调样式，应设 normalem 并重新检查。 |

## 参考文献意见逐项复核

| 编号 | 对应文献 | 本轮结论 | 核查后的处理 |
| --- | --- | --- | --- |
| R01 | Khorasani et al. | 确认 | `.bib` 缺卷期且 pages 为 DOI 后缀；出版社正式为 25(6):2249–2267 (2023)。补足，不需改 DOI。 |
| R02 | Shakhgildyan | 确认 | 出版社同时载明 2022 online publication 与 2025 issue date，正式卷页29:545–583。应更新正式书目；2022有真实来源，不应叫虚假年份。 |
| R03 | Huang et al. 的书目状态与内容 | 收窄 | working-paper 状态可写得更清楚，2024原始日期可保留。当前摘要支持 intermediate split ratio 的表述；全文和所引历史版本未核实完，不能指定其所有比较静态结论。 |
| R04 | Aoyagi 的模型归类 | 收窄 | 原文模型是累计随机绩效，但本稿用的是“feasibility or foundational experimentation”的集合概括；不能断言它逐篇明确写错。按M05建议拆分、精确归属，不再定性为已证实的单篇误引。 |
| R05 | Blum and Hardt 漏引及用途 | 已说明 | 论文实际 PDF 已包含该条，脚注用它支持 overfitting，未把其当作均衡努力命题的证明。撤销任何“当前漏引或误用”的读法。 |
| R06 | Barrau/Frogerais 不是最直接引用 | 编辑建议 | 与线性 Gaussian filter 不完全对应，但可以作相关算法背景；没有证据说引用错误。保留、精简或解释均可。 |
| R07 | Bergemann 是 auction 而非 contest | 收窄 | 原段也写广义 mechanism-design 背景，跨领域引用合理；不能说不允许引用。建议将 auction information design 显式并列，避免领域归属含糊。 |
| R08 | Moscarini 引用键2007但年份2011 | 已说明 | 引用键不参与发表年份的真实性；未发现由键名造成的当前错误。所用 working-paper 版本仍可作常规核对，但不列为已证实问题。 |
| R09 | Management science 大小写 | 确认 | `.bib` 的期刊名称确未规范大小写；改为 Management Science 是书目样式修正，不影响研究结果。 |
| R10 | 专名大写保护 | 收窄 | PDF 确把 Hamiltonian、Poisson、Kalman、Tullock 等专名小写，可加 braces。上一轮提到 NUTS 不够精确：该文题目是 No-U-Turn Sampler，而非字面“NUTS”；应保护真正打印的词，不凭简称修改原题。 |
| R11 | Meta-Kaggle 提取版本日期 | 收窄 | manuscript 未明确完整抽取快照，DOI/样本截止不能替代下载日期。仓库已有冻结输入与哈希，不能说数据没有版本控制。可补 extraction/version 信息，本轮未独立查明实际下载日期，不应由压缩包时间推定。 |
| R12 | ISSN/ISBN与HTTP DOI不统一 | 编辑建议 | 内容存在且样式不一致，但 HTTP dx.doi.org 不是错误引用，也不证明链接失效；正式书目完整性与可选字段统一应分开。 |

本轮再核的原始来源：

- [Khorasani 出版社记录](https://pubsonline.informs.org/doi/abs/10.1287/msom.2021.0378)，支持 R01。
- [Shakhgildyan 出版社记录](https://link.springer.com/article/10.1007/s10058-022-00309-3)，区分 online 与 issue 年份，支持 R02。
- [Aoyagi 作者接受稿](https://ir.library.osaka-u.ac.jp/repo/ouka/all/3424/Aoyagi_2010-ip.pdf)，模型第2节，支持 M05/R04。
- [Mihm and Schlapp 作者稿](https://sites.insead.edu/facultyresearch/research/doc.cfm?did=62678)，模型和反馈机制，支持分开概括。
- [Lemus and Marshall 作者稿](https://g-marshall.github.io/kaggle1.pdf)，2020年版本的估计部分及 Appendix B.5.1，支持 M10。没有用这个历史版本替代对正式版本所有细节的认证。
- [Huang SSRN 记录](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4976827)的可检索摘要及[作者主页](https://p9huang.github.io/)，支持 R03 的有限结论；直接全文访问仍受限。
- [Blum and Hardt 原始出版记录](https://proceedings.mlr.press/v37/blum15.html)，支持 R05 的范围。
- [Bergemann et al. 作者公开稿](https://economics.mit.edu/sites/default/files/2022-12/optimal%20information%20disclosure%20in%20classic%20auctions.pdf)，研究的是 auction information design，支持 R07 的归属。

## 局部符号与措辞逐项复核

| 编号 | 原意见 | 本轮结论 | 核查后的处理 |
| --- | --- | --- | --- |
| S01 | Λ_max^∞ 被写作函数空间 | 确认，P2 | 主稿2158行确为 `\Lambda_{max}^\infty`；上下文是 heat-semigroup derivative 的 sup-norm bound。应核实并改为 `L^\infty\to L^\infty`，没有其他把 Λ_max^∞ 定义为空间的文字。 |
| S02 | 路线图只到 Section 5 | 编辑建议 | 路线图遗漏 Robustness/Conclusion 属组织上的不完整，但论文没有义务列尽每节；补或删均可，不列科学错误。 |
| S03 | baseline/alternative tie-breaking不同 | 已说明 | 附录分别写清两种规则；再查 legacy refresh 的 bisect_left 与 alternative 的 TeamId 排序，确为有意保留的不同代码规则。没有发现稿件声称二者 tie-breaking 完全相同。可统一以便利复现，但无需因不同就判错。 |
| S04 | path-wise holdout 容易暗示独立 | 已说明 | 首次使用及后文均有选择条件，未发现无条件独立验证的承诺。可缩减术语重复，不能说已构成错误。 |
| S05 | feedback/signal/disclosure precision混用 | 编辑建议 | 多个词都指 λ，在当前上下文可理解；没有发现把频率直接当精度的数值替换。术语统一是可读性优化。 |
| S06 | innovation risk与观测volatility不同 | 已说明 | §5.2明确说明carry-forward、更新频率和非直接latent volatility；保持现有界限即可。 |
| S07 | risk/noise/uncertainty指代 | 编辑建议 | σ、S、λ相关方差已有定义；未确认某一公式把三者混用。只保留有具体指代时的行文改善。 |
| S08 | 两玩家Poisson不能无条件独立 | 已说明 | §3.3第659行明确给定 preceding public history 后的生成机制。原文没有声称整个路径无条件独立，不需要“修复”这一不存在的错误。 |
| S09 | Exact用于数值基准 | 收窄 | 同F05，宜更精确，但不把常见对照用法当成伪称解析解。 |
| S10 | Post. RMSD可能误认为RMSE | 已说明 | 表1注已给完整定义并明确不是跨数据集RMSE；后表引用该定义。没发现数值含义被混用。 |
| S11 | 旧normality标签会误导 | 已说明 | 实际节标题为score calibration，正文明确rescaling不验证Gaussianity；旧label仅为内部引用，不在PDF向读者宣称检验正态性。 |
| S12 | λ=0被写为可行最优 | 已说明 | §5.3脚注、表7注、表8和附录均称limit；未发现把其作为可行点取得的实质主张。表头可按F11更清楚，但不再说全文边界处理错误。 |

## 投稿安排复核

| 编号 | 原意见 | 本轮结论 | 核查后的处理 |
| --- | --- | --- | --- |
| T01 | 56页与页数要求 | 已说明 | 初审已经准确区分初投稿无硬上限、受邀修改稿32页1.5倍行距、online appendix不计。再查官方规则未变。不能把56页作者稿判为初投违规。 |
| T02 | 双匿名版本 | 编辑建议 | 当前保留作者和致谢是工作稿状态，不是写作错误；提交匿名审稿版时再移除。[MS官方规范](https://pubsonline.informs.org/page/mnsc/submission-guidelines)同时支持T01与T02的范围。 |

## 数值复核与执行记录

重新执行初审的80项分组检查，80项通过、0项失败；release检查再次PASS。本轮另用 `recheck_evidence_20261020.py` 进行：

1. 直接统计原始分类CSV，并校验论文图与图源哈希。
2. 从73场辅助后验中位数重算表12四个Spearman相关，结果分别为0.992379、0.999044（λ）及0.945144、0.995927（σ），与论文舍入一致。
3. 逐场比较 baseline/alternative 输入的count_start_hour，确认43场不同。
4. 重算76/146个player–contest的w中位数低于1、整体中位数0.909035、34/73场max-w中位数低于1。
5. 独立数值积分检查附录Gaussian identity和一阶精度项消失条件。
6. 再渲染参考文献第54页、附录表第50页，确认下划线问题，未确认表9越界。

80项核对是从保存的结果/逐场数据出发，不能称作从原始Kaggle数据完整复现。部分主回归正文数字、表8与表10的比较值在审计程序中按审查时抄录，未实现对任意未来文字修改的完整自动提取。该程序适用于本次快照复核，后续修改需同步人工核对，不能当作投稿论文永远正确的自动证书。

完整非线性均衡、所有采样链、历史榜单规则和证明引用的全部技术假设仍不在本轮认证范围内。这些范围不能因为“逐项复核”而被自动升级。

## 修订后的行动清单

先修Figure3的证据说明，再修榜单/提交路径/排序来源的名称、具体文献归纳、函数空间符号及书目信息。随后集中修改§5.3机制衔接和主图表注的推断口径。新增非线性反事实、弱识别及政策映射属于可提高科学说服力的工作，应与语言纠错分别安排。

关于“达到MS水平”，本轮能给的是有依据的编辑判断和可执行的改进项，不能给出期刊接收保证。也不应以待扩展研究或个人版式偏好为由，把论文已经诚实限定的结论重新判为错误。
