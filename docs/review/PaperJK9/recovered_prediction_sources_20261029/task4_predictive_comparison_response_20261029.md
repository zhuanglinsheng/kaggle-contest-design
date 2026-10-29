# Task 4：提交行为概率预测比较及回应

**论文：** PaperJK9  
**整理日期：** 2026-10-29  
**检验完成日期：** 2026-10-25  
**状态：** 根据保留的工作日志、完成审计和本对话中的既有结果整理；本次未重新估计或计算预测。Task4详细评分文件、训练后验及大型缓存已于2026-10-27按用户要求删除。本报告恢复结论说明，不恢复完整计算包，也不表示逐场原始结果现已重新核验。

## 可用于回应 Jussi 的文字

I completed a focused predictive comparison of the empirical first-order
equilibrium submission model with a constant-rate Poisson benchmark, a
deadline-dependent benchmark, and a free activity kernel within the same
filtering framework. The comparison covered all 73 contests under two
temporal train/test splits. The outcome was each contestant's hourly
submission count, evaluated using the joint posterior predictive probability
for the two contestants.

The equilibrium model improves the contest-weighted average predictive log
score relative to the free kernel in both splits, with positive contest-level
bootstrap intervals. However, it performs worse than the constant-rate
benchmark in both splits. Its average advantage over the deadline-dependent
benchmark is not clearly established. Thus, the comparison supports some
predictive value of the equilibrium-kernel restrictions relative to the free
kernel, but does not establish that the equilibrium model predicts at least
as well as simple nonstrategic submission models overall.

These results concern the leading-order empirical specification. They do
not independently identify strategic behavior or validate the nonlinear
counterfactuals. Prediction is conditional on the selected contestant pair
and the constructed time window. A daily probability-of-any-submission
exercise and a direct public-gap regression have been discussed but have
not been implemented.

## 1. Jussi 的要求与本次检验对象

Jussi 的原话是：

> Sharpen the empirical contribution: A predictive comparison against simpler, nonstrategic submission models would establish what the equilibrium restrictions add beyond the filtering model. Could you please suggest a focused set of checks that would be most informative?

本次检验的问题是：均衡所规定的提交规则，能否比简单的非战略规则更好地预测未来提交活动，以及这些限制在过滤框架之外具有多少增量预测价值。比较允许不利于论文模型的结果，不预设均衡模型必须获胜。

预测变量为参赛者在每个小时内的提交次数：

\[
Y_{k,t}=\text{参赛者 }k\text{ 在第 }t\text{ 小时的提交次数},
\qquad k\in\{i,j\}.
\]

此前已经进行的是**计数分布的概率预测**，不是只预测一个确定次数；目标也不是最终排名、私人分数或比赛质量。

## 2. 已经实际比较的四个模型

沿用原检验的A/B/C/D命名，避免与后来讨论的新回归模型混淆。

|模型|设定|在比较中的作用|
|---|---|---|
|A：固定提交率|每位参赛者独立的常数Poisson提交率|最简单的历史活动频率基准|
|B：期限驱动|在基础活动率上加入剩余期限的变化|检验简单时间规律的预测能力|
|C：自由活动核＋过滤框架|保留观测和过滤递推形式，放松提交核的期限与差距系数|检验特定一阶均衡核限制的增量价值|
|D：论文模型|论文经验实现的一阶均衡提交核及相应估计设定|待检验的均衡模型|

C不是后来提出的“直接用原始公共差距和时间做Poisson回归”。C的行为系数也影响过滤漂移；因此C/D比较的是同一过滤框架中的两套行为规则，不是完全固定感知路径后单独切换战略优化。

## 3. 时间切分、指标及核验

- 覆盖全部73场比赛。主切分在初始化前缀后将约75%用于训练，剩余部分用于预测；另采用完整源日期边界切分，作为时间切分检查。
- **完整日期切分仍评价小时提交计数。** 它不是按天的二元“是否提交”预测。
- 使用对应训练窗口的估计，不以全样本参数后验代替训练后验。预测阶段每小时先预测，再使用该小时末可获得的新公共读数更新状态；不使用未来私人分数或未来提交进行训练。
- 主指标为每小时双方联合提交计数的对数预测概率，先在比赛内平均，再对73场比赛等权平均，单位为nats/双方小时；分数越高越好。参数不确定性通过后验积分进入预测。
- 比较报告配对分数差和比赛层bootstrap区间；最终记录为500,000次bootstrap。两种切分高度重叠，不作为146场独立比赛合并推断。
- 当时已完成训练后验诊断、预测积分精度、原目标密度核验、逐小时计数匹配和稀有概率的独立复核；完成审计记录四模型、两种切分全部73例通过当时的数值检查。
- 另做了五种随机化PIT及零提交概率等校准检查。它们评价预测分布，不构成战略机制的识别。

这些是历史完成记录的汇总；本次没有用已删除的原始评分和后验重新执行上述计算。

## 4. 已完成检验的结果

下表的数值来自仍保留的工作日志。对数分数差为D减benchmark，正值表示论文模型更好。

|比较|主小时切分平均差|完整日期边界切分平均差|主要判断|
|---|---:|---:|---|
|D−A：固定提交率|−0.060447295|−0.056970332|两种切分的bootstrap区间均完全为负，D较差|
|D−B：期限驱动|＋0.114599333|历史记录平均为正，此处不复原精确点值|区间含零，不能确认D有稳定优势|
|D−C：自由活动核|＋0.081218105|＋0.083629989|两种切分的bootstrap区间均完全为正，D较好|

保留日志中有明确最终端点的D−C区间为：

- 主小时切分：**[＋0.006507456，＋0.174589470]**；
- 完整日期边界切分：**[＋0.011429330，＋0.173488057]**。

日志同时保留了较早预算的D−A区间，但这里不把它们混写成最终500,000次bootstrap的精确端点。保留的最终结论是：两个D−A区间均为负。D−B日期比较的下端接近零，其端点模拟误差也曾单独检查，不应以近零端点作强优势判断。

主切分中，D仅在25/73场比赛中优于A；逐一剔除任何单场比赛，D−A的平均差仍为负。因此，落后于A并非仅由某一个极端比赛驱动。

此前报告和本对话还记录：D低估零提交的概率，校准表现弱于A。这一项在此保留为**定性历史发现**；校准明细已删除，本次不恢复精确PIT统计量或声称重新核验。

## 5. 可以与不可以得出的结论

**可以得出：** 一阶均衡核相对同一过滤框架中的自由核具有一定平均预测价值；该结论在两种切分下保持。但这种增量不能单独解释为真实Nash行为的证据，也可能包含约束复杂度带来的作用。

**目前不能得出：** 均衡模型对未来提交的整体预测能力不弱于简单模型。固定提交率A在已有检验中更好，必须保留这一不利结果。

适用范围还包括：

1. D是用于经验估计的一阶均衡规则，不是Task3反事实所使用的原非线性模型重新估计。
2. 源数据的日期内时点是构造的；比较条件于事后选定的焦点参赛者对和给定构造终点，不是精确真实时钟下的实时外部验证。
3. 当前结果是逐小时滚动条件预测，不是训练结束时对整个剩余比赛路径的一次性无条件预测。
4. C/D仍共享部分行为限制，包括给定参数时双方相对提交份额恒定；本比较没有全面验证所有均衡限制。

## 6. 尚未实施的建议

下列方案仅经过讨论，没有新估计结果：

- 直接以公共差距、差距平方和剩余时间解释提交次数的非结构Poisson回归；
- 以“未来一个完整日期内是否至少提交一次”为Y的概率预测；
- 对应的历史活跃日期频率、Logistic回归benchmark，以及Brier score和日级校准比较。

日级事件概率更贴合日期数据的提议，不是已证明更优的预测目标。若另行开展，应事先固定目标、窗口、模型和评分规则，再完整报告所有比较；不能将新目标的结果与旧小时计数分数直接混合。

## 7. 来源与当前保存状态

- [工作日志](WORKLOG.md)：2026-10-25的基准比较、主切分完整结果及“Task4双切分完成”段落，是本报告定量结论的当前文字来源。
- [完成证据审计](completion_evidence_audit_20261025.json)：`prediction_C_primary_hour`、`prediction_C_complete_day`、`prediction_D_primary_hour`、`prediction_D_complete_day`和`task4_complete`记录当时全部验收状态及已删除评分/比较文件的哈希。哈希仅是来源记录，不替代已删除的文件内容。
- [本报告来源摘录](task4_report_sources_20261029.json)：保留整理时引用的日志行和审计字段，不新增科学计算。
- [清理记录](cleanup_20261027/README.md)：说明2026-10-27删除Task4详细评分、后验和大型积分缓存的授权范围。

**本次只新增轻量报告和来源记录，并更新索引；论文、科学实现、Task1–3证据、原始Meta Kaggle压缩包均未修改。**
