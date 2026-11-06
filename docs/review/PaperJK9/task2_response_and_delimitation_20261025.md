> Historical review / 历史审查记录：本文对应文件名或正文注明的版本与日期。
> 原评论和阶段状态保留用于追溯；其中的问题、Pending 或未核查措辞不自动构成
> 对当前 PaperJK11 或长期结果的状态判断。当前代码、输入输出对应与运行步骤见
> [复现指南](../../REPRODUCIBILITY.md)及 [图表来源索引](../../PAPER_EVIDENCE.csv)。

# Task 2：回应口径与未闭合数值范围

日期：2026-10-25。此文件归纳已执行的核查，不修改论文。Jussi 的要求允许 resolve **or clearly delimit**；以下明确采用目前有证据支持的限定口径，仍不宣称三区域已完整求定。

## 可用于回应 Jussi 的文字

I have rechecked the uniform-in-precision comparison and the global competitor-screening argument against their stated assumptions. I did not identify a further substantive error in these arguments. Their small-incentive thresholds are existence conditions, rather than calibrated error bounds for the empirical designs.

I also carried out additional numerical repairs and independent checks of payoffs, best responses, and output convergence. These improve the evidence for particular candidates, but do not certify the complete three-region partition. Some stronger-incentive, weak-feedback candidates still fail the validation checks, and the available caches do not recover the complete original candidate set. I have therefore retained the unresolved classifications and documented their scope. The current numerical appendix already describes the 1,179 provisional grid classifications, including the 926 points affected by potentially competitive failed candidates, and does not present the smoothed boundaries as robust regime frontiers. This qualification should be retained.

## 为什么暂不修改正文或把局部修复写成完整分类

1. `paper/PaperJK9.tex` 的 `Numerical illustration` 段已明确给出 14,553 点、1,179 个未解决点、926 个竞争失败威胁，以及按成功候选点估计临时归类和平滑边界的局限。该表述与当前可核对记录相符。
2. 原未解决点的展示分类是 finite 16、full 19、zero 1,144；不能从展示颜色反推类别已验证。逐点审查清单仍保留原临时类别和失败威胁。
3. 新恢复样本覆盖 2,040 个设计；147 个原失败设计中此前修复 6 个，尚余 141 个。924 个原标记点能恢复相关失败威胁，另 2 个缺少对应原设计缓存。完整候选覆盖不能默认成立。
4. 其余 253 个未解决点仍存在比较区间等问题；在未解决点中，1,156 点还有可竞争的单网格成功候选。这些不是额外新增的未解决点，不能与1,179相加。
5. 集中批次在 65 点上排除了恢复候选中的若干竞争组合；尚未移除全部相关失败/覆盖/独立核验问题。新的逆坐标强例也未通过原状态价值检查和空间加密。因此，本次没有新的可靠分类点可写回原图。

原附录中“失败候选按设计家族的完整分布未知”仍应保留：恢复子集能追踪一些小正精度和直接零精度失败，但原候选缓存并不完整，不能据此还原全部原失败的家族分布。

## 适用边界

- 理论审查覆盖统一精度量词、两端极限、小激励阈值及全局竞争设计排除的衔接；不是形式化证明认证。
- 完整三区域分类仍未得到数值认证。此处完成的是对剩余不确定性范围和现有文字准确性的核对，不把数值困难解释为均衡不存在或解析命题错误。
- 如果以后获得完整缓存或新的可靠强例求解器，可在相同诊断标准下继续推进。当前无需为已有明确披露的限制增加大段正文；报告和原附录已有足够位置保留这些限定。

## 证据

- [理论与数值总报告](task2_theory_numerics_progress.md)：完整证明定位、原分类和修复记录。
- [逐点不确定性清单](task2_cell_uncertainty_inventory_20261024.csv)：1,179 个原未解决点。
- [候选覆盖](task2_candidate_coverage_20261024.md)、[集中批次](task2_focused_batch_progress_20261025.md)。
- [最新逆坐标与独立审查](task2_inverse_findings_20261025.md)：强例拒绝依据及源快照入口。
