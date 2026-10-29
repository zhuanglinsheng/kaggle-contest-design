# PaperJK9：回应与证据索引

更新：2026-10-29。Task1、2、3的完整证据按清理后的范围维护；新增Task4既有检验的轻量结果报告，未恢复详细预测产物或重新计算。无关预测实验和旧产物已删除，详见[清理记录](cleanup_20261027/README.md)。下列入口代表当前结论；其他较早报告按其日期和当时方法理解。

|任务|已完成的工作与边界|当前入口|
|---|---|---|
|Task1|已核对匹配的左端努力/漂移、数据依赖初始化、实证与模拟的读数差异；确认对应关系，不扩展固定输入识别证明|[英文回应与实现对应](task1_implementation_response_20261024.md)|
|Task2|全精度范围和全局比较论证已复核；数值仍有1179个未确定点，四分类图保留这一范围，平滑不提高确定性|[理论与数值总报告](task2_theory_numerics_progress.md)、[限定口径](task2_response_and_delimitation_20261025.md)、[正式图更新](task2_manuscript_figure_update_20261025/change_manifest.json)|
|Task3|73场×九情景及附录表均已用原非线性模型重算；四例×两项先验/下界变化下均有同方向联合改善，七项原方案、一项调整幅度后通过|[两表原模型重算](nonlinear_table_rebuild_20261025/README.md)、[代表性方向敏感性](task3_nonlinear_prior_sensitivity_20261026/README.md)|

## Task4：既有结果说明

[Task4概率预测比较报告](task4_predictive_comparison_response_20261029.md)汇总此前完成的73场、四模型、两种切分检验：D优于自由活动核C，但低于固定提交率A，相对期限模型B的优势不确定。报告含英文回应、方法、可追溯的分数差及解释边界。完整日期切分仍评价小时计数；日级是否提交概率和直接公共信号回归尚未实施。详细产物已于2026-10-27清理，本报告不是重新核验或恢复完整计算包。

## 当前论文与最终证据

- [PaperJK9.pdf](../../../paper/PaperJK9.pdf)：当前稿已加入6.3 Additional Checks及附录D两表；下列两表完成审查记录对应原模型重算完成时的版本。
- [两表完成审查](nonlinear_table_rebuild_20261025/manuscript_stage/completion_audit.json)：全部73场、实际PDF数字及版面核对。
- [敏感性最终审查](task3_nonlinear_prior_sensitivity_20261026/final_audit.json)：四例、八组替代后验，完整抽样及方向判定。
- [Task2逐点不确定性清单](task2_cell_uncertainty_inventory_20261024.csv)：1179点，不把数值失败解释为不可行或均衡不存在。
- [Task1输入核查](input_diagnostics_audit_20261024.json)：73个冻结输入及六项诊断复算；其余旧核查按当时论文版本理解。

一阶近似仅用于参数估计；当前反事实设计、质量评价与比较采用原非线性模型。6322未确定的全披露基准仍传递为子组中位数范围；其他未解竞争候选保留。结果不是连续域全局最优认证。

敏感性支持的是所选四例与两项替代设定中的联合改善方向，不能外推到全样本、所有先验或精确幅度。2551的联合改善也不能解释为提高反馈精度本身有益。

## 维护与来源

- [当前状态](CURRENT_STATE_20261025.json)
- [工作记录](WORKLOG.md)：按时间保留，已删除实验的历史路径不再代表可用产物。
- [最新综合说明](task_list_findings_20261025.md)
- [内部英文回应草稿，未发送](task_list_response_draft_20261025.md)
- [清理与保留清单](cleanup_20261027/manifest.json)、[清理后核验](cleanup_20261027/verification.json)

三个历史名称带Task4的方法文件及`predictive_integration_20261025/`中的三个小文件被Task3历史成本后验核验复用，因此属于保留依赖；说明见[共享依赖](predictive_integration_20261025/README.md)。Task4预测结果和缓存已删除。

- [Additional Checks论文最小修改记录](additional_checks_manuscript_update_20261029.md)：第6.3节、附录D两表及过时表述的同步修正。

- [附录正式表格补全](additional_checks_publication_20261029/README.md)：预测表六项均值、统一95% bootstrap标记与先验表案例特征；日期期限基准差值恢复至四位小数，无新增估计。
