# PaperJK9：正式结果、源码与研究记录

当前目录保留方法说明、历史审查和冻结源码。计算工作文件已迁入仓库根目录的
`tmp/`；旧记录正文中的历史路径通过[迁移清单](../../storage_layout.json)定位。
历史记录里的进程状态、清理结论、附录编号和文件位置对应其记录时点。
当前目录约定见[文件分类](../../STORAGE_LAYOUT.md)，运行步骤见[复现指南](../../REPRODUCIBILITY.md)。

## 正式论文与结果

- [PaperJK9.pdf](../../../paper/PaperJK9.pdf)：当前正式稿。
- [正式发布目录](../../../results/releases/paperjk9-20261029/)：长期维护的精简证据与来源索引。
- [理论图](../../../results/releases/paperjk9-20261029/assets/numerical_regions_four_class_bw.pdf)、
  [逐点不确定性](../../../results/releases/paperjk9-20261029/evidence/theory_unresolved.csv)：保留 1,179 个未确定点。
- [非线性主表](../../../results/releases/paperjk9-20261029/evidence/table_numbers.json)、
  [固定精度附录表](../../../results/releases/paperjk9-20261029/evidence/fixed_precision_summary.json)：正式逐竞赛结果可重新汇总。
- [成本敏感性](../../../results/releases/paperjk9-20261029/evidence/cost_sensitivity/final_audit.json)：所选四例与替代设定的结果。
- [预测比较](../../../results/releases/paperjk9-20261029/evidence/prediction/)：恢复的分数、切分、bootstrap、源码与候选链索引。

## 方法与维护入口

- [Task1 实现对应](task1_implementation_response_20261024.md)。
- [Task2 理论与数值口径](task2_response_and_delimitation_20261025.md)。
- [Task3 原始搜索控制器](nonlinear_table_rebuild_20261025/)；
  当前可迁移入口为 `python -m pipeline.jk9_design`，输出在 `tmp/runs/design/`。
- [Task4 考古恢复源码](recovered_prediction_sources_20261029/)；
  当前入口为 `python -m pipeline.jk9_prediction`，输出在 `tmp/runs/prediction/`。
- [工作记录](WORKLOG.md)与本目录较早的说明保留其历史语境，不作为待执行任务清单。

一阶近似用于参数估计；非线性设计结果保留未确定比较与已搜索范围的限制。
此次整理处理结构、来源和运行路径，不要求重新执行整套科学计算。
