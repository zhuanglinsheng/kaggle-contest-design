# Additional Checks：论文最小修改记录

2026-10-29。按用户要求，在第6章增加6.3 Additional Checks，正文仅概述代表案例的联合改善方向及两种切分下预测比较的稳定结论，并引用附录与GitHub仓库。附录D新增表13（四例、两种成本设定的方向支持）和表14（三个预测基准、两种切分）。同步修正引言、成本尺度讨论、结论及联合设计附录中的过时“未检验”表述；现有主表和Task2图不变。

来源：task3_nonlinear_prior_sensitivity_20261026/README.md及其final_audit.json；task4_predictive_comparison_response_20261029.md及task4_report_sources_20261029.json。未新增估计或计算。

先验结果仅支持所选四例的联合方向，一例缩小奖金降幅。预测结果如实报告优于自由活动核、弱于常数率基准；期限基准未确认优势。历史预测原始文件已清理，未补造缺失的日期切分精确值与区间端点，附录明确当前来源范围。GitHub链接指向项目仓库；本次仅本地修改，未提交或推送。

核查：latexmk编译成功；最终日志无未定义引用及Overfull；正文第32页与新增附录第57页已渲染检查。

后续按用户意见将6.3节展开为两个短段落：说明四例选择目的、替代成本设定、重新估计及非线性质量评价；说明73场小时计数预测的训练/测试、三项基准和log score，以及结果范围。保留5.3节自动引用和蓝色GitHub链接。

再次按上下文修订：第6章导语补充6.3覆盖范围；6.3从5.3联合设计的成本敏感性切入，再说明5.2私人路径诊断与提交预测的关系，避免将本节写成独立任务报告。

6.3最终语气调整为收尾汇总：以Additional sensitivity checks support开篇，回顾已完成检查并引用附录与仓库；不使用We first examine式新分析开场，不把预测基准比较概括为全面通过。

术语与篇幅调整：6.3预测段压缩，采用标准homogeneous/nonhomogeneous Poisson model术语；附录表14行名同步，notes解释常数强度和随剩余时间变化的强度。

6.3预测段再次聚焦实质：与不施加均衡条件的非战略提交模型比较，正文不再列Poisson术语；保留同一过滤框架下限制的增量价值和平均提交率基准更好的结果。具体分布及第三项基准留在附录。

参考其他论文简短稳健性段落的写法后，将先验段压为两句：所选四例方向稳健、表13结果与蓝色GitHub链接。计算流程与单例缩小降奖幅度留在附录；预测段末改为直接引用表14，避免重复仓库链接。文风参考：Vasilopoulos and Tayler, Real Estate and Construction Sector Dynamics over the Business Cycle, LancasterWP2021_002, footnote 4。

预测段同样压缩为结果与引用：同一过滤框架下均衡限制提高样本外预测，两种切分一致；平均提交率基准更好的结果保留，方法不在正文重复。

按完整小节重组：总领句概括两项补充检查，统一汇总结果，结尾共同引用附录两表；GitHub链接明确指全篇支持文档，包含Task1实现核查、数值验证和补充分析，不仅先验敏感性。Task1报告已确认存在于本地仓库；本次未提交或推送。文风另参考Management Science文章Escaping Product Market Rivalry Through Innovation第5.3节（https://pubsonline.informs.org/doi/10.1287/mnsc.2024.08304）。

最终补入Task1：6.3以实现一致性、成本设定及均衡限制的预测价值统领；明确实现符合所述离散化、初始化及观测时点约定，不声称经验与模拟时点约定相同。两段汇总三项检查，附录两表和全篇GitHub入口统一置于结尾。

按用户要求将Task1实现核查从第一段移至第二段末尾，随全篇GitHub支持材料一起引出。第一段不再单列实现核查。

通读后整体语言修订：将抽象的equilibrium restrictions改为submission rule derived from equilibrium，直接说明预测小时提交次数；将未引入的four representative cases改为有选择目的的four contest scenarios（小节省及大奖金调整），两种时间切分也以首次引入语气描述。先验方向证据与预测比较分别表述，全文材料入口和Task1核查仍在第二段末。

按读者视角进一步修订：明确案例选取覆盖内容；不用抽象Equilibrium restrictions作主语，而直接说明对照模型在同一过滤框架下自由估计提交对公共反馈和剩余时间的反应，不施加均衡条件，再报告未来小时提交次数的预测比较。
