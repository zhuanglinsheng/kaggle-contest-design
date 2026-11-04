# 仓库文件分类与目录约定

本次整理的验收标准是：来源、代码、配置和运行步骤完整，文件用途明确，
读取路径能够衔接。历史计算已经产生的结果保留其来源记录；不要求重新执行整套科研计算。

## 长期维护的内容

| 内容 | 位置 | 保留原因 |
| --- | --- | --- |
| 当前计算入口、模型、求解器和测试 | `pipeline/`、`tests/` | 从输入生成结果的完整实现 |
| 原始预处理和模拟源码 | `metakaggle/`、`synthetic_data/` 中的源码 | 说明数据与实验如何构造 |
| 原始数据快照 | `metakaggle/__rawdata__/meta-kaggle.zip` | 外部来源输入；哈希记录在 `storage_layout.json`，不属于中间结果 |
| 冻结的科学输入 | `results/reestimation/lambda100_sd225/input_snapshots/` | 保留既有 397 项拟合输入引用及内容身份，避免依赖可变在线数据 |
| 正式结果及其依据 | `results/releases/paperjk9-20261029/` | 每竞赛最终结果、汇总、诊断、未解决项、配置和来源索引 |
| 论文直接使用的其他精简结果 | `results/` 下发布清单选中的 CSV/JSON/PDF | 当前论文的参数、回归、模拟和图表依据 |
| 论文源文件与正式图表、PDF | `paper/` | 对外呈现的成果 |
| 方法说明、历史审查与源码快照 | `docs/`，尤其 `docs/review/` | 保留研究过程的解释与代码来源，不放运行缓存 |
| 理论推导原文 | `archive/research/precision_conditions/*.txt` | 长期方法材料，不因扩展名为 TXT 就视为运行输出 |
| 已明确保留的旧发布记录 | `results/releases/paperjk7-review-20261016/` | 输入索引与历史发布溯源 |

`docs/PAPER_EVIDENCE.csv` 映射论文图表到证据和生产代码；
`results/releases/paperjk11-20261104/manifest.json` 是当前正式文件的有限清单。
正式证据中的后验接受记录、随机种子和抽样索引需要保留，不能因为文件小或由程序生成
就一概当作缓存。
理论推导和旧清理清单同样属于长期记录；它们与求解器日志、过程缓存分开分类。

## 全部中间结果统一放在 tmp/

| 目录 | 内容 |
| --- | --- |
| `tmp/runs/` | 当前维护入口产生的新拟合、预测、非线性设计搜索、精化和工作汇总 |
| `tmp/work/results/` | 其他实验程序的运行输出，沿用旧 results 内部层级以便衔接 |
| `tmp/inputs/` | 从冻结输入构造的工作副本，不是科学输入的唯一存档 |
| `tmp/cache/` | SQLite/逐点缓存、绘图缓存和 Python 字节码 |
| `tmp/build/` | 模型可执行文件、共享库、生成头文件及用于编译的源码副本 |
| `tmp/latex/` | LaTeX 辅助文件及构建期间的 PDF；完成后仅复制正式 PDF 到 paper |
| `tmp/history/` | 从旧目录迁来的运行记录、试验输出、候选链、旧逐点缓存和已被替代的结果 |
| `tmp/recovered/` | 考古恢复的抽样链、逐小时评分及其本地数据包 |
| `tmp/cache_archives/` | 此前已生成的缓存归档及清单，不继续新增压缩 |
| `tmp/checks/` | 精简导出、路径与构建检查的临时工作树 |

抽样链、逐小时预测、数值网格和搜索进度都是生成正式结果的中间数据。
长期保留其配置、来源、接受状态和正式结果；大体积运行材料统一放在 `tmp/`。
正式证据中的哈希和候选链索引仍可用于检查本地历史材料。
依赖历史链的命令需要相应本地文件；从冻结输入重新计算的入口不以这些旧链为初始化前提。

`tmp/` 整体被 Git 忽略。环境目录 `.venv*` 属于依赖环境；原始 ZIP 属于来源输入，
它们与科研中间结果分开分类。

## 路径迁移与代码衔接

- `docs/storage_layout.json` 逐项记录旧路径、新路径及用途；本次以重命名迁移，
  没有删除或压缩文件。
- 73 个正式后验记录保存在发布证据中，292 条链的相对路径已改为 `tmp/history/...`，
  链哈希和接受记录内容保持不变。
- `pipeline.storage` 统一定义运行、缓存、构建位置。旧记录中的路径由
  `historical_path` 按迁移清单定位，不改写历史记录正文。
- 历史生成代码的有用快照归入 `docs/review/source_snapshots/`，不会混在正式结果里。
  这些快照用于溯源；当前运行使用 `pipeline/` 中的维护入口及其适配器。
- 理论图使用的两份关键 CSV 已纳入正式证据的 `evidence/theory_source/`，
  绘图读取正式依据，重建图输出到 `tmp/runs/theory_figure/`。
- 科研运行不会自动覆盖正式发布证据；确定要更新论文结果时才显式更新发布快照与锁。

## 常用命令

```sh
make storage-audit       # 检查中间结果是否仍散落在受管理目录
make verify-release      # 正式文件和输入的内容校验
make verify-chains       # 有本地链时，核对 73 个正式记录及 292 条链的哈希
make reproduce-derived  # 从正式逐竞赛结果汇总论文表格
make paper              # 辅助产物留在 tmp/latex，正式 PDF 放在 paper/
python -m pipeline.jk9 prepare-inputs  # 从冻结输入构造 tmp/inputs/main/
```

新拟合默认进入 `tmp/runs/`；预测进入 `tmp/runs/prediction/`；
非线性设计进入 `tmp/runs/design/`。运行说明见 `REPRODUCIBILITY.md`。

## 2026-11-04 当前副本说明

本次追加迁移 499 处中间文件根目录，均为重命名，没有删除、压缩或移动 Git 已跟踪文件。
发布保护清单同时覆盖 JK11 当前清单与历史 JK9 科学证据；正文直接使用的 `.bbl` 保留在 paper。
历史迁移和恢复日志描述当时状态，不保证当前副本含有那些本地数据包。
292 条主设计历史链及预测恢复材料包当前未提供，归类为可再生中间材料，
不作为发布材料完整性失败；严格历史字节核验仍需原文件。
使用 `make verify-reproducibility` 检查保留输入、来源、代码和正式结果。
这不认证缺少记录的历史预测积分，也不把历史验收状态移植到新计算。
