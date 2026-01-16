# problems0114（2026-01-14）

> 目的：以“论文审稿人 + 代码 artifact”视角，对当前仓库的关键问题做客观、严格、可执行的记录，便于后续逐项修复与形成可复现的论文证据链。

## 0) 总体结论（当前状态）

| 结论 | 说明 |
|---|---|
| 暂不合格（不建议直接投审） | `multiplex` 主流程存在确定性运行错误；可复现性/文档/版本一致性不足；若不先修复工程与复现问题，审稿人会直接以“不可复现/不可运行”拒稿。 |

## 1) 问题清单（按优先级/严重程度）

| ID | 严重级别 | 模块 | 位置（文件:行） | 问题描述 | 审稿/复现影响 | 建议修复（最小闭环） |
|---:|---|---|---|---|---|---|
| P0-1 | 致命 | Multiplex 配置解析 | `hypergraph_binning/multiplex/pipeline.py:49` | `quality_cfg = cfg.get("quality", )` 默认返回 `None`，随后 `.get()` 会直接报错 | 示例 yml 直接跑崩；复现者/审稿人第一步就失败 | 改为 `cfg.get("quality", {})`，并补齐 `quality` 配置 schema（或显式禁用该功能开关） |
| P0-2 | 致命 | Multiplex 质量过滤 | `hypergraph_binning/multiplex/pipeline.py:205` | 调用 `hg.to_adjacency()`，但代码库 `Hypergraph` 未实现该方法 | “低置信度不强制分配”无法落地；宣称与实现不一致 | 实现 `to_adjacency()`（稀疏/采样版更佳）或移除该依赖并改为可用的 contact 统计接口 |
| P0-3 | 致命（潜在） | CLI 命令注册 | `hypergraph_binning/cli.py:39` 与 `:43` | `if __name__ == "__main__": app()` 在 `export-bins` command 定义之前（脚本执行时该命令不会注册） | 用户通过 `python -m ...` 运行时命令缺失；影响可用性与复现 | 将所有 `@app.command` 定义移动到 `app()` 调用之前，或移除 `__main__` 中的早调用 |
| P1-1 | 重大 | 可复现性（随机性） | `hypergraph_binning/multiplex/embedding.py:11`、`:34`；`hypergraph_binning/spectral/eigen.py:19`、`:14` | `seed` 参数未用于 `eigsh` 初始化（`v0=None`），同数据多次运行可能产生不同 embedding/聚类结果 | 论文复现性差；审稿人会要求固定随机性或报告方差 | 使用 `seed` 生成固定 `v0`；输出目录保存 `resolved_config`、随机种子、依赖版本、hash 等 |
| P1-2 | 重大 | 文档可用性 | `README.md:1` | README 为占位内容（`TEST、TEST`），缺少安装/用法/示例/复现说明 | 审稿人/复现者无法判断如何运行与复现实验 | 补齐 README：安装、输入格式、运行命令、输出解释、最小示例、CheckM2/AMBER 等评估流程 |
| P1-3 | 重大 | 方法核心权重“未定稿” | `hypergraph_binning/hypergraph/build.py:40`；`hypergraph_binning/hypergraph/stream.py:70` | 物理层权重 `w_e = q' * 2/(k-1)` 旁边留 `TODO` | “TODO” 属于审稿雷点；权重是方法核心，会被追问推导与合理性 | 代码与论文中给出明确推导/动机 + 消融：不归一化/`1/(k-1)`/`1/k` 等对 contamination/completeness 的影响 |
| P1-4 | 重大 | 参数/实现一致性 | `hypergraph_binning/io/bam.py:54`；`hypergraph_binning/pipeline/pipeline.py:95` | `read_coverage_min`、`q_cap` 在配置中出现但当前实现标注“未使用/仅兼容” | 审稿人会质疑“参数写了但不生效”或“方法描述不严谨” | 要么删掉（避免误导），要么实现并在文档说明其作用与默认值 |
| P2-1 | 中等（性能） | Quality 评估复杂度 | `hypergraph_binning/multiplex/confidence.py:146`、`:393`（`.toarray()`）；`:306`（`pdist`）；`:112`（`silhouette_samples`） | 多处潜在 O(N²) 或稠密化，contigs 多时会爆内存/时间 | 与“可扩展性”叙事冲突，复现者跑不动 | 改为稀疏/采样/近似指标（例如仅在邻域或 contact 子图上评估），并在文档标注适用规模 |
| P2-2 | 一般（工程一致性） | 版本一致性 | `pyproject.toml:7` vs `hypergraph_binning/__init__.py:1` | `version=0.1.2` 与 `__version__=0.1.0` 不一致 | 影响发布可信度与复现实验记录 | 统一版本号来源（建议以 `pyproject` 为准并自动同步） |
| P2-3 | 一般（叙述一致性） | 化学层构图叙述 | `hypergraph_binning/multiplex/features.py` | 当前实现为“每个节点一个 KNN 超边”的化学超图；若文档/论文宣称“mutual-KNN”则不一致 | 审稿人会抓“描述与实现不一致” | 明确论文叙述：是 KNN 超边还是 mutual-KNN（若需 mutual，补实现或更正文档） |

## 2) 审稿人高概率追问（需要准备的证据链）

| 追问点 | 审稿人关切 | 当前代码能否直接回答 | 需要补的证据/实验 |
|---|---|---|---|
| MAPQ→置信度映射是否合理 | 是否有理论/定义支持；阈值如何定 | 部分可（`p_i=1-10^{-MAPQ/10}` 实现存在） | 引用 MAPQ 定义；阈值敏感性（`mapq_min/segment_min_bases/min_segments_per_read`）；对假边率/边数/CheckM2 的影响曲线 |
| 物理层权重 `2/(k-1)` 的推导 | 为什么这样归一化；是否避免“稀释效应” | 目前不够（仍有 TODO） | 写清推导/直觉 + 消融对比；最好加入与 Hi-C/Pore-C 相关的参考或对照实验 |
| Auto‑β/Auto‑k 是否“挖掘结构” | 是否稳定、可复现、不会人为指定 | 有实现框架（`auto_tune.py`）但需证明 | 稳定性实验（多 seed/子采样/不同数据集）；与 CAMI/真实样本对比；报告方差与失败案例 |
| 低置信度 contig 不强制分配是否有效 | contamination 是否下降、代价多大 | 当前 pipeline 未跑通 | 修通后报告过滤前后 CheckM2、unassigned 比例、bin 内 contact/TNF 一致性变化 |

## 3) 建议修复顺序（最小可审稿闭环）

| 优先级 | 目标 | 完成标准（可验证） |
|---|---|---|
| P0 | 先“跑得通 + 功能不虚标” | `config.example.yml` 能完整跑完 `hgbin multiplex ...` 且不报错；质量过滤分支可用或默认关闭且文档一致 |
| P1 | 复现性与论文叙事补齐 | 同输入同 seed 得到稳定结果；输出目录记录 `resolved_config + 版本 + seed`；README 可让第三方复现 |
| P2 | 性能与可扩展性 | 大规模数据不稠密化；质量评估采用采样/稀疏策略；给出规模上限与复杂度说明 |

