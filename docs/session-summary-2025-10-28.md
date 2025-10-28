# 会话纪要（2025-10-28）

本纪要摘录本次关键决策、算法要点与运行要点，便于后续快速恢复与复现。

## 关键信息
- 目标：单源 Pore-C → 超图谱聚类 → MAG 分箱；导出 per-bin FASTA 供 CheckM2 评估
- 项目位置：`d:\hypergraphBinning`
- 主命令：
  - `hgbin pipeline config.yml`（自动选 k；流式超图）
  - `hgbin export-bins <contigs.fasta> <bins.tsv> <bin_fasta_dir>`（导出 bin FASTA）

## 数据与过滤参数（当前建议）
- BAM：建议 name-sorted（`samtools sort -n`）
- 过滤：MAPQ≥20–30；segment≥500–1000bp；min_segments_per_read=3；read_coverage_min=0.6；max_hyperedge_size=20

## 算法要点（超图谱）
- 超边：一条 Pore-C 读段→一条多接触超边 e，权重 `w_e = min(q_r, 0.95)/(k-1)`
- 拉普拉斯：`L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}`（矩阵自由算子；支持流式）
- 自动 k：最大谱隙 → ±2 邻域用 silhouette 微调；结果写入 `k_used.txt`

## 产物
- `bins.tsv`、`bin_sizes.tsv`、`hypergraph.stats.json`、`k_used.txt`
- `edges_chunks/edges_chunk_*.npz`（流式边块）
- 导出：`bin_fasta/*.fasta` + `bin_fasta_stats.tsv`

## 常见调参
- 内存紧：`streaming.enabled: true`；`edges_per_chunk: 100000`
- 噪声高：提高 `mapq_min`/`segment_min_bases`；`min_segments_per_read: 4`
- 超大超边：收紧 `max_hyperedge_size: 12–16`

## 下一步（可选）
- 接入 CheckM2 评估命令封装；
- 自动 k 可加 k_max 配置、抽样规模参数；
- 细化阶段（弱节点剥离、短 contig 后分配）。

> 如需更完整的上下文，请参考 `docs/chat-2025-10-28.md` 并粘贴完整对话。