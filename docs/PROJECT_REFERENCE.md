# porebin Project Reference

This document is the current source of truth for the repository.

It covers:

- the current recommended pipeline
- legacy paths that still exist in the codebase
- mathematical definitions and formulas
- input and output file formats
- CLI interfaces
- per-file and per-function inventory

The current codebase status described here matches `porebin==0.1.0`.

## 1. Current Project Status

### 1.1 Project positioning

`porebin` is a host-centric metagenome binning tool for Nanopore Pore-C data.

Its defining design choices are:

- preserve each multi-way Pore-C read as one hyperedge
- avoid clique expansion in the main hypergraph pipeline
- combine a contact hypergraph and a feature hypergraph for coarse clustering
- run a refine stage that performs conservative host-centric binning first, then residual relation mining for non-binned contigs

### 1.2 Current main path

The recommended path is:

1. `bam2contacts`
2. `build`
3. `cluster --method spectral`
4. `refine`
5. `export`

Or the end-to-end wrapper:

1. `run-bam`

In this path:

- BAM is converted to `contacts.parquet`
- `contacts.parquet` is the evidence-layer source of truth
- the canonical contact row logic lives in `porebin/contact_hypergraph.py`
- coarse clustering uses the joint contact-feature hypergraph spectral path
- refine uses the parquet-based conservative host-binning / residual-relation path

### 1.2.1 `seed` and `threads` in the current main path

For the current CLI/main path, the intended parameter semantics are:

- `seed` is a coarse-clustering control
- `threads` is a coarse-clustering / pairwise-baseline resource control

Current active behavior:

- `porebin cluster --seed` affects the spectral coarse path by controlling the random initialization used by the eigensolver
- `porebin cluster --threads` affects the spectral coarse path where HDBSCAN parallelism is available
- `porebin run-bam --seed` feeds the coarse clustering stage and the pairwise-baseline Leiden partition when that baseline mode is used
- `porebin run-bam --threads` feeds the coarse clustering stage and pairwise-baseline external sort / reduction path when that baseline mode is used
- the current parquet refine main path does not expose or consume dedicated `seed` / `threads` controls
- the current export path is effectively single-threaded and does not expose a dedicated `threads` control

Important distinction:

- the repository still contains older/auxiliary helpers where `seed` is used inside refine/splitting logic
- that does not change the current main-path CLI semantics described above

### 1.3 Legacy paths still present

The repository also contains older or auxiliary paths:

- PPL `.contacts` normalization path in `porebin/normalize.py`
- pairwise clique-expansion baseline in `porebin/pairwise_baseline.py`
- legacy refine path `refine_bins(...)` in `porebin/refine_legacy.py`
- legacy spectral bisection helpers in `porebin/cluster_legacy.py`
- simple FASTA exporter in `porebin/export_bins.py`

These are still part of the repository and are documented below, but they are not the primary path for the current hypergraph pipeline.

## 2. Architecture Overview

### 2.1 High-level pipeline

```text
name-sorted BAM
  -> porebin.bam_contacts.bam_to_contacts_parquet
  -> out/contacts/contacts.parquet
  -> out/coverage/coverage.tsv

contacts.parquet
  -> porebin.contact_hypergraph.iter_canonical_contact_rows
  -> canonical contact rows

canonical contact rows + contigs.fasta
  -> porebin.build_graph.build_graph
  -> graph audit/export artifacts

canonical contact rows + contigs.fasta (+ coverage.tsv)
  -> porebin.hypergraph_joint_spectral + porebin.cluster.cluster_spectral_hypergraph
  -> coarse bins.tsv

contacts.parquet + bins.tsv + contigs.fasta
  -> porebin.refine.refine_bins_parquet
  -> bins.refined.tsv + contig_host_scores.tsv + accessory_associations.tsv

contigs.fasta + bins.tsv
  -> porebin.export.export_bins
  -> FASTA export
```

### 2.2 Source-of-truth model

There are three distinct truth layers:

- evidence-layer truth:
  `contacts.parquet`
- contact-row canonicalization truth:
  `porebin/contact_hypergraph.py`
- operator/inference truth:
  `porebin/hypergraph_joint_spectral.py` and `porebin/refine.py`

`porebin/build_graph.py` is an audit/export layer for the contact hypergraph. It is not a separate mathematical definition of the contact hypergraph.

## 3. Mathematical Model

### 3.1 Evidence from BAM

For one alignment `a` inside one read `r`:

- `MAPQ(a)` is mapping quality
- `NM(a)` is edit distance if present
- `ell(a)` is aligned length

The per-alignment mapping correctness proxy is:

```math
p_{\mathrm{ok}}(a) = 1 - 10^{-MAPQ(a)/10}
```

Implementation convention:

- if `MAPQ == 255` or missing or invalid, use `p_ok(a) = 0.5`

If `NM` exists, define an identity proxy:

```math
id(a) = \max\left(0, 1 - \frac{NM(a)}{\ell(a)}\right)
```

Implementation convention:

- if `NM` is missing, use `id(a) = 1`

Per-alignment evidence:

```math
e(a) = p_{\mathrm{ok}}(a) \cdot id(a) \cdot \ell(a)
```

Aggregate alignment evidence from read `r` to contig `c`:

```math
E_{r,c} = \sum_{a: c(a)=c} e(a)
```

Normalize into soft incidence weights:

```math
\pi_{r,c} = \frac{E_{r,c}}{\sum_{c'} E_{r,c'}}
```

Important:

- `pi_{r,c}` is a normalized evidence share
- it is not a posterior probability

Read-level quality weight:

```math
q(r) = \frac{\sum_a p_{\mathrm{ok}}(a)\,id(a)\,\ell(a)}{\sum_a \ell(a)}
```

Quality control only:

```math
C(r) = \sum_c \pi_{r,c}^2
```

```math
k_{\mathrm{eff}}(r) = \frac{1}{C(r)}
```

### 3.2 Canonical contact row

The canonical contact row is defined after:

- removing empty contig names
- merging duplicate contigs by summing weights
- dropping non-positive contig weights
- renormalizing `pi`

The repository now distinguishes:

- `k_input`: input order recorded in the source row if present
- `k_valid`: effective order after canonicalization

```math
k_{\mathrm{valid}}(r) = \left|\{c : \pi_{r,c} > 0\}\right|
```

This is the order used by the current graph building and spectral path.

### 3.3 Contact hypergraph

Vertices are contigs.

Hyperedges are reads/contacts.

Soft incidence:

```math
H_c[v,e] = \pi_{e,v}
```

Spectral contact hyperedge weight:

```math
W_c(e) = \frac{q(e)}{k_{\mathrm{valid}}(e)-1}
```

Contact hyperedge degree:

```math
D_{e_c}(e) = \sum_v H_c[v,e]
```

Contact vertex degree:

```math
D_{v_c}(v) = \sum_e W_c(e)\,H_c[v,e]
```

The normalized contact operator is:

```math
\Theta_c =
D_{v_c}^{-1/2}
H_c
\operatorname{diag}(W_c)
\operatorname{diag}(D_{e_c}^{-1})
H_c^\top
D_{v_c}^{-1/2}
```

### 3.4 Audit/export view of the contact hypergraph

`build_graph.py` writes a bipartite edge list view:

```math
\mathrm{edge\_weight}(v,e) = \mathrm{OrderNorm}(k_{\mathrm{valid}}(e)) \cdot q(e) \cdot \pi_{e,v}
```

with:

```math
\mathrm{OrderNorm}_{\mathrm{pair}}(k) = \frac{2}{k(k-1)}
```

```math
\mathrm{OrderNorm}_{\mathrm{star}}(k) = \frac{1}{k-1}
```

This is an audit/export representation, not a separate mathematical truth.

### 3.5 Feature hypergraph

For each contig `v`, define a feature vector:

- canonical TNF136
- optional `log1p(coverage)`

After concatenation, features are z-scored per dimension:

```math
x_v = zscore([\mathrm{TNF136}(v), \log(1+\mathrm{cov}(v))])
```

Then build a kNN feature hypergraph:

- one hyperedge per contig
- `e_v = {v} union kNN(v)`

Incidence:

```math
H_f[u,e_v] = \mathbf{1}[u \in e_v]
```

Feature hyperedge weight:

```math
W_f(e) = 1
```

Operator:

```math
\Theta_f =
D_{v_f}^{-1/2}
H_f
\operatorname{diag}(W_f)
\operatorname{diag}(D_{e_f}^{-1})
H_f^\top
D_{v_f}^{-1/2}
```

### 3.6 Joint spectral coarse clustering

The joint operator is:

```math
\Theta_{\mathrm{joint}} = \lambda \Theta_c + (1-\lambda)\Theta_f
```

Implementation:

- compute the top `(d+1)` eigenvectors of `Theta_joint`
- drop the first trivial vector
- row-normalize the remaining embedding
- cluster the embedding using HDBSCAN
- `seed` controls the eigensolver initialization vector
- `threads` is passed to HDBSCAN when the implementation supports parallel core-distance computation

This produces coarse bins interpreted as candidate host communities for downstream anchor discovery.

### 3.7 Refine: conservative host binning and residual relation mining

The current refine path is `refine_bins_parquet(...)`.

Coarse bins are not treated as final truth. They are candidate host communities used for anchor discovery and later conservative binning.

Implementation note for the current main path:

- the active parquet refine CLI path is currently deterministic with respect to explicit user `seed` / `threads` controls
- `_split_bins_parquet(...)` exists in the repository as an auxiliary parquet split helper, but it is not currently wired into the main-path CLI refine command

#### Candidate host set

The current rule is:

```math
B = \{ b : \mathrm{bin\_weight}(b) > 0 \}
```

with the current piecewise status weights:

- `strong -> 1.0`
- `weak -> 0.35`
- `impure -> 0.0`

Only non-impure bins remain candidate hosts in the parquet refine path.

#### Anchor discovery

The existing pre-cleanup stage still computes:

- `anchor_mask[c]`
- `anchor_weight[c]`
- `bin_status[b]`
- `bin_weight[b]`

For contig `c`, using pre-refine support statistics:

```math
\mathrm{purity}(c) =
\frac{\max(0, \mathrm{intra}(c)-\mathrm{other}(c))}
{\mathrm{intra}(c)+\mathrm{other}(c)+\varepsilon}
```

```math
\mathrm{strength}(c) =
\min\left(1, \frac{\mathrm{intra}(c)}{\mathrm{median\_intra}(b(c))+\varepsilon}\right)
```

```math
a_c = \mathrm{clip}_{[0,1]}\left(\mathrm{purity}(c)\sqrt{\mathrm{strength}(c)}\right)
```

If the discrete `anchor_mask[c]` is true, the implementation enforces:

```math
a_c \leftarrow \max(a_c, 0.85)
```

The parquet refine path then derives a stricter `hard_anchor_mask`:

- `strong` bins: every anchor contig becomes a hard anchor
- `weak` bins: only the single best anchor becomes a hard anchor
- `impure` bins: no hard anchors

Hard anchors are the only contigs allowed to define read-level host direction during conservative binning.

#### Read qualification

For each canonical contact row, the current refine path first summarizes hard-anchor support by host. Reads are classified as:

- `informative`: one host direction is clearly dominant
- `anchor_sparse`: there is not enough hard-anchor support to define a direction
- `anchor_conflict`: there are hard anchors, but they do not support one clearly dominant host

The current implementation uses a simple dominance-ratio veto:

- `dominance_ratio = top1_anchor_mass / (top2_anchor_mass + eps)`
- `dominance_ratio >= 2.0` is required for an `informative` read

`anchor_sparse` reads do not vote.

`anchor_conflict` reads are excluded from conservative host binning, but their support can still accumulate into the residual relation head.

#### Conservative host-support update

The refine support map `support[c][b]` still accumulates contact-driven host support, but now the source of read-level host direction is restricted:

```math
\theta_{c,b} \mathrel{+}= q(r)\,\pi_{r,c}\,\max(0, A_r(b)-s^{\mathrm{hard}}_{r,c,b})
```

where `A_r(b)` is the hard-anchor host mass for read `r` and host `b`, and `s^{hard}_{r,c,b}` removes the target contig's own hard-anchor contribution when applicable.

Operationally, this means:

- only hard anchors define read-level host direction
- target contigs never define their own read-level host direction
- no informative hard anchors means the read does not support conservative binning

#### Weak coarse prior

The current weak coarse-label prior is still used, but only when the contig already has informative-read support or is itself a hard anchor:

```math
\alpha_c = \mathrm{prior\_strength} \cdot w_{b_0(c)} \cdot (\mathrm{sum\_support}(c)+\varepsilon)
```

with `prior_strength = 0.05`.

This keeps the coarse prior as a regularizer rather than allowing it to create an apparently interpretable host assignment from zero evidence.

#### Conservative binning states

For each contig, the parquet refine path computes normalized host support and uncertainty summaries:

- `top1_host`
- `top2_host`
- `margin`
- `entropy`
- `effective_hosts = exp(entropy)`

It then assigns one of the following refine states:

- `assigned_bin`
- `relation_only`
- `abstain_insufficient_information`
- `abstain_conflicting_evidence`
- `abstain_unreliable_background`

The current assignment policy is intentionally conservative:

- hard anchors in candidate hosts are directly retained in the final refined bins
- non-anchor contigs enter `bins.refined.tsv` only when the coarse host remains dominant, the score margin is strong enough, and there are enough informative reads
- contigs with residual multi-host structure but without safe bin membership remain `relation_only`
- low-information or unstable contigs enter one of the abstain states

#### Outputs

- `bins.refined.tsv` contains only `assigned_bin` contigs
- `contig_host_scores.tsv` contains the full refine-state summary for all contigs
- `accessory_associations.tsv` contains only `relation_only` contigs and does not modify the refined bins

### 3.8 Pairwise baseline

The pairwise baseline deliberately uses clique expansion for comparison with the hypergraph method.

For one read with order `k`:

```math
w_{\mathrm{pair}} = \frac{2}{k(k-1)} = \frac{1}{\binom{k}{2}}
```

Each unordered contig pair emitted by that read gets the same weight, and the sum of pair weights per read is `1`.

## 4. Inputs and Outputs

### 4.1 Main inputs

#### Contigs FASTA

- path: user-provided `contigs.fasta`
- used by:
  - BAM evidence validation
  - graph indexing
  - TNF feature construction
  - export

#### Name-sorted BAM

- required for `bam2contacts` and `run-bam`
- interpreted as one read name = one hyperedge

#### contacts.parquet

The current BAM-derived schema includes:

- `contact_id`
- `contigs`
- `contig_weights`
- `k`
- `k_eff`
- `weight`
- `support_count`
- `n_segments`
- `mapq_min`
- `p_ok_mean`
- `aligned_len_sum`
- `deoverlap_query_union_len_sum`
- `nm_sum`
- `mapq_missing_count`
- `nm_missing_count`
- `len_missing_count`

Important semantics:

- `contig_weights` are normalized evidence shares
- `weight` is read-level quality weight `q(r)`
- canonicalization may produce `k_valid != k`

#### coverage.tsv

- optional feature input for spectral coarse clustering
- optional signal used by pre-refine cleanup in parquet refine

### 4.2 Main outputs

#### BAM evidence outputs

- `out/contacts/contacts.parquet`
- `out/contacts/qc_bam2contacts.json`
- `out/coverage/coverage.tsv`

#### Graph audit outputs

- `out/graph/contig_index.tsv`
- `out/graph/contigs.tsv`
- `out/graph/edges.tsv`
- `out/graph/contacts_meta.tsv`
- `out/graph/graph_meta.json`

`contacts_meta.tsv` currently records:

- `contact_idx`
- `k_input`
- `k_valid`
- `weight`

#### Coarse clustering outputs

- `out/bins.tsv`
- `out/run.json`

#### Refine outputs

- `out/refined/bins.refined.tsv`
- `out/refined/contig_host_scores.tsv`
- `out/refined/accessory_associations.tsv`
- `out/refined/run_refine.json`

#### Export outputs

- `out/final_bins/bins_fasta/bin_<id>.fasta`
- `out/final_bins/unbinned.fasta`
- `out/final_bins/unbinned.tsv`
- `out/final_bins/bins.summary.tsv`
- `out/final_bins/export_meta.json`

## 5. CLI Reference

The CLI is implemented in `porebin/cli.py`.

### `porebin bam2contacts`

Function:

- `bam2contacts(...)`

Inputs:

- `--bam`
- `--contigs`
- `--out`
- `--parquet-batch-size`

Outputs:

- `contacts.parquet`
- `coverage.tsv`
- `qc_bam2contacts.json`
- `run.json`

### `porebin build`

Function:

- `build(...)`

Inputs:

- `--contigs`
- `--contacts`
- `--out`
- `--order-norm`
- `--parquet-batch-size`

Outputs:

- graph audit files under `out/graph`

### `porebin cluster`

Function:

- `cluster(...)`

Inputs:

- `--graph`
- `--out`
- `--method`
- `--seed`
- `--threads`
- `--bam` kept only for compatibility

Current active clustering method:

- `spectral`

### `porebin refine`

Function:

- `refine(...)`

Inputs:

- `--contigs`
- `--bins-tsv`
- `--contacts`
- `--coverage-tsv`
- `--out`

Current implementation path:

- `refine_bins_parquet(...)`

Current main-path control semantics:

- current CLI refine does not expose `seed` or `threads`
- the active parquet refine path does not currently consume those controls

### `porebin export`

Function:

- `export(...)`

Inputs:

- `--contigs`
- `--bins-tsv`
- `--out`

Current main-path control semantics:

- current CLI export does not expose `threads`
- the active export path is currently single-threaded

### `porebin run-bam`

Function:

- `run_bam(...)`

Default mode:

- `bam2contacts -> build -> cluster(spectral) -> refine -> export later by user`

Current main-path control semantics:

- `--seed` is a top-level reproducibility control for coarse clustering (and for baseline Leiden when `--pairwise-baseline` is used)
- `--threads` is a top-level resource control for coarse clustering (and for baseline sort/reduce when `--pairwise-baseline` is used)
- these top-level controls are not separate knobs for the current parquet refine stage

Baseline mode:

- `--pairwise-baseline`

## 6. Main Path vs Legacy Path

### 6.1 Current main path

Main current path:

- `bam_to_contacts_parquet(...)`
- `iter_canonical_contact_rows(...)`
- `build_graph(...)`
- `cluster_spectral_hypergraph(...)`
- `refine_bins_parquet(...)`
- `export_bins(...)`

### 6.2 Legacy path inventory

#### Legacy PPL normalization path

- `normalize_contacts(...)` in `porebin/normalize.py`

This converts an older PPL `.contacts` TSV layout into a simpler `contacts.parquet`.

#### Legacy refine path

- `refine_bins(...)` in `porebin/refine_legacy.py`

This older path is based on:

- decontam
- recruit
- split

and still keeps BAM/PPL-era logic. It is not the current CLI refine path.

#### Legacy spectral helpers

These historical bisection helpers now live in `porebin/cluster_legacy.py`.

`porebin/cluster.py` compatibility-imports them so older callers that still import from
`porebin.cluster` do not break during the refactor.

The helper set is:

- `_choose_k_by_eigengap`
- `_graph_bic_trigger`
- `_divisive_bisect_by_auto_bic`
- `_divisive_bisect_by_coverage_bic`
- `_spectral_sweep_bisect`
- `_sweep_best_conductance`
- `_coverage_bic_trigger`
- `_bic`
- `_loglik_1gauss`
- `_loglik_2gmm`
- `_log_norm_pdf`

The current CLI spectral path does not use recursive spectral bisection.

#### Pairwise baseline

This path is still active, but it is a comparison baseline rather than the main method.

#### Simple exporter

- `export_bins_fasta(...)` in `porebin/export_bins.py`

This is a smaller legacy/simple exporter. The main export path is `porebin/export.py`.

## 7. File-by-File Reference

### 7.1 Package root

#### `porebin/__init__.py`

Purpose:

- package version definition

Interfaces:

- `__version__`

### 7.2 Shared utilities

#### `porebin/utils.py`

Purpose:

- logging
- filesystem helpers
- run metadata
- FASTA iteration

Interfaces:

- `console`: Rich console used by CLI error output
- `setup_logging(verbose=False)`: configure logging
- `ensure_dir(path)`: create directory if needed
- `utc_now_iso()`: UTC timestamp helper
- `write_json(path, payload)`: JSON writer with directory creation
- `record_run(out_dir, command, params, seed=None)`: run.json context manager
- `dedupe_preserve_order(items)`: stable deduplication
- `iter_fasta_records(path)`: stream `(name, header, seq)`
- `iter_fasta_names(path)`: stream FASTA names only

### 7.3 Sequence constants

#### `porebin/tnf_constants.py`

Purpose:

- define canonical TNF136 order and helper mappings

Interfaces:

- `TNF136_LIST`
- `CANON_TO_IDX`
- `revcomp_4mer(s)`
- `canonical_4mer(s)`
- `enumerate_directed_4mers_lex()`
- `build_idx256_to_idx136()`

### 7.4 BAM evidence layer

#### `porebin/bam_contacts.py`

Purpose:

- convert queryname-sorted BAM into BAM-derived `contacts.parquet`
- compute `pi_{r,c}`, `q(r)`, QC fields, and coverage

Interfaces:

- `BamContactsError`
- `BamContactsStats`
- `bam_to_contacts_parquet(...)`

Important behavior:

- keep primary and supplementary alignments
- drop secondary alignments
- de-overlap within read+contig on query coordinates

### 7.5 Canonical contact rows

#### `porebin/contact_hypergraph.py`

Purpose:

- centralize canonical contact-row semantics

Interfaces:

- `ContactHypergraphError`
- `CanonicalContactRow`
- `canonicalize_contact_row(...)`
- `iter_canonical_contact_rows(...)`

Current role:

- source of truth for contact row cleaning before graph/inference consumption

### 7.6 Graph audit/export layer

#### `porebin/build_graph.py`

Purpose:

- write audit/export artifacts for the contact hypergraph

Interfaces:

- `GraphBuildError`
- `order_norm(k, method="pair")`
- `BuildStats`
- `build_graph(...)`
- `_k_summary(...)`
- `_order_norm_formula(...)`

Outputs:

- `contig_index.tsv`
- `contigs.tsv`
- `edges.tsv`
- `contacts_meta.tsv`
- `graph_meta.json`

### 7.7 Spectral operators and hypergraph math

#### `porebin/hypergraph_joint_spectral.py`

Purpose:

- build contact and feature hypergraph operators
- compute joint spectral embedding
- cluster embedding
- postprocess by contact-connected components

Interfaces:

- `JointSpectralError`
- `ContactComponentPostprocessMeta`
- `postprocess_labels_by_contact_components(...)`
- `load_contig_index(graph_dir)`
- `ContactIncidence`
- `build_contact_incidence_from_parquet(...)`
- `compute_tetranuc_features_from_fasta(...)`
- `compute_tnf136_features_from_fasta(...)`
- `_read_tnf136_meta(...)`
- `_write_tnf136_meta(...)`
- `load_or_build_tnf136_features(...)`
- `load_coverage_feature_optional(...)`
- `zscore_features(X)`
- `build_feature_knn_edges(X, knn_k)`
- `FeatureIncidence`
- `build_feature_incidence(neighbors, knn_k)`
- `ThetaOperator`
- `make_theta_operator(H_csr, W, De, Dv)`
- `_auto_d(V)`
- `spectral_embed_joint(...)`
- `hdbscan_cluster(Z, min_cluster_size, threads)`
- `write_bins_tsv(out_bins_tsv, idx_to_name, labels)`
- `json_dumps_small(obj)`

### 7.8 Coarse clustering orchestration

#### `porebin/cluster.py`

Purpose:

- orchestrate coarse clustering
- provide pairwise baseline clustering
- keep the active coarse-clustering entrypoints small
- compatibility-import legacy spectral helpers from `porebin/cluster_legacy.py`

Current active interfaces:

- `GraphClusterError`
- `cluster_spectral_hypergraph(...)`
- `cluster_leiden_pairwise(...)`

Metadata/input helpers:

- `_read_graph_meta(path)`
- `_read_contig_index(path)`
- `_read_edges(path, contig_offset)`
- `_read_incidence(path)`

Compatibility exports for legacy callers:

- `_choose_k_by_eigengap(...)`
- `_read_contig_lengths(...)`
- `_auto_min_contig_len(...)`
- `_coverage_from_bam(...)`
- `_coverage_from_tsv(...)`
- `_graph_bic_trigger(...)`
- `_divisive_bisect_by_auto_bic(...)`
- `_divisive_bisect_by_coverage_bic(...)`
- `_spectral_sweep_bisect(...)`
- `_sweep_best_conductance(...)`
- `_coverage_bic_trigger(values)`
- `_bic(xs, k)`
- `_loglik_1gauss(xs)`
- `_loglik_2gmm(xs)`
- `_log_norm_pdf(x, mu, var)`

#### `porebin/cluster_legacy.py`

Purpose:

- keep legacy spectral bisection logic out of the active coarse path
- preserve the historical helper implementations used by older paths

Interfaces:

- `_choose_k_by_eigengap(...)`
- `_read_contig_lengths(...)`
- `_auto_min_contig_len(...)`
- `_coverage_from_bam(...)`
- `_coverage_from_tsv(...)`
- `_graph_bic_trigger(...)`
- `_divisive_bisect_by_auto_bic(...)`
- `_divisive_bisect_by_coverage_bic(...)`
- `_spectral_sweep_bisect(...)`
- `_sweep_best_conductance(...)`
- `_coverage_bic_trigger(values)`
- `_bic(xs, k)`
- `_loglik_1gauss(xs)`
- `_loglik_2gmm(xs)`
- `_log_norm_pdf(x, mu, var)`

### 7.9 Refine module

#### `porebin/refine.py`

Purpose:

- conservative host-centric binning
- residual relation mining for non-binned contigs
- parquet refine main path
- keep shared refine helpers used by the parquet path
- compatibility-export the legacy `refine_bins(...)` entrypoint from `porebin/refine_legacy.py`

Core interfaces:

- `RefineError`
- `RefineStats`
- `PreCleanupResult`
- `ReadAnchorEvidence`
- `pre_refine_cleanup(...)`
- `_summarize_read_anchor_evidence(...)`
- `_entropy(probs)`
- `_effective_hosts(entropy_nats)`
- `_clip01(x)`
- `_bin_weight_from_status(status)`
- `_anchor_weight_from_support(...)`

Legacy refine entry:

- `refine_bins(...)`

Current refine entry:

- `refine_bins_parquet(...)`

Common helpers:

- `auto_min_contig_len(...)`
- `_read_contig_lengths(...)`
- `_read_bins_tsv(path)`
- `_bin_sizes(...)`
- `_bin_sizes_from_assignment(...)`
- `_drop_bins_below_min(...)`
- `_coverage_from_bam(...)`
- `_coverage_from_tsv(...)`
- `_bin_coverage_stats(...)`

Parquet support scan:

- `_iter_contacts_parquet(...)`
- `_scan_contacts_support_and_affinity_parquet(...)`

GMM / reassignment helpers:

- `_fit_2gmm_params(xs)`
- `_gmm_post_comp1(...)`
- `_reassign_or_unbin(...)`
- `_recruit_gmm(...)`

Parquet split path:

- `_split_bins_parquet(...)`

Note:

- `_split_bins_parquet(...)` is currently an auxiliary/helper path and is not invoked by the current CLI parquet refine command

Legacy refine helpers:

- `_decontam(...)`

Induced graph helper:

- `_InducedBuilder`
- `_next_bin_id(existing)`

Scoring / threshold helpers:

- `_top_candidates(...)`
- `_prune_topk(...)`
- `_auto_threshold_otsu(values)`
- `_otsu_threshold(values, bins=256)`
- `_median(values)`
- `_mad(values, center=None)`
- `_bic(xs, k)`
- `_loglik_1gauss(xs)`
- `_loglik_2gmm(xs)`
- `_log_norm_pdf(x, mu, var)`

#### `porebin/refine_legacy.py`

Purpose:

- keep the older BAM/PPL-era refine pipeline out of the active parquet refine file
- preserve the legacy `refine_bins(...)` implementation and its PPL-specific helpers

Interfaces:

- `refine_bins(...)`
- `_scan_contacts_support_and_affinity(...)`
- `_recruit(...)`
- `_split_bins(...)`

### 7.10 Pairwise baseline

#### `porebin/pairwise_baseline.py`

Purpose:

- build contig-contig baseline graph by clique expansion

Interfaces:

- `PairwiseBaselineError`
- `PairwiseBuildStats`
- `parse_contacts_stream(contacts_path)`
- `group_by_readid(stream, assume_sorted)`
- `expand_to_pairs(contigs)`
- `emit_raw_pairs_from_contacts(...)`
- `external_sort_and_reduce(...)`
- `build_pairwise_edges(...)`
- `_looks_like_header_row(row)`
- `_is_gnu_sort(sort_exe)`

Note:

- `cli.py` also imports `build_pairwise_edges_from_bam(...)`; if this symbol is absent or renamed, CLI and module must be reconciled.

### 7.11 Export layers

#### `porebin/export.py`

Purpose:

- main export policy
- apply minimum bin bp and short-contig policy

Interfaces:

- `MIN_BIN_BP`
- `ExportError`
- `ExportStats`
- `export_bins(...)`
- `_read_bins_tsv(path)`
- `_load_removed_by_refine(refine_dir)`
- `_unbinned_reason(...)`
- `_write_fasta_record(...)`
- `_natural_bin_sort_key(bin_id)`

#### `porebin/export_bins.py`

Purpose:

- simple lower-level FASTA exporter by bins.tsv

Interfaces:

- `ExportBinsError`
- `export_bins_fasta(...)`
- `_read_bins_tsv(path)`
- `_write_fasta_record(...)`

### 7.12 PPL normalization path

#### `porebin/normalize.py`

Purpose:

- normalize older PPL `.contacts` TSV into a basic `contacts.parquet`

Interfaces:

- `NormalizeError`
- `PplColumns`
- `normalize_contacts(...)`
- `_make_contacts_schema(pa, include)`
- `_open_parquet_writer(pq, path, schema, logger)`
- `_looks_like_header(row)`
- `_columns_from_header(header)`
- `_infer_columns_from_sample(sample_rows)`
- `_find_first(names, candidates)`
- `_norm_header_token(value)`
- `_parse_score_tags(score)`
- `_looks_like_score(value)`
- `_looks_like_contig(value)`
- `_looks_like_uuid(value)`
- `_looks_like_plain_contig(value)`
- `_norm_cell(value)`
- `_looks_numeric(value)`
- `_normalize_status(value)`

### 7.13 CLI

#### `porebin/cli.py`

Purpose:

- command-line entrypoints

Interfaces:

- `main(verbose=False)`
- `bam2contacts(...)`
- `build(...)`
- `cluster(...)`
- `export(...)`
- `refine(...)`
- `run_bam(...)`
- `_die(message, code=1)`

## 8. Documentation Files

### `README.md`

Role:

- repository landing page

### `docs/PROJECT_REFERENCE.md`

Role:

- authoritative architecture, math, and file/interface reference

### `docs/CONTEXT.md`

Role:

- short status summary and pointer to the authoritative reference

## 9. Test Inventory

### `tests/test_bam_contacts_deoverlap.py`

- checks BAM de-overlap logic and QC output

### `tests/test_bam_contacts_weight.py`

- checks `q(r)`, `k_eff`, MAPQ=255 handling, and no concentration penalty in BAM evidence

### `tests/test_bam_contacts_flags.py`

- checks supplementary kept, secondary dropped, unmapped skipped, and supplementary counter stability

### `tests/test_contact_hypergraph.py`

- checks canonical contact row behavior and `k_valid` consistency across build/spectral paths

### `tests/test_joint_operator_shapes.py`

- checks spectral operator dimensions and eigensolver behavior

### `tests/test_spectral_cluster.py`

- checks the spectral coarse path can separate two clusters

### `tests/test_many_bins_not_limited.py`

- checks HDBSCAN leaf mode does not artificially collapse many bins

### `tests/test_refine_mvp.py`

- checks parquet refine outputs, residual relation behavior, and weak-bin hard-anchor behavior

### `tests/test_pairwise_baseline.py`

- checks pairwise clique-expansion normalization and no self-loops

### `tests/test_tnf136_constants.py`

- checks TNF136 ordering and 256-to-136 mapping correctness

## 10. Current Documentation Guidance

When editing this repository, use the following policy:

- if README and code disagree, check this file and the code
- if this file and code disagree, the code is the final truth and this file should be updated
