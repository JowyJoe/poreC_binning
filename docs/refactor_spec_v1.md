# Refactor Spec v1

Status: draft

Scope of this document:

- Phase 0: current-state audit
- Phase 1: target architecture and I/O contracts
- Phase 1.5: spec hardening

This document is intentionally faithful to the current repository state. It separates:

- implemented
- partially implemented
- not implemented

It does not promote unimplemented design ideas to current behavior.

## 1. Current Architecture Summary

### 1.1 Current end-to-end flow

The current default hypergraph binning pipeline is:

`bam2contacts -> build -> cluster -> refine`

With downstream association handled separately as:

`refine -> associate`

Where:

- `bam2contacts` converts a queryname-sorted BAM into `contacts.parquet` plus `coverage.tsv`.
- `build` creates graph-side indices and metadata under `graph/`.
- `cluster` performs joint contact-feature hypergraph spectral clustering and writes coarse `bins.tsv`.
- `refine` currently runs `refine_bins_parquet(...)`, which owns post-binning refinement outputs:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
- `associate` consumes refined bins plus the residual pool and writes downstream `associations.tsv`.

`export` is a separate presentation layer and is not part of the main inference chain.

### 1.2 Current module map

Main-line files:

- `porebin/bam_contacts.py`
- `porebin/contact_hypergraph.py`
- `porebin/build_graph.py`
- `porebin/cluster.py`
- `porebin/hypergraph_joint_spectral.py`
- `porebin/refine.py`
- `porebin/associate.py`
- `porebin/export.py`
- `porebin/cli.py`

New but only partially integrated support files:

- `porebin/scg.py`
- `porebin/refine_qc.py`
- `porebin/refine_state.py`

Legacy / compatibility files:

- `porebin/refine_legacy.py`
- `porebin/cluster_legacy.py`
- `porebin/pairwise_baseline.py`

### 1.3 Current semantic interpretation of stages

#### bam2contacts

Implemented.

Purpose:

- turn read-level alignments into read-level hyperedges
- compute per-read soft contig evidence shares
- compute per-read quality weights
- export optional coverage proxy

Current outputs:

- `contacts/contacts.parquet`
- `coverage/coverage.tsv`
- `contacts/qc_bam2contacts.json`

#### coarse clustering

Implemented.

Purpose:

- generate coarse initial bins

Current method:

- contact hypergraph + feature hypergraph
- joint spectral embedding
- HDBSCAN
- contact-component postprocess

Current output:

- `bins.tsv`

#### refine

Implemented as a Phase-3 bin-centric orchestration skeleton, but not yet as the final post-binning algorithm.

Current user-facing purpose:

- load coarse bins and normalize refine input scope
- perform post-binning refinement
- export final refined bins as a first-class product
- export residual pool as a first-class product

Current internal support layers:

- QC snapshot generation
- minimal action-log generation
- a transitional per-contig host-support compatibility bridge

Current outputs:

Primary outputs:

- `bins.refined.tsv`
- `residual_pool.tsv`

QC / audit outputs:

- `bin_qc.refined.tsv`
- `refine_actions.tsv`
- `run_refine.json`

Compatibility / transition output:

- `contig_host_scores.tsv`

Current limitation:

- the active flow is now organized as a bin-centric skeleton, but candidate operations are still thin
- accepted operations currently include:
  - pre-cleanup decontam
  - one live split operation on preliminary final bins
  - one live reassign operation scoped to sibling bins created by an accepted split
  - one live recruit operation on `residual_unresolved` contigs
- merge is not yet active in orchestration
- the current compatibility bridge still derives from legacy contig-state logic internally
- the current live split implementation is:
  - SCG-triggered split admission
  - local 2-way reclustering on the coarse feature layer (TNF136 + optional coverage)
  - contact used as structural validation, not as the active split partition rule
- the current live reassign implementation is:
  - post-split only
  - limited to the sibling pair created by an accepted split
  - contact-support driven for `keep / move_to_sibling / residualize`
  - SCG used only as a non-worsening gate
- the current live recruit implementation is:
  - post-reassign only
  - limited to residual rows with `refined_status = residual_unresolved`
  - contact-affinity driven for `assign_to_bin / keep_residual`
  - SCG used only as a non-worsening gate

Current Phase-3 note:

- `accessory_associations.tsv` is no longer a primary refine output
- residual handoff is now explicit
- contig-level host scores still exist for compatibility while `associate` is being split out
- `refine_actions.tsv` now has a real code-level landing point
- old `relation_only / abstain_*` labels remain internal compatibility bridge semantics, not the public refine contract

#### associate

Implemented as a Phase-2 compatibility skeleton.

Current user-facing purpose:

- consume `bins.refined.tsv` plus `residual_pool.tsv`
- emit downstream `associations.tsv`
- keep final bins read-only

Current limitation:

- current implementation is intentionally thin
- it uses `contig_host_scores.tsv` as an optional compatibility bridge when available
- it does not yet define the final long-term association algorithm

#### SCG support

Implemented as a required refine-side QC / gate layer for the active refine mainline.

Purpose in current code:

- preflight SCG resources and required executables before refine runs
- auto-run ORF prediction + HMM search if built-in resources exist
- cache `scg_hits.tsv`
- compute per-bin SCG QC
- write SCG-aware bin QC snapshots and suspect-bin reports
- participate in suspect-bin detection
- participate in minimal action-level acceptance / veto for refine

Current outputs when enabled:

- `out/scg/proteins.faa`
- `out/scg/hits.domtblout`
- `out/scg/scg_hits.tsv`
- `bin_qc.coarse.tsv`
- `suspect_bins.coarse.tsv`
- `bin_qc.cleaned.tsv`
- `suspect_bins.cleaned.tsv`

Current limitation:

- SCG resource lookup is now manifest-driven
- the repository now contains `porebin/scg_db/manifest.json`
- the repository now contains a bundled HMM resource, `porebin/scg_db/core_bacterial_scg.hmm`
- real-world SCG behavior still depends on whether the bundled marker set is the right one for the dataset
- current SCG action gating is intentionally minimal:
  - decontam acceptance / veto
  - split acceptance / veto
  - reassign non-worsening veto / support
  - merge gating is not yet active in orchestration
- current refine mainline now requires SCG resources and toolchain up front; it fails fast instead of silently degrading
- current live split admission is SCG-led:
  - duplicated SCG / contamination-like signals trigger split-check
  - contact / coverage signals act as auxiliary evidence

#### export

Implemented.

Purpose:

- materialize FASTA files from an existing `bins.tsv`-style assignment
- apply presentation-time filters such as `MIN_BIN_BP`

This stage should not be treated as inference.

### 1.4 User-visible artifact tiers

The current user-facing artifact tiers are:

Primary products:

- coarse:
  - `bins.tsv`
- refine:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
- associate:
  - `associations.tsv`

QC / audit products:

- `bin_qc.refined.tsv`
- `refine_actions.tsv`
- `run_refine.json`
- transitional pre/post snapshots such as:
  - `bin_qc.coarse.tsv`
  - `bin_qc.cleaned.tsv`
  - `suspect_bins.*.tsv`

Compatibility / transition products:

- `contig_host_scores.tsv`

Interpretation rule:

- `contig_host_scores.tsv` must not be presented as a primary scientific result
- it is retained only as a compatibility / audit bridge during migration

## 2. Current Problems

### 2.1 Main architectural mismatch

The current project is still organized around:

- hypergraph coarse clustering
- conservative host-assignment inference
- residual/accessory relation mining

But the intended mainline should be:

- `coarse = initial bins`
- `refine = post-binning refinement`
- `associate = downstream relation mining`

This mismatch is the primary reason the current codebase feels conceptually mixed.

### 2.2 Refine is only partially bin-centric

Phase 3 improved the orchestration layer:

- `bins.refined.tsv`
- `residual_pool.tsv`
- `bin_qc.refined.tsv`
- `refine_actions.tsv`

are now explicit first-class refine outputs.

However, current `refine_bins_parquet(...)` still depends internally on:

- per-read anchor evidence
- per-contig host support
- compatibility contig-state labels:
  - `assigned_bin`
  - `relation_only`
  - `abstain_*`

So the public control flow is now closer to the desired bin-centric design, but the internal decision core still carries a legacy compatibility bridge.

Phase-5 interpretation:

- the legacy contig-state bridge is now treated explicitly as a compatibility layer
- it should not be read as the public refine method contract
- public refine ownership remains:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `bin_qc.refined.tsv`
  - `refine_actions.tsv`
  - `run_refine.json`

### 2.3 Relation mining is coupled to refine

Partially improved in Phase 2, but not fully resolved.

Effects:

- refine now emits an explicit `residual_pool.tsv`
- a new `associate.py` owns `associations.tsv`
- `contig_host_scores.tsv` still carries legacy relation-oriented semantics and remains a compatibility bridge, but it no longer owns the public relation-mining output

### 2.4 SCG is not yet part of the main refinement decision loop

Current SCG integration stops at:

- SCG hit generation
- per-bin QC snapshot
- suspect-bin reporting

SCG does not yet:

- drive refine state transitions
- gate split acceptance
- gate merge acceptance
- gate reassign / recruit acceptance
- define final bin QC outputs in the main refine result contract

### 2.5 Multiple refinement paradigms coexist

Current repository contains:

- `refine_bins_parquet(...)`: active parquet host-assignment path
- `refine_bins(...)` in `refine_legacy.py`: older BAM/PPL path
- helper functions in `refine.py` for split/reassign/recruit that are not wired into the active parquet mainline

This creates conceptual and maintenance ambiguity.

### 2.6 Export still performs meaningful filtering

`export.py` still applies:

- `MIN_BIN_BP`
- auto min-contig-length rules

Therefore:

- `bins.refined.tsv` is not the only effective determinant of final exported bins
- final FASTA output semantics are partly deferred to export

This is undesirable for a future bin-centric refine stage.

### 2.7 Build and cluster are partially decoupled

`build_graph.py` still writes `edges.tsv`, but current spectral v2 clustering re-reads:

- `contacts.parquet`
- FASTA-derived TNF
- optional `coverage.tsv`

This means the graph-building layer is partly indexing/metadata and partly a leftover representation layer.

## 3. Reusable Module Inventory

This section lists modules and helpers that are reusable under the new architecture.

### 3.1 Strongly reusable

#### Contact data and canonicalization

- `porebin/contact_hypergraph.py`
- `iter_canonical_contact_rows(...)`
- `canonicalize_contact_row(...)`

Why reusable:

- canonical contact semantics are sound
- shared by build, cluster, and refine-side scans

#### BAM to hyperedge conversion

- `porebin/bam_contacts.py`

Why reusable:

- already defines current project-wide semantics for:
  - `pi_{r,c}`
  - `q(r)`
  - `k_eff`
  - coverage proxy

#### Coarse spectral machinery

- `porebin/cluster.py`
- `porebin/hypergraph_joint_spectral.py`

Why reusable:

- active coarse mainline already exists
- current refactor goal does not require replacing coarse yet

#### Contact support scanner

- `_scan_contacts_support_and_affinity_parquet(...)` in `porebin/refine.py`

Why reusable:

- provides a general contact-derived support / affinity pass that can serve:
  - bin QC
  - suspect contig detection
  - candidate reassign / recruit generation

#### Decontam primitive

- `_decontam(...)` in `porebin/refine.py`

Why reusable:

- useful as a candidate generator / cleanup primitive
- already operates at bin level

#### Induced bipartite split backend

- `_InducedBuilder`
- `_split_bins_parquet(...)`

Why reusable:

- useful as a local partition backend
- should be reused after removing its current trigger semantics from the method core

#### Coverage utilities

- `_coverage_from_tsv(...)`
- `_coverage_from_bam(...)`
- `_bin_coverage_stats(...)`

Why reusable:

- still useful as optional QC and auxiliary evidence

#### SCG skeleton

- `porebin/scg.py`

Why reusable:

- correct architectural placement for built-in marker DB, cached hit generation, and per-bin QC
- needs repair and completion, not replacement

#### QC snapshot layer

- `porebin/refine_qc.py`
- `porebin/refine_state.py`

Why reusable:

- already provides a path toward bin-centric refinement bookkeeping

### 3.2 Reusable only after role change

- `_reassign_or_unbin(...)`
- `_recruit_gmm(...)`
- `_split_bins_parquet(...)` trigger policy
- legacy contig-state bridge helpers in `porebin/refine.py`

Why only conditionally reusable:

- current decision rules are heuristic-heavy
- they should become candidate generators or fallback tools, not main method logic
- Phase 3 now explicitly treats them as non-core helpers rather than active refine orchestration

### 3.3 Legacy or control paths

- `porebin/refine_legacy.py`
- `porebin/cluster_legacy.py`
- `porebin/pairwise_baseline.py`

Role:

- preserve old or baseline behavior
- should not define the new mainline architecture

## 4. Coupling Points That Must Be Split

### 4.1 Refine output coupling

Current refine owns both:

- final bin output
- residual/accessory association output

Must split into:

- refine owns final bins and residual pool
- associate owns relation outputs

### 4.2 Refine internal semantic coupling

Current refine mixes:

- cleanup
- candidate-host gating
- contig host assignment
- residual relation mining
- partial QC snapshot generation

These responsibilities need to be re-layered.

### 4.3 SCG-QC coupling gap

Current SCG reports are generated, but the main refine control flow does not consume them.

This is a coupling gap:

- SCG exists structurally
- SCG is not yet functionally connected to final refinement decisions

### 4.4 Export-policy coupling

Current final FASTA interpretation still depends on export-time thresholds.

Refine should eventually own the inferential decision about:

- final kept bins
- residual pool
- tiny bins

## 5. New Architecture

This section defines the target architecture for the refactor. It is a target specification, not a statement of current implementation.

### 5.1 Top-level modules

#### coarse

Role:

- generate initial bins from hypergraph contact structure and sequence-derived features

Output meaning:

- initial candidate bins only
- not yet final refined bins

#### refine

Role:

- perform post-binning refinement over coarse bins
- produce final bins
- produce residual pool for downstream association

Core responsibilities:

- bin QC snapshot
- suspect bin detection
- decontam
- local split
- contig reassign / recruit
- bin merge
- tiny-bin filtering
- final refined bin export
- residual pool export

#### associate

Role:

- perform downstream residual/accessory/MGE association mining
- consume final refined bins and residual contigs
- never modify final bins

## 6. Module Boundaries

### 6.1 coarse boundaries

coarse does:

- contact hypergraph construction from current pipeline inputs
- feature extraction for clustering
- joint spectral embedding
- HDBSCAN clustering
- output initial bins

coarse does not:

- use SCG
- decide final bin purity/completeness
- output residual association interpretation
- revise final refined bins

### 6.2 refine boundaries

refine does:

- load initial coarse bins
- compute bin-level QC
- apply post-binning operations
- output final bins and residual pool
- compute final bin-level QC summaries, including SCG-aware summaries when resources exist

refine does not:

- perform downstream host/MGE relation interpretation as a primary goal
- emit relation-mining results as part of its core output contract
- use SCG inside coarse clustering

### 6.3 associate boundaries

associate does:

- load final refined bins
- load residual pool
- load contacts and optional auxiliary evidence
- compute relation outputs

associate does not:

- change `bins.refined.tsv`
- reassign contigs into final bins
- act as a hidden second refinement pass

## 7. Refine I/O Contract

This is the target contract for the new refine stage.

### 7.1 Refine inputs

Required:

- contigs FASTA
- coarse `bins.tsv`
- `contacts.parquet`

Optional:

- `coverage.tsv`
- SCG resources under `porebin/scg_db/`

Refine may internally create SCG cache files, but should not require the user to supply `scg_hits.tsv` in the default path.

### 7.2 Refine outputs

Required primary outputs:

- `bins.refined.tsv`
- `bin_qc.tsv`
- `residual_pool.tsv`
- `run_refine.json`

Required audit outputs:

- operation logs for accepted or rejected edits
- suspect-bin report

Optional outputs:

- SCG cache and QC reports
- candidate-operation logs

### 7.3 Refine output semantics

- `bins.refined.tsv` is the authoritative final bin assignment output of the refinement stage.
- `residual_pool.tsv` contains contigs intentionally left outside final bins after refinement.
- Export should later materialize FASTA from `bins.refined.tsv` and residual outputs, but should not redefine refine semantics.

### 7.4 Field-level refine output schemas

This subsection hardens the refine contract to the field level. These schemas are target schemas for the refactored refine stage.

#### 7.4.1 `bins.refined.tsv`

Required columns:

| column | type | required from | meaning |
| --- | --- | --- | --- |
| `contig_name` | string | Phase 3 | FASTA contig identifier |
| `bin_id` | string | Phase 3 | final refined bin identifier |

Semantic rules:

- `bins.refined.tsv` contains only contigs retained in final refined bins.
- Residual or unbinned contigs are not written to this file.
- A contig must appear at most once in this file.
- For contigs in refine input scope, absence from `bins.refined.tsv` implies presence in `residual_pool.tsv`.
- Refine input scope is defined as all contigs present in the input FASTA. Contigs appearing in coarse `bins.tsv` but absent from the FASTA are out-of-scope data anomalies and should be recorded in `run_refine.json`, not in final outputs.

`bin_id` naming contract:

- `bin_id` is a string identifier.
- Existing coarse bin IDs may be preserved when a bin survives refinement unchanged.
- Newly created bins from split or merge operations may receive new refine-time IDs.
- New refine-time IDs should be unique within a run and deterministic given identical inputs and deterministic settings.
- Refine is not required to preserve coarse bin numbering when the bin topology changes.

#### 7.4.2 `residual_pool.tsv`

`residual_pool.tsv` is a required refine output and the formal handoff to the future `associate` stage.

Required columns:

| column | type | required from | meaning |
| --- | --- | --- | --- |
| `contig_name` | string | Phase 3 | FASTA contig identifier not retained in final bins |
| `reason` | enum string | Phase 3 | immediate reason why the contig is residual at the end of refine |
| `stage` | enum string | Phase 3 | refinement stage at which the contig most recently entered or remained in the residual pool |
| `coarse_bin_id` | string | Phase 3 | original coarse bin assignment if known, else `-1` |
| `refined_status` | enum string | Phase 3 | final refine-side classification of the residual contig |
| `note` | string | Phase 3 | optional operator/debug note, empty string allowed |

`reason` initial enum set:

- `coarse_unassigned`
- `short_contig`
- `tiny_bin`
- `low_contact_support`
- `contact_inconsistency`
- `coverage_inconsistency`
- `tnf_inconsistency`
- `scg_veto`
- `no_acceptable_target`
- `split_tiny_fragment`
- `final_qc_fail`
- `assignment_unresolved`
- `associate_candidate`
- `split_residual`
- `reassign_unresolved`
- `recruit_unresolved`

`stage` initial enum set:

- `input_filter`
- `refine_base`
- `decontam`
- `split`
- `reassign`
- `recruit`
- `merge`

`refined_status` initial enum set:

- `residual_unresolved`
- `residual_filtered`
- `residual_holdout`
- `residual_associate_candidate`

Representation rules:

- `coarse_bin_id` uses `-1` when no coarse assignment exists or the contig was absent from coarse bins.
- Empty string is not the preferred null representation for `coarse_bin_id`; use `-1` for consistency with current coarse semantics.
- `note` may be empty.
- A contig must appear at most once in `residual_pool.tsv`.
- A contig must not appear in both `bins.refined.tsv` and `residual_pool.tsv`.

Relationship to final bins:

- `bins.refined.tsv` and `residual_pool.tsv` are mutually exclusive by `contig_name`.
- `residual_associate_candidate` marks residual entries eligible for downstream `associate`.
- `residual_unresolved` and `residual_filtered` remain residual outputs but need not appear in `associations.tsv`.

#### 7.4.3 `bin_qc.refined.tsv`

`bin_qc.refined.tsv` is the formal bin-level QC table for the final refined bins.

Required columns:

| column | type | required from | population rule |
| --- | --- | --- | --- |
| `bin_id` | string | Phase 3 | always populated |
| `total_bp` | int | Phase 3 | always populated |
| `n_contigs` | int | Phase 3 | always populated |
| `contact_consistency` | float or `NA` | Phase 3 | populated when contact-derived QC is available |
| `coverage_dispersion` | float or `NA` | Phase 3 | populated when coverage is available, else `NA` |
| `tnf_dispersion` | float or `NA` | Phase 3 | column required in Phase 3; value may remain `NA` until TNF-based refine QC is wired |
| `unique_scg` | int or `NA` | Phase 4 | `NA` when SCG unavailable or not yet integrated |
| `duplicated_scg` | int or `NA` | Phase 4 | `NA` when SCG unavailable or not yet integrated |
| `completeness_like` | float or `NA` | Phase 4 | `NA` when expected markers unavailable |
| `contamination_like` | float or `NA` | Phase 4 | `NA` when expected markers unavailable |
| `suspect_flag` | `0` or `1` | Phase 3 | always present |
| `split_check_flag` | `0` or `1` | current implementation | whether the bin entered active split admission |
| `split_check_reasons` | string | current implementation | comma-joined split-check trigger and auxiliary reasons, else empty |
| `split_priority_source` | string | current implementation | split admission source such as `scg_priority`, else empty |

Recommended optional columns:

- `suspect_reasons`
- `scg_status`
- `coverage_status`
- `tnf_status`

Phase rules:

- Phase 3 must emit the full column set above, even if some values are `NA`.
- Phase 3 must populate at least:
  - `bin_id`
  - `total_bp`
  - `n_contigs`
  - `contact_consistency` if contacts exist
  - `suspect_flag`
- Phase 4 must populate SCG columns when SCG resources and hits are available.
- When SCG is unavailable, the SCG columns remain present but take `NA`.
- `suspect_flag` must degrade gracefully: when SCG is unavailable, it is computed from non-SCG QC only.
- Current Phase-4 implementation also emits `scg_status` with values:
  - `disabled`
  - `partial`
  - `enabled`

#### 7.4.4 `refine_actions.tsv`

`refine_actions.tsv` is the formal audit log of refinement decisions. It is not the primary scientific output, but it is a required traceability artifact.

Required columns:

| column | type | required from | meaning |
| --- | --- | --- | --- |
| `action_id` | string | Phase 3 | unique action identifier within a refine run |
| `action_type` | enum string | Phase 3 | operation type |
| `target_bin` | string | Phase 3 | target bin for the action, or `-1` if not applicable |
| `source_bin` | string | Phase 3 | source bin for the action, or `-1` if not applicable |
| `affected_contigs` | string | Phase 3 | JSON-encoded array of affected contig names, sorted lexicographically |
| `accepted` | `0` or `1` | Phase 3 | whether the action was accepted |
| `accept_reason` | string | Phase 3 | short controlled reason when accepted, else empty |
| `reject_reason` | string | Phase 3 | short controlled reason when rejected, else empty |
| `qc_before_ref` | string | Phase 3 | reference to the QC snapshot used for evaluation |
| `qc_after_ref` | string | Phase 3 | reference to the QC snapshot after application; empty if rejected or not recomputed |
| `scg_gate_used` | `0` or `1` | Phase 4 | whether SCG participated in evaluating this action |
| `scg_gate_result` | enum string | Phase 4 | SCG gate outcome for this action |
| `scg_gate_reason` | string | Phase 4 | compact reason for the SCG gate outcome |

`action_type` initial enum set:

- `decontam`
- `split`
- `reassign`
- `recruit`
- `merge`
- `tiny_bin_filter`
- `no_op`

Audit scope:

- This file is an audit/debug artifact and a formal traceability output.
- It should not be treated as the main scientific output of refine.
- `run_refine.json` stores run-level summaries; `refine_actions.tsv` stores per-action records.

Phase rules:

- Phase 3 may initially use coarse snapshot labels or simple round labels in `qc_before_ref` / `qc_after_ref`.
- Phase 3 may leave `qc_after_ref` empty for rejected or no-op proposals.
- Phase 3 may emit `[]` in `affected_contigs` for action candidates that are bin-wide and not yet resolved to an exact contig list at proposal time.
- Phase 4 may use SCG only as:
  - `support`
  - `neutral`
  - `veto`
  and must not turn SCG into a unified scoring function.

Current implementation note:

- the active parquet mainline currently emits:
  - `decontam`
  - `split`
  - `reassign`
  - `no_op`
- `reassign` currently emits one contig per action row

#### 7.4.5 `run_refine.json`

`run_refine.json` is the run-level metadata summary for refine.

It should contain:

- inputs
- resources and availability state
- schema version
- aggregate counts
- per-phase status flags
- output file paths
- summary QC counts
- SCG state summary
- compatibility/deprecation notes where relevant

It should not duplicate:

- the full per-action table from `refine_actions.tsv`
- the full per-contig membership of final bins
- the full residual pool rows

Boundary against `refine_actions.tsv`:

- `run_refine.json` stores summaries and references
- `refine_actions.tsv` stores detailed action rows

Boundary against QC TSVs:

- `run_refine.json` stores aggregate counts and file pointers
- `bin_qc.refined.tsv` stores per-bin QC details
- `residual_pool.tsv` stores per-contig residual disposition

## 8. Associate I/O Contract

This is the target contract for the new associate stage.

### 8.1 Associate inputs

Required:

- `bins.refined.tsv`
- `residual_pool.tsv`
- `contacts.parquet`

Optional:

- downstream MGE evidence
- external annotations

### 8.2 Associate outputs

Required:

- `associations.tsv` or a more specific association result set
- run metadata and audit JSON

### 8.3 Associate semantics

- associate consumes residual contigs and final bins
- associate produces relation interpretations only
- associate must be one-way with respect to refined bins

### 8.4 Field-level associate schema

Phase 2 introduced a compatibility `associate` skeleton. The schema remains fixed here to constrain further work.

#### 8.4.1 Associate inputs

Required:

- `bins.refined.tsv`
- `residual_pool.tsv`
- `contacts.parquet`

Optional:

- `coverage.tsv`
- SCG-derived QC summaries
- external annotations or metadata
- `contig_host_scores.tsv` as a Phase-2 compatibility bridge only

Input-side constraints:

- `associate` must treat `bins.refined.tsv` as read-only.
- `associate` should consume only residual rows that are eligible for downstream interpretation.
- The recommended default is to process only residual rows with `refined_status = residual_associate_candidate`.

#### 8.4.2 `associations.tsv`

Required columns:

| column | type | required from | meaning |
| --- | --- | --- | --- |
| `contig_name` | string | Phase 2 | residual contig being associated |
| `association_type` | enum string | Phase 2 | structural association class |
| `host_bin_ids` | string | Phase 2 | comma-separated refined bin IDs, sorted, one or more |
| `primary_host_bin_id` | string | Phase 2 | single best host bin ID, or empty if no primary host is defined |
| `support_strength` | float or `NA` | Phase 2 | continuous support score if defined by the chosen algorithm |
| `support_class` | enum string | Phase 2 | discretized support interpretation |
| `uncertainty` | float or `NA` | Phase 2 | continuous uncertainty measure, higher means more uncertain |
| `evidence_summary` | string | Phase 2 | compact description of evidence sources used |

`association_type` initial enum set:

- `single_host`
- `multi_host`
- `broad_host`

`support_class` initial enum set:

- `high`
- `moderate`
- `low`
- `provisional`

Output rules:

- `host_bin_ids` may contain multiple IDs.
- `primary_host_bin_id` must be one of `host_bin_ids` when non-empty.
- `support_strength` is a continuous value when the associate algorithm defines one; otherwise `NA` is permitted in early implementation.
- `uncertainty` should be continuous when defined; otherwise `NA` is permitted in early implementation.
- `associations.tsv` should contain only residual contigs with sufficient structural evidence.
- Residual contigs without enough structure stay in `residual_pool.tsv` only and do not need association rows.

One-way contract:

- `associate` must not write back into `bins.refined.tsv`.
- `associate` must not change residual membership retroactively.

## 9. SCG Boundaries

### 9.1 What SCG is allowed to do

SCG is allowed to support refine by:

- flagging suspect bins
- acting as an acceptance / veto signal for:
  - split
  - merge
  - reassign
  - recruit
- contributing final bin-level QC:
  - unique markers
  - duplicated markers
  - completeness-like
  - contamination-like

### 9.2 What SCG is not allowed to do

SCG must not:

- enter coarse embedding
- enter spectral operator construction
- enter HDBSCAN
- directly drive initial contig clustering
- act as a contig-level assignment engine

### 9.3 Current implementation status

Implemented:

- SCG resource resolution skeleton
- optional SCG hit generation
- SCG hit cache format
- per-bin SCG QC stats
- suspect-bin reporting

Partially implemented:

- SCG-aware reporting inside `run_refine.json`

Not implemented yet:

- robust marker resource package in repository
- SCG-driven acceptance / veto inside refine operations
- full integration of SCG into final refine outputs

## 10. Residual Pool Definition

Target definition:

- the residual pool is the set of contigs not retained in final refined bins after bin-centric refinement

This includes, depending on future implementation:

- unresolved contigs
- contigs rejected by decontam
- contigs from rejected or tiny split communities
- unassigned contigs not recruited back
- contigs intentionally preserved for downstream accessory/MGE association analysis

The residual pool must be explicit and materialized as a refine output, not inferred indirectly from relation-mining files.

### 10.1 Residual pool includes

The residual pool includes all FASTA contigs in refine input scope that are not retained in `bins.refined.tsv`, including:

- contigs that were unassigned at coarse import
- contigs released from tiny coarse bins during refine-side coarse import normalization
- contigs removed by decontam
- contigs released from split operations but not retained in accepted child bins
- contigs that fail reassignment
- contigs that fail recruitment into final bins
- contigs released by final tiny-bin filtering
- contigs that remain unresolved after all attempted refine operations
- contigs intentionally held out for downstream association analysis

### 10.2 Residual pool does not include

The residual pool does not include:

- contig names present in `bins.tsv` but absent from the input FASTA; these are invalid input anomalies and should be reported in `run_refine.json`
- output-layer-only artifacts from `export.py`; export should not define residual membership
- purely diagnostic entities such as QC-only pseudo-records or bin summaries

### 10.3 Residual pool and unresolved

`assignment_unresolved` is not a separate file category.

Rule:

- unresolved is a residual condition, expressed through:
  - `reason = assignment_unresolved` or an operation-native unresolved reason such as `split_residual`, `reassign_unresolved`, or `recruit_unresolved`
  - and/or `refined_status = residual_unresolved`

Associate processing rule:

- `associate` does not need to process all residual contigs.
- The expected default is:
  - residuals with enough structural support become `residual_associate_candidate`
  - residuals without enough signal remain unresolved residuals only

## 11. Invariants

The refactor should preserve the following invariants.

### 11.1 Mainline invariants

- coarse outputs initial bins only
- refine outputs final bins
- associate never modifies final bins

### 11.2 Data invariants

- contact semantics remain defined by `contacts.parquet` produced by `bam2contacts`
- `pi_{r,c}` remains a normalized evidence share, not a posterior probability
- `q(r)` remains a read-level quality weight

### 11.3 SCG invariants

- SCG remains optional at runtime when resources are unavailable
- SCG may strengthen or veto refinement operations, but does not become a replacement clustering engine

### 11.4 Traceability invariants

- each refactor phase must leave a runnable and inspectable intermediate state
- each major output must remain auditable through logs or TSV/JSON side products

## 12. Refine Operation Model

This section defines the internal conceptual model for the future bin-centric refine stage.

### 12.1 Unified refine flow

The refine stage should be organized as:

1. load coarse bins and normalize refine input scope
2. bin QC snapshot
3. suspect bin detection
4. candidate operation generation
5. candidate operation evaluation
6. acceptance / veto
7. operation application
8. post-operation QC refresh
9. final bin export
10. residual pool export
11. run summary export

### 12.2 Operation definitions

#### split

Concept:

- locally partition one suspect bin into two or more candidate sub-bins

Purpose:

- resolve mixed bins
- reduce contamination-like or structural inconsistency

Expected evidence:

- contact inconsistency
- optional coverage inconsistency
- optional TNF inconsistency
- optional SCG duplication or contamination-like signal

#### merge

Concept:

- combine two refined bins into one candidate bin

Purpose:

- recover bins that were over-split
- improve completeness without unacceptable contamination increase

Expected evidence:

- strong inter-bin structural support
- compatible QC before and after merge
- optional SCG complementarity without duplication worsening

#### reassign

Concept:

- move a contig from one existing bin to another existing bin

Purpose:

- correct local contamination or misplacement

Expected evidence:

- alternative-bin support stronger than current-bin support
- operation does not worsen bin-level QC after acceptance checks

Current implementation note:

- the active parquet mainline currently supports a narrow `reassign v1`
- `reassign v1`:
  - runs only after an accepted split
  - only considers the sibling pair created by that split
  - may keep the contig in the current child bin
  - may move the contig to the sibling child bin
  - may push the contig into the residual pool
- it does not yet perform global bin-to-bin reassignment

#### recruit

Concept:

- move a residual or currently unbinned contig into an existing bin

Purpose:

- recover missed contigs after decontam or coarse under-assignment

Expected evidence:

- sufficient support to a target bin
- operation does not worsen target-bin QC after acceptance checks

Current implementation note:

- the active parquet mainline now supports a conservative `recruit v1`
- `recruit v1`:
  - only consumes residual rows with `refined_status = residual_unresolved`
  - may assign the contig to any existing final bin
  - may keep the contig in the residual pool
  - does not create new bins
  - does not perform merge-like operations
  - uses contact support / affinity as the primary recovery signal
  - uses SCG only as a non-worsening gate

#### decontam

Concept:

- remove contigs from a bin into the residual pool

Purpose:

- improve bin purity and internal consistency

Expected evidence:

- weak within-bin support
- structural inconsistency
- optional coverage/TNF inconsistency
- optional SCG-related veto or support

#### tiny-bin filtering

Concept:

- dissolve bins that fail terminal size or terminal QC retention rules

Purpose:

- prevent tiny or unstable bins from being emitted as final bins

Result:

- affected contigs move to the residual pool

### 12.3 Required support by refactor phase

Phase 3 minimum required support:

- normalize refine input scope
- bin QC snapshot
- suspect bin detection
- decontam
- residual pool export
- final `bins.refined.tsv` export
- `refine_actions.tsv` skeleton

Phase 3 strongly preferred first-class support:

- local split skeleton and accepted/rejected action logging

Phase 3 allowed to remain candidate-only or disabled:

- merge
- reassign
- recruit

When disabled in Phase 3:

- these operations should be explicitly recorded as not attempted or not yet enabled through run metadata or action logging

Phase 4 expands:

- SCG-aware suspect detection
- SCG-aware acceptance / veto
- final SCG-aware QC outputs

Phase 5 expands:

- downgrade old heuristics into candidate generators or fallbacks

Phase 6 expands:

- unify CLI, module docstrings, and user-facing narrative so the public workflow is:
  - `cluster -> initial bins`
  - `refine -> final bins + residual_pool.tsv`
  - `associate -> downstream relation mining`

Operation Activation - Part 1 expands:

- make split the first live refinement operation
- run split on preliminary final bins rather than on coarse/cleaned bins directly
- require SCG resources for live split evaluation

### 12.4 Legacy heuristic role constraints

The following current helpers must not define the new method core:

- coverage 2-GMM trigger logic
- `_reassign_or_unbin(...)`
- `_recruit_gmm(...)`
- current `_split_bins_parquet(...)` trigger policy

Allowed future roles:

- candidate generator
- auxiliary evidence provider
- fallback path

Disallowed future role:

- sole or primary definition of refine acceptance logic

### 12.5 Legacy helper role table

This table constrains how legacy helpers may continue to exist during migration.

| function / helper | current role | target role | current call path | keep / wrap / deprecate / remove | migration note |
| --- | --- | --- | --- | --- | --- |
| `_build_legacy_compatibility_rows(...)` | transitional builder for legacy contig-state bridge rows | compatibility helper | directly called by `refine_bins_parquet(...)` before final bins / residual export | wrap | keep while `contig_host_scores.tsv` and residual compatibility mapping still exist |
| `_compat_bridge_row(...)` | row constructor for compatibility bridge | compatibility helper | called only from `_build_legacy_compatibility_rows(...)` | keep | no public semantics; internal audit structure only |
| `_legacy_residual_record(...)` | maps legacy contig-state labels into residual rows | compatibility helper | called only from `_derive_final_assignment_and_residuals(...)` | wrap | remove once residual pool is generated directly from operation-native refine outcomes |
| `_write_contig_host_scores_tsv(...)` | writes per-contig host-score compatibility table | deprecated compatibility artifact writer | directly called by `refine_bins_parquet(...)` | deprecate | retain only while `associate.py` may consume the compatibility bridge |
| `_reassign_or_unbin(...)` | LLR/GMM-based contig move-or-unbin helper | fallback helper | not called by active `refine_bins_parquet(...)`; historical local helper only | keep | may be wrapped later as optional fallback; must not define active refine semantics |
| `_recruit_gmm(...)` | LLR/GMM-based unbinned recruit helper | fallback helper | not called by active `refine_bins_parquet(...)`; historical local helper only | keep | may be wrapped later as optional fallback; must not define active refine semantics |
| `_split_bins_parquet(...)` | coverage-GMM-triggered split helper with induced bipartite backend | candidate generator | not called by active `refine_bins_parquet(...)`; historical local helper only | wrap | keep backend potential, but treat current trigger policy as deprecated |
| `relation_only / abstain_*` internal labels | transitional contig-state labels inherited from host-assignment refine | compatibility helper | produced inside `_build_legacy_compatibility_rows(...)` and consumed only by compatibility export / residual mapping | deprecate | must not appear as public refine-stage semantics |
| `refine.refine_bins(...)` | top-level legacy delegate entrypoint | deprecated compatibility entrypoint | delegates immediately to `porebin.refine_legacy.refine_bins(...)` | deprecate | keep only for backward-compatible CLI/API surfaces |
| `porebin/refine_legacy.py::refine_bins(...)` | historical BAM/PPL refine workflow | deprecated legacy module path | active only when caller explicitly enters legacy path | keep | retain for regression, backward compatibility, and baseline comparison; do not extend as mainline |

## 13. Compatibility and Migration Table

This table hardens output migration strategy for later phases.

| artifact | current role | future role | Phase 2 handling | Phase 3 handling | Phase 4 handling | decision |
| --- | --- | --- | --- | --- | --- | --- |
| `accessory_associations.tsv` | removed from active refine mainline in Phase 2 | deprecated alias at most; prefer `associate/associations.tsv` | stop writing it from refine; record deprecation in metadata | keep deprecated only if external wrappers still require it | remove completely unless compatibility pressure remains | deprecate -> remove |
| `contig_host_scores.tsv` | per-contig host-assignment audit table retained by refine | compatibility/debug bridge only; optional input to Phase-2 `associate` skeleton | keep as compatibility bridge for `associate.py` | keep only while bin-centric refine is not yet fully in place | deprecate from mainline docs once direct relation bridge is no longer needed | deprecate |
| coarse `bins.tsv` | initial bins from coarse clustering | remains initial-bin contract for refine input | keep | keep | keep | keep |
| `bins.refined.tsv` | currently core-like contigs from host-assignment refine | authoritative final refined bins | keep path and filename | redefine semantics to final binning result | keep as stable final-bin artifact | keep, semantics change |
| `bin_qc.coarse.tsv` | coarse QC snapshot side output | transitional QC audit file before new refine pipeline fully stabilizes | keep | keep as pre-refine audit | keep; may coexist with `bin_qc.refined.tsv` | keep |
| `bin_qc.cleaned.tsv` | post-cleanup QC snapshot side output | transitional audit file or migrate into round-based QC snapshots | keep | keep while new refine skeleton is introduced | may be renamed or folded into round snapshots later | wrap |
| `suspect_bins.coarse.tsv` / `suspect_bins.cleaned.tsv` | report-only suspect-bin side outputs | candidate input reports for refine operations | keep | keep and connect to refine orchestration | keep; SCG-aware semantics improved in Phase 4 | keep |
| old `run_refine.json` relation fields | replaced in Phase 2 by residual-handoff and compatibility bridge metadata | keep only residual-handoff summary in refine; relation-specific summary moves to associate metadata | record `residual_pool_role`, `associate_module_expected`, and deprecation notes | further reduce host-assignment compatibility detail once new refine core lands | keep SCG/resource and residual summaries only; no relation ownership in refine | move / deprecate |
| export outputs (`bins_fasta/*`, `unbinned.fasta`, `unbinned.tsv`) | materialization plus additional filtering | presentation layer over refine results and residual pool | unchanged | unchanged, but explicitly documented as presentation layer | unchanged until Phase 6 narrative alignment | keep, semantics clarified |

## 14. Refine vs Export Boundary

Current reality:

- `export.py` still applies `MIN_BIN_BP` and auto `min_contig_len` filtering
- therefore export currently influences the effective FASTA-level presentation of results

Target boundary:

- refine defines the formal final binning result
- export materializes and presents that result

Interpretation rules:

- `bins.refined.tsv` is the formal final result of refine
- `residual_pool.tsv` is the formal final residual result of refine
- export-side filtering is presentation-time behavior, not the authoritative scientific definition of bin membership

Phase 6 narrative target:

- either move all inferential final-keep rules into refine
- or keep export filters explicitly labeled as presentation-only options

Until Phase 6:

- docs and run metadata must avoid implying that export redefines refine semantics

## 15. SCG No-Resource Contract

This section constrains behavior when SCG resources are missing, incomplete, or only partially usable.

### 15.1 When SCG resources are missing

Examples:

- `manifest.json` missing
- manifest-declared HMM file missing
- `prodigal` missing
- `hmmsearch` missing

Required behavior:

- refine must fail fast before entering the main orchestration
- no partial refine run should proceed without SCG resources and toolchain
- `run_refine.json` is not expected when preflight fails before refine starts

Required run metadata fields:

- `scg_enabled`
- `scg_status`
- `scg_reason`
- `scg_marker_set_id`
- `scg_db_version`
- `scg_expected_markers_count`

Initial `scg_status` enum set:

- `disabled`
- `partial`
- `enabled`

### 15.2 When SCG resources exist but hits are incomplete

Examples:

- cache exists but expected marker metadata is incomplete
- marker search ran but produced only partial hits
- expected marker set is unknown while hit rows exist

Required behavior:

- refine still runs
- observed SCG counts may be reported
- `completeness_like` and `contamination_like` may be `NA` if expected marker count is unknown
- `run_refine.json` must record partial-hit status

Output rules:

- `unique_scg` and `duplicated_scg` may be populated from observed hits
- `completeness_like` and `contamination_like` require known `expected_markers`; otherwise `NA`
- acceptance gates that require expected-marker normalization must degrade to non-SCG or observed-hit-only logic, depending on implementation phase

### 15.3 Phase constraints for SCG integration

Current implementation status:
- active:
  - refine-time SCG resource/tool preflight
  - suspect-bin enhancement
  - decontam gate
  - live split gate and live split application
- not yet active:
  - merge gate in live orchestration
  - reassign / recruit gate in live orchestration
  - SCG-driven candidate generation beyond split admission

## 16. Phase 2-6 Execution Plan

### Phase 2

Goal:

- physically split relation-mining responsibilities into `associate`

Expected code motion:

- identify relation-only/accessory output logic in `refine.py`
- create `associate.py` or equivalent module
- keep compatibility wrappers only where necessary

### Phase 3

Goal:

- replace the current refine mainline with a bin-centric refinement pipeline skeleton

Expected focus:

- orchestration first
- operation stages next
- final bins and residual pool become the primary refine products

### Phase 4

Goal:

- fully connect SCG into bin-level QC and operation acceptance

Expected focus:

- SCG resource loading
- cache reuse correctness
- expected-marker handling
- suspect-bin decision integration

### Phase 5

Goal:

- downgrade old heuristics into secondary roles

Expected focus:

- classify old helpers as:
  - keep
  - wrap
  - deprecate
  - remove later

### Phase 6

Goal:

- align CLI, docs, naming, and user-facing narrative with the new three-module architecture

Expected focus:

- CLI help text
- run metadata
- module naming
- docstrings and user-facing documentation
