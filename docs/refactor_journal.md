# Refactor Journal

This journal records actual refactor work performed in sequence. It is intended to make the process:

- phased
- traceable
- interruptible
- resumable

## Phase 0: Repository Re-audit

Status: completed

### Goal

- confirm where current coarse / refine / SCG / legacy / export / relation outputs live
- map the current call chain and data flow
- identify current mainline modules, legacy modules, and reusable candidates

### What was done

- re-audited current CLI entrypoints in `porebin/cli.py`
- re-audited coarse mainline in:
  - `porebin/cluster.py`
  - `porebin/hypergraph_joint_spectral.py`
- re-audited contact and build layers in:
  - `porebin/bam_contacts.py`
  - `porebin/contact_hypergraph.py`
  - `porebin/build_graph.py`
- re-audited current refine mainline and auxiliary helpers in:
  - `porebin/refine.py`
  - `porebin/refine_legacy.py`
- re-audited SCG support layer in:
  - `porebin/scg.py`
  - `porebin/refine_qc.py`
  - `porebin/refine_state.py`
- re-audited export layer in `porebin/export.py`
- summarized current architecture, problems, reusable components, and coupling points in `docs/refactor_spec_v1.md`

### Files changed

- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### Not done in this phase

- no algorithm behavior changes
- no function movement
- no CLI changes
- no SCG resource installation

### Current risks

- current refine narrative in code and docs still reflects host-assignment centric semantics
- repository still contains mixed active / legacy / helper refinement logic
- existing docs outside this journal/spec may still describe the old mainline

### Next step

- Phase 1: define new architecture, module boundaries, I/O contracts, SCG boundaries, and invariants

## Phase 1: New Architecture and I/O Contracts

Status: completed

### Goal

- define the new three-module architecture:
  - coarse
  - refine
  - associate
- define refine and associate boundaries
- define SCG boundaries
- define target I/O contracts and invariants

### What was done

- added the following sections to `docs/refactor_spec_v1.md`:
  - `New Architecture`
  - `Module Boundaries`
  - `Refine I/O Contract`
  - `Associate I/O Contract`
  - `SCG Boundaries`
  - `Residual Pool Definition`
  - `Invariants`
  - `Phase 2-6 Execution Plan`
- explicitly recorded that:
  - current refine is not yet bin-centric
  - current associate module does not yet exist
  - current SCG integration is partial and optional
  - current export still performs final-size filtering

### Files changed

- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### Not done in this phase

- did not create `associate.py`
- did not rewire `refine.py`
- did not alter any current outputs
- did not change tests

### Current risks

- there is now a deliberate gap between current implementation and target architecture
- future phases must preserve runnability while migrating responsibilities
- special care will be needed to keep backward-compatible outputs during transition

### Next step

- Phase 2: physically split relation-mining responsibilities out of `refine.py` into an `associate` module skeleton

## Phase 1.5: Spec Hardening

Status: completed

### Goal

- harden the Phase 0/1 architecture into an implementation-constraining spec
- define field-level schemas for refine and associate outputs
- define the residual pool as an engineering object
- define the refine internal operation model
- define compatibility and migration handling for old outputs
- define the SCG no-resource runtime contract

### What was done

- expanded `docs/refactor_spec_v1.md` with field-level refine output schemas for:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `bin_qc.refined.tsv`
  - `refine_actions.tsv`
  - `run_refine.json`
- expanded `docs/refactor_spec_v1.md` with field-level associate schema for:
  - `associations.tsv`
- hardened residual-pool semantics by defining:
  - included sources
  - excluded objects
  - relationship to unresolved residuals
- added `Refine Operation Model`, including:
  - unified refine flow
  - conceptual definitions of split / merge / reassign / recruit / decontam / tiny-bin filtering
  - phase-based minimum support expectations
  - constraints on legacy heuristic roles
- added a compatibility and migration table covering current outputs and future handling
- added an explicit `Refine vs Export Boundary` section
- added an explicit `SCG No-Resource Contract`

### Files changed

- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### Not done in this phase

- did not create `associate.py`
- did not modify refine orchestration
- did not move any relation-mining code
- did not alter coarse logic
- did not alter SCG runtime behavior
- did not modify tests

### Current risks

- the spec is now stronger than the implementation, by design
- Phase 2 and Phase 3 must translate the new contracts without breaking current runnable paths too abruptly
- some target metrics in `bin_qc.refined.tsv`, especially `tnf_dispersion` and SCG-normalized fields, still require later implementation work

### What is now sufficiently constrained for Phase 2

- refine / associate module boundaries
- refine output ownership versus associate output ownership
- residual-pool contract
- field-level target outputs for refine and associate
- compatibility handling expectations for current relation outputs

### What still needs later refinement

Phase 3:

- exact refine orchestration and round structure
- concrete residual-pool population logic in code
- concrete `refine_actions.tsv` emission points

Phase 4:

- SCG resource-state metadata in `run_refine.json`
- SCG-driven suspect detection and acceptance / veto logic
- interpretation of partial-hit SCG states

### Next step

- Phase 2: split relation-mining responsibilities into an `associate` module skeleton while preserving compatibility outputs

## Planned Execution for Phase 2-6

## Phase 2: Physically Split Out Associate Skeleton

Status: completed

### Goal

- split downstream relation-mining responsibility out of `porebin/refine.py`
- create a standalone `porebin/associate.py` skeleton
- make `residual_pool.tsv` an explicit refine output
- establish one-way handoff:
  - `refine -> bins.refined.tsv + residual_pool.tsv`
  - `associate -> associations.tsv`

### What was done

#### Step 1: identified current relation-mining responsibility inside `refine.py`

Concrete ownership split identified in the active parquet refine path:

- `refine`-owned code still centered on:
  - coarse-bin loading and cleanup
  - anchor / hard-anchor derivation
  - contact-driven host-support accumulation
  - final core-like bin writing to `bins.refined.tsv`
- downstream relation-coupled code was concentrated in the output/classification block of `refine_bins_parquet(...)`:
  - writing `contig_host_scores.tsv`
  - classifying contigs into:
    - `assigned_bin`
    - `relation_only`
    - `abstain_*`
  - writing `accessory_associations.tsv`
- major coupling points identified:
  - non-core contigs were previously exported only through relation-oriented outputs
  - residual membership had to be inferred indirectly from `relation_only` / `abstain`
  - relation output ownership lived inside refine instead of a downstream module

#### Step 2: created `porebin/associate.py`

- added `associate_residuals(...)` as the new module entrypoint
- fixed its inputs to:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `contacts.parquet`
  - optional `contig_host_scores.tsv` compatibility bridge
- fixed its output to:
  - `associations.tsv`
- explicitly constrained the module to be one-way:
  - it consumes final bins
  - it does not rewrite `bins.refined.tsv`
  - it does not change final assignments

Current Phase-2 implementation choice:

- `associate.py` is intentionally thin
- it reuses `contig_host_scores.tsv` only as an optional compatibility bridge
- if the compatibility bridge is absent, it emits an empty but schema-correct `associations.tsv`

#### Step 3: removed relation output ownership from `refine.py`

- `refine_bins_parquet(...)` no longer writes `accessory_associations.tsv`
- `refine` now writes:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `contig_host_scores.tsv` as a compatibility/audit bridge
  - refined QC snapshot outputs
- `run_refine.json` now records:
  - `residual_pool_role = explicit_handoff_to_associate`
  - `associate_module_expected = true`
  - compatibility/deprecation notes for legacy relation outputs

#### Step 4: made `residual_pool.tsv` explicit

- added explicit residual export in `refine.py`
- current row schema:
  - `contig_name`
  - `reason`
  - `stage`
  - `coarse_bin_id`
  - `refined_status`
  - `note`
- current Phase-2 population is a compatibility mapping from legacy refine states:
  - `relation_only -> residual_associate_candidate`
  - selected `abstain_*` states -> `residual_unresolved` or `residual_filtered`
- this is not yet the final bin-centric residual logic planned for Phase 3, but it establishes the explicit handoff contract

#### Step 5: minimal CLI alignment

- added an `associate` subcommand in `porebin/cli.py`
- updated `refine` command comments/help to describe:
  - `residual_pool.tsv` as the downstream handoff
  - `contig_host_scores.tsv` as compatibility/audit only

### Files changed

- `porebin/associate.py`
- `porebin/refine.py`
- `porebin/cli.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not rewrite refine into the full bin-centric operation model
- did not implement the final association algorithm
- did not deeply modify `scg.py`
- did not modify coarse math core
- did not move legacy split / recruit / reassign heuristics into their Phase-5 roles yet

### Current risks

- `contig_host_scores.tsv` still carries host-assignment-centric semantics and remains a temporary coupling bridge
- `residual_pool.tsv` population is still compatibility-derived from legacy refine states, not yet generated by the future bin-centric operation model
- `associate.py` currently depends on optional compatibility input for non-empty output
- `refine.py` still internally thinks in terms of `relation_only` / `abstain_*`, even though those labels no longer define the public module boundary

### Next step

- Phase 3: restructure `refine.py` into a bin-centric refinement framework
- first focus:
  - make `bins.refined.tsv` and `residual_pool.tsv` the real internal products
  - introduce explicit refine-stage operation flow
  - reduce reliance on legacy contig-state labels as internal control logic

## Phase 3: Bin-centric Refine Skeleton

Status: completed

### Goal

- change the refine main control flow from contig host-assignment centric to a bin-centric orchestration skeleton
- make these refine outputs the real first-class products:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `bin_qc.refined.tsv`
  - `refine_actions.tsv`
  - `run_refine.json`
- keep legacy contig-state outputs only as compatibility bridges

### What was done

#### Step 1: established a bin-centric refine orchestration skeleton

`refine_bins_parquet(...)` is still built on the existing conservative evidence machinery, but its top-level orchestration is now explicitly structured around:

1. load coarse bins / normalize refine scope
2. coarse QC snapshot
3. pre-cleanup candidate generation and application
4. cleaned QC snapshot and suspect-bin detection
5. compatibility bridge generation
6. final bins export
7. residual pool export
8. refine action log export
9. refined QC snapshot
10. run summary export

This is now recorded in `run_refine.json` via:

- `refine_orchestration_model = bin_centric_skeleton`
- `pipeline_stages = [...]`

#### Step 2: made final bins and residual pool the internal primary products

- `refine.py` no longer writes `bins.refined.tsv` and `residual_pool.tsv` as incidental side effects inside the compatibility loop
- instead, it now:
  - first builds a compatibility bridge table in memory
  - then derives:
    - final refined bin assignment
    - explicit residual rows
  - then writes:
    - `bins.refined.tsv`
    - `residual_pool.tsv`
    - `contig_host_scores.tsv` only as a compatibility/audit bridge

This does not yet remove the legacy internal semantics, but it does move the public refine products to the center of the control flow.

#### Step 3: added real `refine_actions.tsv` output

- introduced `RefineActionRecord` in `porebin/refine_state.py`
- started writing `refine_actions.tsv` from `refine.py`
- current Phase-3 action log records:
  - accepted `decontam` actions when pre-cleanup removes contigs
  - rejected `split` candidates for suspect bins as placeholder candidate-only records
  - a single accepted `no_op` action when no candidates exist

This is intentionally minimal, but it is now a real artifact rather than a spec-only placeholder.

#### Step 4: downgraded old heuristics and old contig states in the orchestration layer

Current roles after Phase 3:

- `_split_bins_parquet(...)`
  - candidate generator only
  - not in active orchestration
- `_reassign_or_unbin(...)`
  - compatibility helper only
  - not in active orchestration
- `_recruit_gmm(...)`
  - compatibility helper only
  - not in active orchestration
- `relation_only / abstain_*`
  - compatibility bridge semantics for:
    - `contig_host_scores.tsv`
    - Phase-3 residual mapping
  - no longer treated as the public refine contract

These downgraded roles are now also recorded in `run_refine.json`.

#### Step 5: let `refine_state.py` and `refine_qc.py` carry more of the skeleton

- `porebin/refine_state.py`
  - now contains:
    - `ResidualRecord`
    - `RefineActionRecord`
- `porebin/refine_qc.py`
  - now carries richer QC row fields aligned with the Phase-3 schema direction:
    - `contact_consistency`
    - `coverage_dispersion`
    - `tnf_dispersion`
    - `suspect_flag`
    - `suspect_reasons`
  - QC TSV writing now emits these fields directly

### Files changed

- `porebin/refine.py`
- `porebin/refine_state.py`
- `porebin/refine_qc.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not deeply modify `scg.py`
- did not integrate SCG into acceptance / veto logic yet
- did not promote `_split_bins_parquet`, `_reassign_or_unbin`, or `_recruit_gmm` into the active bin-centric operation loop
- did not enhance `associate.py`
- did not change coarse math core

### Current risks / blockers

- the internal decision core still depends on legacy compatibility labels derived from contig-level host evidence
- residual generation is still compatibility-derived, not yet produced by a full operation-native refine engine
- `split`, `reassign`, `recruit`, and `merge` are not yet first-class active operations in the new orchestration
- SCG is still only QC-side and not yet part of action acceptance / veto

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py`
- result:
  - `5 passed`

### Next step

- Phase 4: integrate SCG as bin-level QC / suspect / acceptance-gate signal
- first-touch files should be:
  - `porebin/scg.py`
  - `porebin/refine_qc.py`
  - `porebin/refine.py`

## Phase 4: SCG as Bin-level QC / Acceptance Gate

Status: completed

### Goal

- move SCG from an isolated QC branch into the active refine quality-supervision layer
- keep SCG out of coarse clustering
- use SCG only for:
  - suspect-bin enhancement
  - action-level acceptance / veto
  - final QC consistency output

### What was done

#### Step 1: stabilized SCG resource / cache / return contracts

- updated `porebin/scg.py` so `ensure_scg_hits(...)` now returns an explicit SCG state:
  - `disabled`
  - `partial`
  - `enabled`
- fixed the cache-reuse contract so cached runs no longer lose `expected_markers` when resources are available
- defined partial-cache behavior:
  - cached hits without available resource metadata now return `state = partial`
  - refine can still use observed hit counts while normalized completeness/contamination may remain unavailable
- kept no-resource behavior non-fatal:
  - refine still runs
  - SCG fields become `NA`
  - action gating falls back to non-SCG behavior

#### Step 2: connected SCG to suspect-bin detection

- `porebin/refine_qc.py` now carries explicit SCG-aware QC fields:
  - `scg_status`
  - `unique_scg`
  - `duplicated_scg`
  - `completeness_like`
  - `contamination_like`
  - `suspect_flag`
  - `suspect_reasons`
- suspect detection now combines:
  - structural reasons
  - SCG reasons
- when SCG is unavailable:
  - suspect detection degrades to non-SCG structural rules only

#### Step 3: connected SCG to action acceptance / veto

Activated in Phase 4:

- `decontam`
  - SCG now participates as a real gate
  - decontam can be vetoed if SCG worsens in the protected direction
  - vetoed removals are restored before downstream refine stages continue
- `split`
  - SCG now participates as a placeholder support/neutral signal for suspect split candidates
  - split is still not actively applied in the Phase-4 orchestration

Not yet activated:

- merge gate in live orchestration
- reassign / recruit SCG gate in live orchestration

#### Step 4: made refined QC the formal SCG carrier

- `bin_qc.*.tsv` now expresses SCG status explicitly:
  - `disabled`
  - `partial`
  - `enabled`
- when SCG is disabled:
  - `unique_scg`
  - `duplicated_scg`
  - `completeness_like`
  - `contamination_like`
  are emitted as `NA`
- when SCG is partial:
  - observed counts may be present
  - normalized values may still be `NA`

#### Step 5: wrote SCG gate metadata into action log and run summary

- `refine_actions.tsv` now includes:
  - `scg_gate_used`
  - `scg_gate_result`
  - `scg_gate_reason`
- `run_refine.json` now includes:
  - `scg_state`
  - `scg_cache_reused`
  - `scg_resources_loaded`
  - `scg_expected_markers_count`
  - `refine_actions_scg_evaluated`
  - `refine_actions_scg_vetoed`
  - `refine_actions_scg_supported`

### Files changed

- `porebin/scg.py`
- `porebin/refine.py`
- `porebin/refine_qc.py`
- `porebin/refine_state.py`
- `tests/test_scg_qc.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not let SCG enter coarse
- did not let SCG drive contig-level initial assignment
- did not turn SCG into a unified scoring function
- did not activate merge / reassign / recruit SCG gates in the live orchestration
- did not change `associate.py`

### Current risks / blockers

- current SCG gating is still minimal and concentrated on decontam plus split placeholders
- split / merge / reassign / recruit are not yet fully operation-native in the active refine loop
- the repository still lacks bundled SCG resources, so default real-world execution still commonly exercises the no-resource path
- partial-hit interpretation remains lightweight; there is still no fragment-collapse or richer marker post-processing

### Verification

- ran:
  - `pytest -q tests/test_scg_qc.py tests/test_refine_mvp.py`
- result:
  - `7 passed`

### Next step

- Phase 5: downgrade remaining legacy heuristics into candidate generators / fallback helpers with explicit status
- first-touch files should be:
  - `porebin/refine.py`
  - `porebin/refine_legacy.py`
  - `docs/refactor_spec_v1.md`

## Phase 5: Downgrade Legacy Heuristics and Fix Their New Roles

Status: completed

### Goal

- classify remaining legacy helpers and heuristics explicitly
- reduce the influence of legacy contig-state semantics on the active refine mainline
- keep old logic available only as:
  - candidate generator
  - fallback helper
  - compatibility helper
  - deprecated helper

### What was done

#### Step 1: audited the remaining legacy-adjacent helpers

Re-audited the following objects:

- `_split_bins_parquet(...)`
- `_reassign_or_unbin(...)`
- `_recruit_gmm(...)`
- legacy contig-state bridge helpers:
  - `_build_legacy_compatibility_rows(...)`
  - `_compat_bridge_row(...)`
  - `_legacy_residual_record(...)`
  - `_write_contig_host_scores_tsv(...)`
- `porebin/refine_legacy.py::refine_bins(...)`

Confirmed active call-path status:

- active `refine_bins_parquet(...)` does **not** call:
  - `_split_bins_parquet(...)`
  - `_reassign_or_unbin(...)`
  - `_recruit_gmm(...)`
- active `refine_bins_parquet(...)` still calls the legacy compatibility bridge in order to:
  - write `contig_host_scores.tsv`
  - derive residual records during the migration period
- `porebin/refine_legacy.py` remains reachable only through the legacy `refine.refine_bins(...)` delegate

#### Step 2: hardened helper-role classification in spec

- added a fielded legacy-helper role table to `docs/refactor_spec_v1.md`
- classified each helper by:
  - current role
  - target role
  - call path
  - keep / wrap / deprecate decision
  - migration note

#### Step 3: reduced legacy-contig-state pressure in `refine.py`

- extracted the legacy bridge loop into `_build_legacy_compatibility_rows(...)`
- made the compatibility bridge explicit in comments and docstrings
- kept public refine outputs centered on:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `bin_qc.refined.tsv`
  - `refine_actions.tsv`
  - `run_refine.json`
- kept `contig_host_scores.tsv` only as a transitional audit / compatibility artifact
- moved helper-role reporting onto a shared `LEGACY_HELPER_ROLES` mapping so runtime metadata and spec stay aligned

#### Step 4: clarified `refine_legacy.py`

- marked `porebin/refine_legacy.py` as a legacy/backward-compatibility module
- clarified that `refine.refine_bins(...)` is a deprecated compatibility entrypoint, not the active architectural mainline

### Files changed

- `porebin/refine.py`
- `porebin/refine_legacy.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not delete legacy helpers
- did not rewrite the active refine algorithm
- did not promote split / reassign / recruit helpers back into the active loop
- did not alter coarse
- did not alter associate behavior
- did not add new SCG capabilities

### Current risks / blockers

- the active refine path still needs the legacy compatibility bridge for:
  - `contig_host_scores.tsv`
  - compatibility-derived residual mapping
- `_split_bins_parquet(...)`, `_reassign_or_unbin(...)`, and `_recruit_gmm(...)` remain in the file, even though their roles are now downgraded
- `refine_legacy.py` still exists as importable code, so CLI/docs must continue to avoid presenting it as the mainline

### Verification

- ran:
  - `pytest -q tests/test_scg_qc.py tests/test_refine_mvp.py`
- result:
  - `7 passed`

### Next step

- Phase 6: unify CLI/help/docstrings and user-facing narrative around:
  - `coarse -> initial bins`
  - `refine -> final bins + residual pool`
  - `associate -> downstream relation mining`

## Phase 6: Unify CLI, Docstrings, and User-Facing Narrative

Status: completed

### Goal

- align CLI and user-facing narrative with the new architecture
- make the public workflow read consistently as:
  - `cluster -> initial bins`
  - `refine -> final bins + residual_pool.tsv`
  - `associate -> downstream relation mining`
- downgrade compatibility and legacy artifacts in user-facing text without changing core behavior

### What was done

#### Step 1: unified CLI narrative

- updated `porebin/cli.py` app help to reflect the current three-stage user workflow
- updated command docstrings/help for:
  - `build`
  - `cluster`
  - `export`
  - `refine`
  - `associate`
  - `run-bam`
- made the CLI describe:
  - `cluster` as hypergraph coarse clustering that writes initial bins
  - `refine` as post-binning refinement that writes final bins plus residual handoff
  - `associate` as downstream association analysis that consumes `residual_pool.tsv`

#### Step 2: unified module docstrings

- added or updated user-facing module/entrypoint docstrings in:
  - `porebin/refine.py`
  - `porebin/associate.py`
  - `porebin/refine_legacy.py`
  - `porebin/cli.py`
- clarified that:
  - `refine.py` is the active post-binning refinement mainline
  - `associate.py` is downstream and read-only with respect to final bins
  - `refine_legacy.py` is a legacy/reference path, not the recommended default

#### Step 3: unified output semantics

- documented user-visible artifact tiers in `docs/refactor_spec_v1.md`
- explicitly separated:
  - primary outputs:
    - `bins.refined.tsv`
    - `residual_pool.tsv`
    - `associations.tsv`
  - QC / audit outputs:
    - `bin_qc.refined.tsv`
    - `refine_actions.tsv`
    - `run_refine.json`
  - compatibility / transition output:
    - `contig_host_scores.tsv`
- made `contig_host_scores.tsv` consistently described as compatibility / audit only

#### Step 4: synchronized documentation language

- updated `docs/refactor_spec_v1.md` to reflect the Phase-6 user-facing narrative
- aligned the architecture summary with the public workflow:
  - `cluster` -> initial bins
  - `refine` -> final bins + residual pool
  - `associate` -> downstream associations

### Files changed

- `porebin/cli.py`
- `porebin/refine.py`
- `porebin/associate.py`
- `porebin/refine_legacy.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not change coarse mathematics
- did not add new refine live operations
- did not change SCG gate behavior
- did not enhance associate scoring or output structure
- did not change output schemas

### Current risks / blockers

- `contig_host_scores.tsv` still exists and may continue to attract user attention unless later phases remove or further hide it
- refine still internally depends on the compatibility bridge, even though user-facing narrative no longer centers on it
- `export.py` remains a presentation layer with extra filtering semantics, so user education must still distinguish export from inference

### Verification

- ran:
  - `pytest -q tests/test_scg_qc.py tests/test_refine_mvp.py`
- result:
  - `7 passed`

### Next step

- next work should focus on algorithmic follow-up rather than more narrative cleanup
- highest-priority follow-up remains the coarse-side method review already discussed:
  - dual-hypergraph fusion parameters
  - embedding / HDBSCAN coupling
  - HDBSCAN behavior and stability

## Operation Activation - Part 1: Live Split

Status: completed

### Goal

- make `split` the first real live refinement operation
- keep `merge / reassign / recruit` inactive
- preserve bin-centric orchestration while allowing accepted split operations to change final bins

### What was done

#### Step 1: established split-only live-operation scope

- kept coarse unchanged
- kept associate unchanged
- kept SCG in a gate-only role
- activated only `split`

#### Step 2: added split data structures and QC support

- added `SplitCandidate` and `SplitDecision` to `porebin/refine_state.py`
- added contact-component membership extraction to `porebin/refine_qc.py`
- added an internal QC collection helper in `porebin/refine.py` so split evaluation could compare:
  - source-bin QC before split
  - child-bin QC after tentative split

#### Step 3: implemented live split generation and evaluation

- split candidates are now generated from preliminary final bins, not from coarse/cleaned bins directly
- candidate generation currently uses:
  - suspect bins
  - contact-component structure
  - component fragmentation
- split is currently a 2-way contact-component partition:
  - largest component stays in the source bin
  - second-largest component becomes a new bin
  - remaining components are pushed to residual if the split is accepted
- split evaluation now compares:
  - contact consistency before vs after
  - child-group size sanity
  - residual dominance protection
  - SCG gate outcome

#### Step 4: made accepted split modify real outputs

- accepted split now updates:
  - `bins.refined.tsv`
  - `residual_pool.tsv`
  - `bin_qc.refined.tsv`
  - `refine_actions.tsv`
  - `run_refine.json`
- residual contigs created by split are now written explicitly with:
  - `reason=split_residual`
  - `stage=split`

#### Step 5: enforced SCG-required split behavior

- if live split candidates exist and SCG resources are not enabled, refine now fails fast with a user-actionable error
- split does not silently degrade to a non-SCG path

#### Step 6: added tests

- added coverage for:
  - accepted live split
  - split-created residual contigs
  - SCG-vetoed split
  - no-resource split failure

### Files changed

- `porebin/refine.py`
- `porebin/refine_qc.py`
- `porebin/refine_state.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not activate merge
- did not activate reassign
- did not activate recruit
- did not change coarse
- did not bundle real SCG resources
- did not remove the compatibility bridge

### Current risks / blockers

- live split currently only uses contact-component-driven 2-way candidates; it does not yet use richer subgroup discovery
- split still runs after the preliminary final-bin derivation, so `contig_host_scores.tsv` remains a pre-split compatibility bridge
- real-world split usage still depends on providing actual SCG resources

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py`
- result:
  - `10 passed`

### Next step

- if live operations continue, the next priority should be `reassign`, not `recruit`
- reason:
  - split creates boundary contigs and residual candidates first
  - reassign is the most natural next operation for correcting those boundary cases before attempting broader recruitment


## Split v2 update: SCG-triggered local reclustering

### What changed

- removed contact-component partitioning from the active split-candidate generator
- moved SCG forward into split admission:
  - duplicated SCG and high contamination-like now act as the primary split-check trigger
  - contact / low-support / coverage signals remain auxiliary evidence
- switched the active split generator to a local 2-way reclustering step:
  - reuse coarse TNF136 feature construction
  - reuse coarse coverage feature semantics (`log1p(cov)`)
  - reuse coarse z-score normalization
  - run local `KMeans(n_clusters=2)` only inside the source bin
- kept contact in the split pipeline, but only as structural validation during candidate evaluation
- tightened SCG expectations for admitted bins:
  - SCG-priority split candidates must show SCG improvement (`support`) to be accepted

### Files changed

- `porebin/refine.py`
- `porebin/refine_qc.py`
- `porebin/refine_state.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not activate `reassign`
- did not activate `recruit`
- did not change coarse clustering
- did not change SCG calling logic
- did not remove the pre-split compatibility bridge

### Current risks / blockers

- local split currently depends on feature separability within the bin; if within-bin TNF/coverage contrast is weak, no split candidate may be produced
- contact no longer proposes the active split partition, so heavily feature-ambiguous bins still rely on SCG + validation to avoid bad cuts
- `contig_host_scores.tsv` remains a pre-split compatibility artifact

### Verification

- ran:
  - `pytest -q tests/test_scg_qc.py tests/test_refine_mvp.py`
- result:
  - `10 passed`

### Next step

- if refinement activation continues, the next live operation should still be `reassign`
- reason:
  - split now produces cleaner child bins
  - the next bottleneck is boundary / unresolved contigs after split, not broader recruitment


## Refine contract cleanup: SCG preflight and split-check visibility

### What changed

- tightened the active refine contract so SCG is now a hard preflight requirement:
  - `manifest.json`
  - manifest-declared HMM
  - `prodigal`
  - `hmmsearch`
  must all exist before `refine` starts
- removed `has_split_pressure(...)` from the active code path and deleted the helper itself
- kept `suspect` and `split-check` as separate concepts:
  - `suspect` = broad QC concern
  - `split-check` = admitted into active split generation
- made `split-check` explicit in QC rows:
  - `split_check_flag`
  - `split_check_reasons`
  - `split_priority_source`
- exported split-check summary into `run_refine.json`

### Files changed

- `porebin/refine.py`
- `porebin/refine_qc.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not change split acceptance philosophy
- did not change local recluster feature inputs
- did not add new split-specific TSV outputs
- did not expand residual handling; that remains deferred to later `reassign` / `recruit` work

### Current risks / blockers

- `split-check` is now visible in QC, but only the refined QC snapshot is part of the user-facing intent
- `contig_host_scores.tsv` remains a pre-split compatibility artifact
- edge-contig handling after split is still deferred to future `reassign`

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py`
- result:
  - `10 passed`

### Next step

- if refine operation work continues, the next design step should be `reassign`
- reason:
  - split now has a cleaner contract boundary
  - the next unresolved question is contig-level reassignment after a bin-level split


## SCG resource packaging update

### What changed

- added bundled SCG manifest at `porebin/scg_db/manifest.json`
- registered the current bundled HMM as `core_bacterial_scg.hmm` instead of requiring the filename `marker.hmm`
- updated `porebin.scg.resolve_scg_resources(...)` so the HMM filename comes from manifest first
- kept backward compatibility:
  - use `marker.hmm` if present and manifest does not specify a filename
  - otherwise auto-pick the only `*.hmm` file in the directory when unambiguous

### Why

- the project now has a real bundled HMM file with a non-default name
- SCG resource loading should follow manifest metadata rather than a hard-coded filename

### Files changed

- `porebin/scg.py`
- `porebin/scg_db/manifest.json`
- `porebin/scg_db/README.txt`
- `porebin/refine.py`
- `tests/test_scg_qc.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`


## Reassign v1 activation: post-split sibling-pair cleanup

### What changed

- activated a first live `reassign` operation in the parquet refine mainline
- kept the scope intentionally narrow:
  - `reassign` only runs after an accepted split
  - `reassign` only operates within the sibling pair created by that split
- added `ReassignWindow`, `ReassignCandidate`, and `ReassignDecision` state objects
- added a local pair-support scan over `contacts.parquet` for sibling bins
- implemented a conservative one-pass `reassign` policy:
  - `keep`
  - `move_to_sibling`
  - `residualize`
- kept `contact` as the primary local cleanup signal for `reassign`
- kept SCG as a non-worsening gate only:
  - SCG does not generate `reassign` candidates
  - SCG may veto `move_to_sibling` or `residualize` if source/target QC worsens
- extended `refine_actions.tsv` so accepted/rejected `reassign` decisions are now audited
- extended `run_refine.json` with:
  - `reassign_windows_total`
  - `reassign_candidates_total`
  - `reassign_candidates_move_to_sibling`
  - `reassign_candidates_residualize`
  - `reassign_decisions_accepted`
  - `reassign_decisions_moved`
  - `reassign_decisions_residualized`

### Files changed

- `porebin/refine.py`
- `porebin/refine_state.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not allow `reassign` to jump into arbitrary third bins
- did not activate `recruit`
- did not activate `merge`
- did not revive `_reassign_or_unbin(...)` as active orchestration logic
- did not change split admission or split acceptance philosophy

### Current risks / blockers

- `reassign v1` is intentionally local and conservative; it only cleans up split-created sibling pairs
- global bin-to-bin reassignment is still not implemented
- `recruit` is still required for broader residual recovery after local cleanup

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py`
- result:
  - `12 passed`

### Next step

- if live refinement work continues, the next operation should be `recruit`
- reason:
  - split now creates child bins
  - `reassign` now repairs the obvious sibling-level boundary cases
  - the remaining gap is conservative recovery from `residual_pool.tsv`


## Recruit v1 activation and residual vocabulary cleanup

### What changed

- activated a first live `recruit` operation in the parquet refine mainline
- cleaned `residual_pool.tsv` vocabulary so legacy residual rows no longer emit:
  - `stage = coarse_import`
  - `stage = final_filter`
  - `reason = unresolved`
- legacy/base refine residual rows now use:
  - `stage = refine_base`
  - `reason = assignment_unresolved`
  - `reason = associate_candidate`
- kept the residual schema stable for downstream compatibility:
  - `contig_name`
  - `reason`
  - `stage`
  - `coarse_bin_id`
  - `refined_status`
  - `note`
- kept `associate.py` on the same contract:
  - it still keys off `refined_status = residual_associate_candidate`
- implemented a conservative `recruit v1` policy:
  - only consumes residual rows with `refined_status = residual_unresolved`
  - may assign a residual contig back into any existing final bin
  - may keep the contig in the residual pool
  - does not create new bins
  - does not perform merge-like operations
- kept `contact` as the primary recovery signal for `recruit`
- kept SCG as a non-worsening gate only for `recruit`
- extended `refine_actions.tsv` and `run_refine.json` with recruit audit/statistics

### Files changed

- `porebin/refine.py`
- `porebin/refine_state.py`
- `tests/test_refine_mvp.py`
- `docs/refactor_spec_v1.md`
- `docs/refactor_journal.md`

### What was not done

- did not change the residual TSV schema
- did not rename `coarse_bin_id` or `refined_status`
- did not activate `merge`
- did not revive `_recruit_gmm(...)` as active orchestration logic

### Current risks / blockers

- `recruit v1` is intentionally conservative and one-pass
- global residual recovery is still contact-affinity driven and may miss weak-but-real recoverable contigs
- legacy compatibility rows still feed the initial residual pool, even though their vocabulary is now normalized

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py`
- result:
  - `13 passed`

### Next step

- if live refinement work continues, the next design question should be whether `reassign` and `recruit` remain one-pass or become iterative
- merge remains lowest priority


## Coarse audit instrumentation round 1

### What changed

- kept the active coarse algorithm unchanged at a high level:
  - joint contact-feature spectral embedding
  - HDBSCAN
  - optional contact-component postprocess
- made coarse-side control knobs explicit in the active spectral entrypoints:
  - `lambda_contact`
  - `feature_mode`
  - `feature_knn_k`
  - `embedding_dim`
  - `hdbscan_min_cluster_size`
  - `hdbscan_min_samples`
  - `hdbscan_selection_method`
  - `contact_postprocess`
- added a first coarse feature ablation mode:
  - `feature_mode = tnf_only`
  - existing default remains `tnf_plus_cov`
- added a coarse postprocess master switch:
  - `contact_postprocess = True/False`
  - did not add internal sub-switches for individual postprocess actions
- extended coarse metadata to make the main risk points observable:
  - contact component counts / largest-share summary
  - raw HDBSCAN noise statistics before contact postprocess
  - coverage missing fraction
  - contact-feature neighbor overlap summary
  - whether contact postprocess was enabled

### Files changed

- `porebin/cli.py`
- `porebin/cluster.py`
- `porebin/hypergraph_joint_spectral.py`
- `tests/test_spectral_cluster.py`

### What was not done

- did not change the joint operator formula
- did not implement adaptive `lambda_contact`
- did not split contact postprocess into internal sub-switches
- did not change refine behavior

### Verification

- ran:
  - `pytest -q tests/test_refine_mvp.py tests/test_scg_qc.py tests/test_spectral_cluster.py tests/test_many_bins_not_limited.py tests/test_joint_operator_shapes.py`
- result:
  - `17 passed`

### Next step

- run coarse control experiments on real data:
  - `lambda_contact` grid
  - `feature_mode` comparison
  - `contact_postprocess` on/off
  - optional HDBSCAN selection-method comparison
