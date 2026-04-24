# porebin Architecture

## Tool identity

`porebin` is a genome-centric Pore-C metagenomic binning tool.

Given contigs, Pore-C contact evidence, and coverage, it:

1. constructs canonical evidence
2. discovers candidate genome bins
3. refines candidate bins into final genome bins
4. exports final bins and unresolved contigs

## Mainline modules

```text
porebin_genome/
  cli/
  io/
  evidence/
  coarse/
  refine/
  qc/
  export/
```

### `evidence/`

Builds canonical `contacts.parquet` and `coverage.tsv` from a queryname-sorted BAM.

### `coarse/`

Implements the default coarse discovery mainline:

- contact incidence construction
- TNF136 + coverage features
- feature kNN incidence
- joint operator construction
- spectral embedding
- HDBSCAN clustering

Outputs are candidate genome bins only.

### `refine/`

Implements the genome-centric refine MVP:

- suspect bin detection
- local split
- move-to-target-bin reassignment
- conservative recruitment
- abstain / keep unbinned
- bin QC and action logging

### `export/`

Exports final bins and unresolved contigs as FASTA.

## Public contracts

### Inputs

- `contigs.fasta`
- `contacts.parquet`
- `coverage.tsv`

`reads.namesorted.bam` belongs to the evidence layer, not the core binning contract.

### Outputs

- `coarse/bins.tsv`
- `coarse/run.json`
- `final/bins.refined.tsv`
- `final/unbinned.tsv`
- `final/bin_qc.tsv`
- `final/refine_actions.tsv`
- `final/refine_meta.json`

## Refine semantics

- `assignment_confidence` is a 0..1 rule score
- reassign uses `move_to_target_bin` semantics only
- split requires minimum child size and at least one improved consistency metric
- low-confidence contigs remain explicit in `unbinned.tsv`

## Scope of this branch

This branch keeps only the genome-centric mainline. Legacy host-centric code, legacy CLI paths, pairwise baseline paths, compatibility bridges, and historical refactor notes are not part of the active public surface here.
