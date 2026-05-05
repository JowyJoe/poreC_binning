# porebin

`porebin` is a genome-centric Pore-C metagenomic binning tool. It combines Pore-C contact evidence, sequence composition, and coverage to discover candidate genome bins, refine them into final genome bins, and keep unresolved contigs explicit.

## Install

```bash
pip install -e ".[bam,spectral,dev]"
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate porebin-genome
pip install -e . --no-deps
```

### External tool dependencies for refine

`porebin bin` now enables internal SCG veto by default during refinement. This requires:

- `prodigal`
- `hmmsearch` from HMMER

If either tool is missing, refine will raise an explicit dependency error. For development or testing runs where SCG veto is intentionally disabled, use `--disable-scg`.

Install these command-line tools with conda:

```bash
conda install -c conda-forge -c bioconda prodigal hmmer
```

or with mamba:

```bash
mamba install -c conda-forge -c bioconda prodigal hmmer
```

If you need to sort BAM files before evidence construction, install `samtools` separately:

```bash
conda install -c conda-forge -c bioconda samtools
```

## Public CLI

- `porebin evidence`: build canonical `contacts.parquet` and `coverage.tsv` from a queryname-sorted BAM
- `porebin bin`: run coarse candidate genome-bin discovery and genome-bin refinement
- `porebin export`: export final genome bins and unresolved contigs as FASTA

The public binning contract is:

```text
contigs.fasta + contacts.parquet + coverage.tsv
  -> coarse/bins.tsv
  -> coarse/run.json
  -> final/bins.refined.tsv
  -> final/unbinned.tsv
  -> final/bin_qc.tsv
  -> final/refine_actions.tsv
  -> final/refine_meta.json
```

## Recommended workflow

```bash
# build evidence from a queryname-sorted BAM
porebin evidence \
  --bam reads.namesorted.bam \
  --contigs contigs.fasta \
  --out run_out

# run coarse discovery + refine MVP
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --out run_out

# optionally disable internal SCG veto for environments without prodigal/hmmsearch
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --disable-scg \
  --out run_out

# export final bins and unresolved contigs
porebin export \
  --contigs contigs.fasta \
  --bins-refined-tsv run_out/final/bins.refined.tsv \
  --unbinned-tsv run_out/final/unbinned.tsv \
  --out export_out
```

## Output semantics

### `coarse/bins.tsv`

Candidate genome bins from the coarse discovery stage. These are not final bins.

### `final/bins.refined.tsv`

Final genome-bin assignments with:

- `contig_id`
- `bin_id`
- `assignment_stage`
- `assignment_reason`

### `final/unbinned.tsv`

Contigs that remain unresolved after refinement, with explicit stage and reason.

### `final/bin_qc.tsv`

 Bin-level refine summary including contig count, total length, median coverage, contact coherence, compact SCG status, suspect flag, and refine status.

### `final/refine_actions.tsv`

Accepted and rejected refine actions for split, reassign, merge, and recruit, including per-action `delta_contact` and compact `scg_status`.

### `final/refine_meta.json`

Stage-level counts for suspect bins, split/reassign/recruit candidates, accepted actions, rejected actions, and final unbinned contigs.

## Method boundaries

- coarse discovery does not promote HDBSCAN noise into bins by component-majority postprocessing
- refine is genome-centric and does not use host-centric semantics
- unresolved contigs remain explicit instead of being forced into bins
- SCG veto is enabled by default in refine and requires external `prodigal` and `hmmsearch`

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
