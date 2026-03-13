# porebin (v0.1.0)

Host-centric, hypergraph-preserving metagenome binning from Nanopore Pore-C multi-way contacts.

Core idea:

- treat each multi-way contact as one hyperedge
- preserve the hyperedge structure without clique expansion in the main method
- combine contact hypergraph signal and feature hypergraph signal for coarse clustering
- run a refine stage that performs host-assignment inference on top of coarse candidate host communities

## Install

```bash
pip install -e ".[bam,spectral]"
```

Key dependencies:

- required: `pyarrow`, `python-igraph`, `leidenalg`, `typer`, `rich`
- BAM path: `pysam`
- spectral coarse clustering: `scipy`, `scikit-learn`, `hdbscan`

### Conda install

```bash
conda env create -f environment.yml
conda activate porebin
pip install -e . --no-deps
```

If you use the Conda environment for the main spectral pipeline, `scikit-learn` and `hdbscan`
must also be present. The checked-in `environment.yml` now includes them.

## Documentation

- Current architecture, math, interfaces, and legacy paths:
  [`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md)
- Short context/status note:
  [`docs/CONTEXT.md`](docs/CONTEXT.md)

`docs/PROJECT_REFERENCE.md` is the current source-of-truth document for repository structure and behavior.

## Commands

- `porebin run-bam`: end-to-end BAM pipeline. Add `--pairwise-baseline` for the clique-expansion control.
- `porebin bam2contacts`: convert a name-sorted BAM into `contacts.parquet` and `coverage.tsv`.
- `porebin build`: write graph audit artifacts from `contacts.parquet`.
- `porebin cluster`: run coarse clustering (`spectral` is the active method).
- `porebin refine`: run host-assignment inference from `contacts.parquet` and coarse `bins.tsv`.
- `porebin export`: export FASTA bins and `unbinned.fasta`.

All commands write `out_dir/run.json`. `porebin refine` also writes `out_dir/run_refine.json`.

## Recommended workflow

```bash
# upstream example:
# minimap2 ... | samtools sort -n -o reads.namesorted.bam

porebin run-bam --bam reads.namesorted.bam --contigs contigs.fasta --out out_bam --seed 0
porebin export --contigs contigs.fasta --bins-tsv out_bam/refined/bins.refined.tsv --out out_bam/final_bins
```

Pairwise baseline:

```bash
porebin run-bam --pairwise-baseline --bam reads.namesorted.bam --contigs contigs.fasta --out out_pw --seed 0
porebin export --contigs contigs.fasta --bins-tsv out_pw/bins.tsv --out out_pw/final_bins
```

## Input summary

### Name-sorted BAM

The main BAM path assumes one read name equals one hyperedge.

```bash
samtools sort -n -o reads.namesorted.bam reads.bam
```

### contacts.parquet

`contacts.parquet` is the evidence-layer source of truth in the current pipeline.

The detailed schema and semantics are documented in:

- [`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md)

Important fields include:

- `contigs`
- `contig_weights`
- `k`
- `k_eff`
- `weight`
- BAM/QC evidence columns

## Notes

- Main coarse path: joint hypergraph spectral embedding + HDBSCAN
- Main refine path: parquet-based host-assignment inference with soft gating
- Pairwise baseline is a control path, not the main method
- Historical and legacy paths still exist in the repository and are documented in `docs/PROJECT_REFERENCE.md`
