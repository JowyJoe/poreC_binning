# porebin (v0.1)

Host-centric, hypergraph-preserving metagenome binning from Nanopore Pore-C multi-way contacts.

Core idea: treat each multi-way contact as a **hyperedge** (one read = one hyperedge), preserve it without clique expansion, and cluster contigs using the hypergraph signal.

## Install

```bash
pip install -e .
```

Dependencies (key ones): `pyarrow`, `python-igraph`, `leidenalg`, `typer`, `rich`.
Optional (BAM input via `bam2contacts` / `run-bam`): `pysam`. Optional (spectral coarse clustering): `scipy`, `scikit-learn`, `hdbscan`.

### Install (conda, recommended on servers)

Use the provided `environment.yml` to get a stable stack for `pyarrow/python-igraph/leidenalg`:

```bash
conda env create -f environment.yml
conda activate porebin
pip install -e . --no-deps
```

If you prefer not to use `environment.yml`:

```bash
conda create -n porebin python=3.10 -y
conda activate porebin
conda install -c conda-forge -y pyarrow python-igraph leidenalg typer rich
pip install -e . --no-deps
```

## Commands

- `porebin run-bam`: end-to-end BAM pipeline (recommended). Add `--pairwise-baseline` for the normal-graph control.
- `porebin bam2contacts`: name-sorted BAM → `contacts.parquet` (+ `coverage.tsv`)
- `porebin refine`: host-assignment inference on top of coarse candidate host communities (inputs: `contacts.parquet` + `bins.tsv`)
- `porebin export`: export FASTA with built-in policy (keep bins ≥200kb; short contigs/tiny bins → `unbinned.fasta`)
- Advanced: `porebin build`, `porebin cluster`, `porebin refine`

All commands write `out_dir/run.json` (parameters, time, version, seed). `porebin refine` additionally writes `out_dir/run_refine.json`.

## Typical workflow (no threshold parameters)

```bash
# (Recommended) BAM pipeline: coarse + refine
# upstream example: minimap2 ... | samtools sort -n -o reads.namesorted.bam
porebin run-bam --bam reads.namesorted.bam --contigs contigs.fasta --out out_bam --seed 0
porebin export --contigs contigs.fasta --bins-tsv out_bam/refined/bins.refined.tsv --out out_bam/final_bins

# Pairwise baseline (normal graph control)
porebin run-bam --pairwise-baseline --bam reads.namesorted.bam --contigs contigs.fasta --out out_pw --seed 0
porebin export --contigs contigs.fasta --bins-tsv out_pw/bins.tsv --out out_pw/final_bins
```

## Experimental: hypergraph spectral coarse clustering

This branch provides an optional coarse clustering method:
- `--method spectral` for `porebin cluster` (only supported method)
- `--coarse-method spectral` (default; only supported method) for `porebin run-bam`

Install optional deps:

```bash
pip install -e '.[spectral]'
```

Run:

```bash
porebin run-bam --coarse-method spectral --bam reads.namesorted.bam --contigs contigs.fasta --out out_bam_spectral --seed 0
```

## Input formats

### Name-sorted BAM (for `porebin bam2contacts` / `porebin run-bam`)

We treat each read (QNAME) as one hyperedge, so the BAM must be **queryname-sorted**:

```bash
samtools sort -n -o reads.namesorted.bam reads.bam
samtools index reads.namesorted.bam  # optional
```

This pipeline is designed for minimap2 + samtools, but works with any aligner that outputs standard BAM fields/tags.

### Contacts Parquet (for `porebin build`)

`contacts.parquet` contains one row per contact/hyperedge:
- required: `contact_id`, `contigs` (list[str]), `k`, `weight`, `support_count`
- optional: `contig_weights` (normalized evidence shares \(\\pi_{r,c}\); sums to 1 over contigs touched by the read; not a posterior)
- optional evidence: `mapq_min`, `mapq_mean`, `as_sum`, `n_segments`

## Pairwise baseline (for papers)

`porebin run-bam --pairwise-baseline` builds a **pairwise normal graph** control via clique expansion, with fair per-read weights:
- for each read with order `k`, each pair gets `w_pair = 2/(k*(k-1)) = 1/C(k,2)` so that all pairs from that read sum to 1.

Run:

```bash
porebin run-bam --pairwise-baseline --bam reads.namesorted.bam --contigs contigs.fasta --out out_pw --seed 0
```

