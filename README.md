# porebin (v0.1)

Hypergraph-preserving metagenome binning from Nanopore Pore-C multi-way contacts.

Core idea: treat each multi-way contact as a **hyperedge**, keep it via a **contig–contact bipartite graph**, then run **Leiden** on that bipartite graph (no clique expansion).

## Install

```bash
pip install -e .
```

Dependencies (key ones): `pyarrow`, `python-igraph`, `leidenalg`, `typer`, `rich`.
Optional (BAM input via `bam2contacts` / `run-bam`): `pysam`. Optional (experimental spectral coarse clustering): `scipy`.

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

- `porebin run-bam`: bam2contacts + build + coarse cluster + (optional) refine (recommended)
- `porebin bam2contacts`: name-sorted BAM → `contacts.parquet` (+ `coverage.tsv`)
- `porebin run`: legacy PPL `.contacts` pipeline (normalize + build + cluster)
- `porebin refine`: refine coarse bins (BAM/parquet mode via `--contacts`, or legacy PPL mode via `--ppl-contacts`)
- `porebin export`: export FASTA with built-in policy (keep bins ≥200kb; short contigs/tiny bins → `unbinned.fasta`)
- Advanced: `porebin normalize`, `porebin build`, `porebin cluster`

All commands write `out_dir/run.json` (parameters, time, version, seed). `porebin refine` additionally writes `out_dir/run_refine.json`.

## Typical workflow (no threshold parameters)

```bash
# (Recommended) BAM pipeline: coarse + refine
# upstream example: minimap2 ... | samtools sort -n -o reads.namesorted.bam
porebin run-bam --bam reads.namesorted.bam --contigs contigs.fasta --out out_bam --seed 0
porebin export --contigs contigs.fasta --bins-tsv out_bam/refined/bins.refined.tsv --out out_bam/final_bins

# (Legacy) PPL pipeline
porebin run --ppl-contacts porec.contacts --contigs contigs.fasta --out coarse --seed 0
porebin refine --contigs contigs.fasta --ppl-contacts porec.contacts --bins-tsv coarse/bins.tsv --out refined --seed 0
porebin export --contigs contigs.fasta --bins-tsv refined/bins.refined.tsv --out final_bins
```

## Experimental: hypergraph spectral coarse clustering

This branch provides an optional coarse clustering method:
- `--coarse-method spectral` for `porebin run`
- `--method spectral` for `porebin cluster`

Install optional deps:

```bash
pip install -e '.[spectral]'
```

Run:

```bash
porebin run --coarse-method spectral --ppl-contacts porec.contacts --contigs contigs.fasta --out coarse_spectral --seed 0
```

## Input formats

### Name-sorted BAM (for `porebin bam2contacts` / `porebin run-bam`)

We treat each read (QNAME) as one hyperedge, so the BAM must be **queryname-sorted**:

```bash
samtools sort -n -o reads.namesorted.bam reads.bam
samtools index reads.namesorted.bam  # optional
```

This pipeline is designed for minimap2 + samtools, but works with any aligner that outputs standard BAM fields/tags.

### PPL `.contacts` (for `porebin run/normalize/refine`)

TSV with **11 or 12 columns**, with or without header. The normalizer expects these semantics:
- `readID`: read identifier (used to group segments into a contact)
- `chr`: contig/reference name
- `status`: `passed` is typically used by default; you can override in `porebin normalize` via `--keep-status`
- `score` (optional): tags like `mapq:60;AS:123` (parsed if present)

Note: for best streaming/memory usage, `.contacts` should be grouped by `readID`.

### Contacts Parquet (for `porebin build`)

`contacts.parquet` contains one row per contact/hyperedge:
- required: `contact_id`, `contigs` (list[str]), `k`, `weight`, `support_count`
- optional: `contig_weights` (soft incidence \(P_{r,c}\))
- optional evidence: `mapq_min`, `mapq_mean`, `as_sum`, `n_segments`

## Pairwise baseline (for papers)

`porebin run --pairwise-baseline` builds a **pairwise normal graph** control via clique expansion, with fair per-read weights:
- for each read with order `k`, each pair gets `w_pair = 2/(k*(k-1)) = 1/C(k,2)` so that all pairs from that read sum to 1.

Prepare a passed-only contacts file (column 11 must be `passed`):

```bash
awk -F'\t' '$11=="passed"' hyper.merged.contacts > porec.passed.contacts
```

If needed, sort/group by readID (column 4):

```bash
sort -k4,4 porec.passed.contacts > porec.passed.sorted.contacts
```

Run:

```bash
porebin run --pairwise-baseline --ppl-contacts porec.passed.sorted.contacts --contigs contigs.fasta --out out_pw --seed 0
```

## Minimal example

```bash
mkdir -p demo

cat > demo/contigs.fasta <<'EOF'
>ctgA
AAAA
>ctgB
CCCC
>ctgC
GGGG
EOF

# A tiny headered, 11-column-ish example (only readID/chr/score/status are used)
printf "readID\tseg\tchr\tstart\tend\tstrand\tcol7\tcol8\tcol9\tscore\tstatus\n" > demo/example.contacts
printf "r1\t0\tctgA\t1\t2\t+\t.\t.\t.\tmapq:60;AS:10\tpassed\n" >> demo/example.contacts
printf "r1\t1\tctgB\t1\t2\t+\t.\t.\t.\tmapq:40;AS:7\tpassed\n"  >> demo/example.contacts
printf "r2\t0\tctgA\t1\t2\t+\t.\t.\t.\tmapq:50;AS:8\tpassed\n"  >> demo/example.contacts
printf "r2\t1\tctgC\t1\t2\t+\t.\t.\t.\tmapq:50;AS:9\tpassed\n"  >> demo/example.contacts

porebin run --ppl-contacts demo/example.contacts --contigs demo/contigs.fasta --out demo/coarse --seed 0
porebin export --contigs demo/contigs.fasta --bins-tsv demo/coarse/bins.tsv --out demo/final
```

Outputs:
- `demo/coarse/contacts/contacts.parquet`
- `demo/coarse/graph/edges.tsv` etc
- `demo/coarse/bins.tsv`
- `demo/final/bins_fasta/bin_<id>.fasta` (>=200kb bins only)
- `demo/final/unbinned.fasta` + `demo/final/unbinned.tsv`
