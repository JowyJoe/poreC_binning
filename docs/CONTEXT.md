# porebin context

## Goal
Use Nanopore Pore-C multi-way contacts for metagenome binning while preserving hyperedges (no clique expansion).

We represent each multi-way contact as a **hyperedge**, and keep the hyperedge structure via a **contig–contact bipartite graph**:
- Left nodes: contigs
- Right nodes: contacts (hyperedges)
- Edge exists if contig participates in the contact
- Edge weight (v0.1): `OrderNorm(k) * weight`, with `OrderNorm(k)=2/(k*(k-1))` (skip `k<2`)

## Pipeline
1) `porebin run`: normalize → build bipartite graph → cluster (coarse `bins.tsv`)
2) `porebin refine`: recruit/decontam/split with built-in defaults/auto-thresholds → `bins.refined.tsv`
3) `porebin export`: export FASTA with built-in policy (keep bins ≥200kb; short contigs/tiny bins → `unbinned.fasta`)

Experimental (branch): coarse clustering can be switched to hypergraph spectral clustering
(`--coarse-method spectral` / `--method spectral`), which uses a normalized hypergraph Laplacian
and k-means on the leading eigenvectors.

## PPL `.contacts` (segment-level TSV)
The normalizer expects a TSV produced by PPL-Toolbox with **11 or 12 columns**, with or without header.

Required semantics per segment row:
- `readID`: read identifier (used to group segments into a contact)
- `chr`: contig/reference name (used to build contig set per read)
- `status`: used for filtering/weighting (most steps use `passed` only by default)
- `score` (optional): may contain tags like `mapq:60;AS:123` (parsed if present)

`porebin normalize` aggregates by `readID` (for best memory usage, the file should be grouped by readID):
- `contigs = unique(chr)` per read
- drop contacts with `k=len(contigs) < 2`

Output internal format: `out_dir/contacts/contacts.parquet` with columns:
- `contact_id` (int)
- `contigs` (list[str])
- `k` (int)
- `weight` (float, currently 1.0)
- `support_count` (int, currently 1)
- optional evidence columns (enabled via `--include-tags`): `mapq_min`, `mapq_mean`, `as_sum`, `n_segments`
