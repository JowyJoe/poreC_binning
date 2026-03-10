# porebin context

## Goal
Host-centric metagenome binning from Nanopore Pore-C multi-way contacts while preserving hyperedges (no clique expansion).

Primary goal:
- output bins corresponding to host genomes / host communities.

Secondary goal:
- output conservative structural association summaries for accessory / MGE-like contigs (not a hard classifier).

We represent each multi-way contact as a **hyperedge**, and keep the hyperedge structure via a **contig–contact bipartite view**:
- Left nodes: contigs
- Right nodes: contacts (hyperedges)
- Edge exists if contig participates in the contact
- Evidence-layer edge mass is governed by the read-level quality weight `q(r)` (stored as `contacts.parquet.weight`),
  plus the normalized evidence share `pi_{r,c}` (stored as `contacts.parquet.contig_weights`).

## Pipeline
1) `porebin run-bam`: BAM → contacts.parquet → build graph (audit/artifacts) → coarse cluster (`bins.tsv`, candidate host communities) → (optional) refine inference (`bins.refined.tsv`)
2) `porebin export`: export FASTA with built-in policy (keep bins ≥200kb; short contigs/tiny bins → `unbinned.fasta`)

Coarse clustering (`spectral` in v0.1.0) uses a **joint hypergraph spectral embedding**
(contact hypergraph + feature hypergraph) followed by **HDBSCAN** (no fixed K, no recursive BIC bisection).

Refine is a host-assignment inference layer:
- outputs per-contig host support / posterior-like scores `theta_{c,b}` (not to be confused with `pi_{r,c}`)
- outputs uncertainty summaries (top1/top2/margin/entropy/effective_hosts)
- outputs a separate accessory association head (structural, conservative)

## BAM (name-sorted)
We treat each read (QNAME) as one hyperedge, so the BAM must be queryname-sorted (e.g. `samtools sort -n`).

`porebin bam2contacts` aggregates per QNAME and writes `out_dir/contacts/contacts.parquet`.

Output internal format: `out_dir/contacts/contacts.parquet` with columns:
- `contact_id` (int)
- `contigs` (list[str])
- `contig_weights` (list[float], normalized evidence shares \(\\pi_{r,c}\); sums to 1 over contigs touched by the read; not a posterior)
- `k` (int)
- `k_eff` (float)
- `weight` (float, read quality weight \(q(r)\), clamped to [0,1]; no multi-way concentration penalty)
- `support_count` (int)
- evidence/QC columns: `mapq_min`, `p_ok_mean`, `aligned_len_sum`, `nm_sum`, `n_segments`, `mapq_missing_count`, `nm_missing_count`, `len_missing_count`
