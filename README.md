# porebin (v0.1)

Hypergraph-preserving metagenome binning from Nanopore Pore-C multi-way contacts.

Core idea: treat each multi-way contact as a **hyperedge**, keep it via a **contig–contact bipartite graph**, then run **Leiden** on that bipartite graph. No clique expansion.

## Install

```bash
pip install -e .
```

Dependencies (key ones): `pyarrow`, `python-igraph`, `leidenalg`, `typer`, `rich`.

## Commands

- `porebin normalize`: PPL `.contacts` (segment-level TSV) → internal `contacts.parquet`
- `porebin build`: contigs FASTA + `contacts.parquet` → `graph/` (COO edges)
- `porebin cluster`: Leiden on bipartite graph → `bins.tsv`
- `porebin export`: `bins.tsv` + FASTA → `bins_fasta/bin_<id>.fasta`
- `porebin run`: normalize + build + cluster + export

All commands write `out_dir/run.json` (parameters, time, version, seed).

## Input formats

### PPL `.contacts` (for `porebin normalize`)

TSV with **11 or 12 columns**, with or without header. The normalizer needs these fields:
- `readID`: read identifier (used to group segments into a contact)
- `chr`: contig/reference name
- `status`: default keeps only `passed`
- `score` (optional): tags like `mapq:60;AS:123` (parsed if present)

Notes:
- For best streaming/memory usage, `.contacts` should be grouped by `readID`.
- Output: `out_dir/contacts/contacts.parquet` + `out_dir/contacts/qc.json`.

### Normalized contacts Parquet (for `porebin build`)

`contacts.parquet` contains one row per contact/hyperedge:
- required: `contact_id`, `contigs` (list[str]), `k`, `weight`, `support_count`
- optional evidence: `mapq_min`, `mapq_mean`, `as_sum`, `n_segments`

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

porebin run --ppl-contacts demo/example.contacts --contigs demo/contigs.fasta --out demo/out --seed 0
```

Outputs:
- `demo/out/contacts/contacts.parquet`
- `demo/out/graph/edges.tsv` etc
- `demo/out/bins.tsv`
- `demo/out/bins_fasta/bin_<id>.fasta`
