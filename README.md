# porebin

`porebin` is a genome-centric Pore-C metagenomic binning tool. It combines Pore-C contact evidence, sequence composition, and coverage to discover candidate genome bins, refine them into final genome bins, and keep unresolved contigs explicit.

## Install

```bash
pip install -e ".[dev]"
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate porebin-genome
pip install -e . --no-deps
```

PyTorch is a core dependency because the HG-VAE coarse route is part of the active tool design. Fresh `pip` or `conda` environments install it during setup rather than treating it as an optional plugin.

The conda environment also installs command-line tools needed for full workflow testing: CoverM, samtools, prodigal, and HMMER.

### Server smoke test

From a fresh server checkout:

```bash
cd hypergraphBinning
conda env create -f environment.yml
conda activate porebin-genome
python -m pip install -e . --no-deps
```

Check Python packages and command-line tools:

```bash
python - <<'PY'
import torch
import pyarrow
import pysam
import hdbscan
import igraph
import leidenalg
import porebin_genome

print("porebin", porebin_genome.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

porebin --help
porebin bin --help
coverm --version
samtools --version
prodigal -v
hmmsearch -h | head
```

Run the repository tests:

```bash
python -m pytest -q
```

For a focused binning smoke test:

```bash
python -m pytest -q \
  tests/test_porebin_genome_skeleton_e2e.py \
  tests/test_porebin_genome_coarse_integration.py \
  tests/test_porebin_genome_hyperedge_embedding.py
```

### Server end-to-end run on real data

Set paths first:

```bash
CONTIGS=/path/to/contigs.fasta
BAM=/path/to/contig_aligned_porec.bam
WORK=$PWD/porebin_server_test
THREADS=16

mkdir -p "$WORK"
```

Prepare queryname-sorted and coordinate-sorted BAMs:

```bash
samtools sort -n -@ "$THREADS" -o "$WORK/reads.namesorted.bam" "$BAM"
samtools sort -@ "$THREADS" -o "$WORK/reads.coordsorted.bam" "$BAM"
samtools index "$WORK/reads.coordsorted.bam"
```

Build canonical Pore-C contact evidence and coverage:

```bash
porebin evidence \
  --bam "$WORK/reads.namesorted.bam" \
  --coverage-bam "$WORK/reads.coordsorted.bam" \
  --contigs "$CONTIGS" \
  --out "$WORK/evidence_run"
```

Run the default spectral hypergraph route:

```bash
porebin bin \
  --contigs "$CONTIGS" \
  --contacts "$WORK/evidence_run/evidence/contacts.parquet" \
  --coverage-tsv "$WORK/evidence_run/evidence/coverage.tsv" \
  --coarse-method spectral \
  --pairwise-baseline \
  --out "$WORK/spectral_run"
```

Run the complete HG-VAE route:

```bash
porebin bin \
  --contigs "$CONTIGS" \
  --contacts "$WORK/evidence_run/evidence/contacts.parquet" \
  --coverage-tsv "$WORK/evidence_run/evidence/coverage.tsv" \
  --coarse-method hgvae \
  --embedding-scorer-mode report \
  --out "$WORK/hgvae_run"
```

If you want a fast algorithm-only smoke test without SCG checks, add `--disable-scg` to the `porebin bin` commands. For final benchmarking, keep SCG enabled.

Export HG-VAE final bins:

```bash
porebin export \
  --contigs "$CONTIGS" \
  --bins-refined-tsv "$WORK/hgvae_run/final/bins.refined.tsv" \
  --unbinned-tsv "$WORK/hgvae_run/final/unbinned.tsv" \
  --out "$WORK/hgvae_export"
```

Inspect key outputs:

```bash
python - <<PY
import json
from pathlib import Path

root = Path("$WORK")
for name in ("spectral_run", "hgvae_run"):
    coarse = json.loads((root / name / "coarse" / "run.json").read_text())
    refine = json.loads((root / name / "final" / "refine_meta.json").read_text())
    print(name)
    print("  coarse_method:", coarse.get("coarse_method"))
    print("  n_bins:", coarse.get("n_bins"))
    print("  n_contigs_clustered:", coarse.get("n_contigs_clustered"))
    print("  n_contigs_unbinned:", coarse.get("n_contigs_unbinned"))
    print("  final_n_bins:", refine.get("n_bins_out"))
    print("  final_unbinned:", refine.get("n_unbinned_final"))
PY

ls "$WORK/hgvae_run/coarse"
ls "$WORK/hgvae_run/final"
ls "$WORK/hgvae_export/export"
```

### External command-line dependencies

`porebin evidence` uses CoverM by default to compute contig mean depth from a contig-aligned BAM:

- `coverm`

Install CoverM with conda:

```bash
conda install -c conda-forge -c bioconda coverm
```

or with mamba:

```bash
mamba install -c conda-forge -c bioconda coverm
```

`coverage.tsv` stores per-contig mean read depth for binning abundance consistency. It does not store the coverage-breadth percentage reported by some tools.

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

If you need to sort BAM files before evidence construction or CoverM coverage, install `samtools` separately:

```bash
conda install -c conda-forge -c bioconda samtools
```

## Public CLI

- `porebin evidence`: build canonical `contacts.parquet` from a queryname-sorted BAM and `coverage.tsv` from CoverM or an existing coverage table
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

Optional audit and comparison outputs may also be written when their flags are enabled:

```text
--pairwise-baseline
  -> coarse/pairwise_clique_contacts.parquet
  -> coarse/pairwise_normalized_contacts.parquet
  -> coarse/bins.pairwise_leiden.tsv
  -> coarse/pairwise_leiden_sweep.tsv

--hyperedge-embedding
  -> coarse/hyperedge_embedding.tsv
  -> coarse/hyperedge_embedding_meta.json

--action-scorer-mode features-only|score
  -> final/refine_action_features.tsv
  -> final/refine_action_scores.tsv  # only with score mode

--embedding-scorer-mode report|veto
  -> final/refine_embedding_scores.tsv
```

## Method summary

`porebin` keeps two contact-evidence routes separate and exposes two coarse clustering methods:

1. **Hypergraph main method**: each Pore-C contact/read is a weighted hyperedge. This is the default binning route.
2. **Pairwise baseline**: each Pore-C hyperedge is deliberately expanded into pairwise clique edges and clustered with Leiden. This route is only an optional comparison and does not feed back into the hypergraph main method.

Coarse discovery is selected with:

```text
--coarse-method spectral   # default: contact hypergraph + feature hypergraph -> spectral embedding -> HDBSCAN
--coarse-method hgvae      # ML route: TNF/coverage + Pore-C hyperedges -> HG-VAE latent -> HDBSCAN
```

### Hypergraph-native contact weight

The default coarse and refine contact weight mode is `hypergraph-native`.

For one Pore-C hyperedge `e`:

```text
q_e       = read/contact quality from contacts.parquet weight
alpha_ie = normalized evidence share for contig i in e
k_eff,e   = 1 / sum_i(alpha_ie^2)
W_e       = q_e / max(k_eff,e - 1, 1)^eta
```

`W_e` is the evidence budget for one hyperedge. `eta` controls how strongly high-effective-order contacts are budgeted; the default is `0.5` and can be changed with `--hypergraph-weight-eta`.

The high-order Pore-C structure is carried by the incidence matrix values `alpha_ie`; `W_e` only controls the total influence of that hyperedge in the contact operator.

The legacy mode `--contact-weight-mode original` is still available for ablation:

```text
W_e = q_e / (k_valid - 1)
```

### Pairwise baseline

`--pairwise-baseline` runs a separate comparison route:

```text
Pore-C hyperedges
  -> raw clique-expanded pairwise graph
  -> length correction + ICE-style balancing
  -> Leiden resolution sweep
```

This pairwise route is intentionally simple and is not mass-conserved. It is meant to show what happens when high-order Pore-C contacts are flattened into ordinary pairwise contacts. Its normalized pairwise weights are not used to define hypergraph weights.

### HG-VAE coarse route

`--coarse-method hgvae` trains an unsupervised feature-anchored hypergraph VAE and clusters the learned latent space with HDBSCAN. `--hyperedge-embedding` can still be used with the default spectral route when only the embedding audit/refine scorer is desired.

For contig `i`:

```text
x_i = [TNF136_i, log1p(coverage_i)]
z_i = Encoder(x_i)
xhat_i = Decoder(z_i)
```

The VAE must reconstruct the TNF/coverage feature vector:

```text
L_rec = mean_i ||x_i - xhat_i||^2
L_kl  = mean_i KL(q(z_i | x_i) || N(0, I))
```

Each Pore-C contact remains a hyperedge. For hyperedge `e`:

```text
mu_e = sum_i alpha_ie z_i
xbar_e = sum_i alpha_ie x_i
d_e  = sum_i alpha_ie ||x_i - xbar_e||^2
g_e  = 1 / (1 + d_e / median_positive_d)
L_hg = sum_e W_e g_e sum_i alpha_ie ||z_i - mu_e||^2
```

`L_rec` makes the ML representation learn TNF and coverage directly. `L_hg` uses high-order Pore-C contacts to pull compatible contigs together in latent space. `g_e` downweights contacts whose member contigs disagree in TNF/coverage, so contact evidence does not blindly override sequence/coverage evidence.

The total training objective is:

```text
L = L_rec + beta * L_kl + lambda * L_hg
```

In HG-VAE mode, the resulting latent matrix is the coarse clustering surface:

```text
Z_hgvae -> HDBSCAN -> coarse/bins.tsv
```

The same `coarse/hyperedge_embedding.tsv` can also be reused by `--embedding-scorer-mode report|veto` during refinement.

### Conservative refine scoring

Refinement still begins with rule-based candidate actions:

```text
split -> reassign -> merge -> recruit
```

The optional action and embedding scorers are conservative audit layers:

- they never revive a rule-rejected action
- in `score` or `veto` mode, they may veto an action already accepted by the rules
- in report-only modes, they write diagnostics without changing assignments

## Recommended workflow

```bash
# prepare BAMs when starting from one aligned BAM
samtools sort -n -o reads.namesorted.bam reads.bam
samtools sort -o reads.coordsorted.bam reads.bam
samtools index reads.coordsorted.bam

# build evidence from a queryname-sorted BAM and CoverM mean-depth from a reference-sorted BAM
porebin evidence \
  --bam reads.namesorted.bam \
  --coverage-bam reads.coordsorted.bam \
  --contigs contigs.fasta \
  --out run_out

# or adopt an existing porebin-compatible coverage table without recomputing coverage
porebin evidence \
  --bam reads.namesorted.bam \
  --coverage-tsv coverage.tsv \
  --contigs contigs.fasta \
  --out run_out

# development fallback without CoverM; not recommended for final benchmarking
porebin evidence \
  --bam reads.namesorted.bam \
  --coverage-method internal \
  --contigs contigs.fasta \
  --out run_out

# run coarse discovery + refine MVP
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --out run_out

# run the optional pairwise Leiden baseline without affecting the hypergraph main method
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --pairwise-baseline \
  --out run_out

# train feature-anchored HG-VAE embeddings and report refine embedding scores
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --hyperedge-embedding \
  --embedding-scorer-mode report \
  --out run_out

# run the complete HG-VAE coarse route
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --coarse-method hgvae \
  --embedding-scorer-mode report \
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

Candidate genome bins from the coarse discovery stage. These are not final bins. With `--coarse-method spectral`, they come from joint spectral embedding plus HDBSCAN. With `--coarse-method hgvae`, they come from HG-VAE latent embedding plus HDBSCAN.

### Pairwise baseline outputs

When `--pairwise-baseline` is enabled, `coarse/bins.pairwise_leiden.tsv` contains the separate pairwise Leiden comparison. `coarse/pairwise_leiden_sweep.tsv` records the Leiden resolution sweep and selected resolution. These files are not used by the hypergraph main method or by refinement.

### `coarse/hyperedge_embedding.tsv`

When `--hyperedge-embedding` is enabled, this file contains feature-anchored HG-VAE latent vectors for contigs. The model learns from TNF/coverage reconstruction and Pore-C hyperedge regularization. These embeddings may be used by the optional refine embedding scorer.

This file is written automatically when `--coarse-method hgvae` is used.

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

### `final/refine_action_features.tsv`

Candidate-action feature table for audit or model training. It is written by default with `--action-scorer-mode features-only`.

### `final/refine_action_scores.tsv`

Action scorer decisions when `--action-scorer-mode score` is used with a model directory.

### `final/refine_embedding_scores.tsv`

Embedding scorer diagnostics when `--embedding-scorer-mode report` or `veto` is used.

### `final/refine_meta.json`

Stage-level counts for suspect bins, split/reassign/recruit candidates, accepted actions, rejected actions, and final unbinned contigs.

## Method boundaries

- coarse discovery does not promote HDBSCAN noise into bins by component-majority postprocessing
- the pairwise Leiden baseline is a separate comparison route and does not feed pairwise weights into the hypergraph main method
- the default hypergraph main method uses only Pore-C hyperedge-native quantities (`q_e`, `alpha_ie`, `k_eff`) for contact weights
- HG-VAE learns TNF/coverage directly and uses Pore-C contacts as hypergraph regularization; in `--coarse-method hgvae`, that latent space directly drives HDBSCAN coarse clustering
- refine is genome-centric and does not use host-centric semantics
- unresolved contigs remain explicit instead of being forced into bins
- SCG veto is enabled by default in refine and requires external `prodigal` and `hmmsearch`

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
