# porebin

`porebin` is a genome-centric Pore-C metagenomic binning tool. It combines Pore-C contact evidence, sequence composition, and coverage to discover candidate genome bins, refine them into final genome bins, and keep unresolved contigs explicit.

The refine redesign, formulas, evidence roles, performance constraints, and
implementation sequence are specified in
[`docs/REFINE_DESIGN.md`](docs/REFINE_DESIGN.md).

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

Run the recommended spectral + HG-VAE embedding-assisted route:

```bash
porebin bin \
  --contigs "$CONTIGS" \
  --contacts "$WORK/evidence_run/evidence/contacts.parquet" \
  --coverage-tsv "$WORK/evidence_run/evidence/coverage.tsv" \
  --coarse-method spectral \
  --hyperedge-embedding \
  --out "$WORK/spectral_hgvae_run"
```

Add `--pairwise-baseline` to the command above when you also want the separate
pairwise Leiden comparison.

Run the pure HG-VAE coarse ablation:

```bash
porebin bin \
  --contigs "$CONTIGS" \
  --contacts "$WORK/evidence_run/evidence/contacts.parquet" \
  --coverage-tsv "$WORK/evidence_run/evidence/coverage.tsv" \
  --coarse-method hgvae \
  --out "$WORK/hgvae_ablation"
```

If you want a fast algorithm-only smoke test without SCG checks, add `--disable-scg` to the `porebin bin` commands. For final benchmarking, keep SCG enabled.

Export recommended final bins:

```bash
porebin export \
  --contigs "$CONTIGS" \
  --bins-refined-tsv "$WORK/spectral_hgvae_run/final/bins.refined.tsv" \
  --unbinned-tsv "$WORK/spectral_hgvae_run/final/unbinned.tsv" \
  --out "$WORK/spectral_hgvae_export"
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

SCG refinement always uses the bundled `marker.hmm` and the fixed 107-marker
order embedded in the package. There is no custom HMM or marker-manifest
override.

SCG evidence generation and refinement have separate ownership:

```text
porebin_genome/evidence/scg/  fixed panel, Prodigal/HMMER, parser, cache
porebin_genome/refinement/    split, merge, recruit, QC, orchestration
```

There is no legacy `porebin_genome/refine/` package. SCG discovery produces
contig-level marker evidence; the refinement engine consumes that evidence
through its unified profile, evaluator, and ordered policy.

The replacement currently contains the compact `ContactIndex`, versioned
incremental `RefineState`, independent HG-VAE/TNF/log-coverage/SCG profiles,
and the shared `Proposal -> ActionEvaluator -> ActionPolicy -> Decision`
interface for `split`, `merge`, and `recruit`. Candidate actions update only
incident hyperedges through exact `delta N` and `delta D` formulas, and profile
evaluation rebuilds only affected bins.

SCG-guided split candidate generation is implemented. For each bin with
duplicated canonical SCGs, the most repeated marker supplies deterministic
HG-VAE k-means seeds. The longest child retains the source bin ID; other
children receive contiguous new IDs. Missing embeddings, identical seeds,
empty children, or unseparated marker seeds produce explicit abstention.
Candidate generation does not inspect Pore-C edges; the shared evaluator later
checks local Pore-C separation and SCG/embedding improvement.

Merge candidate generation is also implemented. It scans the current
hyperedges once, accumulates bin-pair support
`T(a,b)=sum_e r_e m_ea m_eb`, normalizes it by
`sqrt(D[a]D[b]+eps)`, and proposes only mutual-best pairs supported by at
least two hyperedges. Merge objects are complete non-empty bins with no SCG
duplication and non-missing HG-VAE evidence. A merge is accepted only when it
creates no duplicated SCG and the two robust HG-VAE regions overlap. Accepted
mutual-best pairs are bin-disjoint and are committed together in one
versioned assignment/contact/profile transaction.

Recruit candidate generation is implemented for current unbinned contigs
only. Pore-C support from incident hyperedges selects one target, while HG-VAE
distance independently selects the stable bin with the smallest
radius-normalized distance. A proposal exists only when the two targets agree;
contact ties, latent score ties, unstable targets, missing embeddings, and
insufficient support produce explicit abstention.
Accepted recruits are applied one at a time, and later candidates are refreshed
locally after state changes so SCG and bin-profile evidence remain current.

The replacement policy uses four ordered gates: structural validity, SCG
safety, Pore-C support, and HG-VAE compatibility. TNF and coverage remain in
bin profiles and final QC but are not repeated per-action gates. There is no
independent reassign/release stage and no opaque action classifier in the new
mainline. The public CLI runs this replacement policy directly.

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
  -> final/refine_stage_log.jsonl
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
```

## Method summary

`porebin` keeps two contact-evidence routes separate and exposes two coarse clustering methods:

1. **Hypergraph main method**: each Pore-C contact/read is a weighted hyperedge. This is the default binning route.
2. **Pairwise baseline**: each Pore-C hyperedge is deliberately expanded into pairwise clique edges and clustered with Leiden. This route is only an optional comparison and does not feed back into the hypergraph main method.

Coarse discovery is selected with:

```text
--coarse-method spectral   # stable route: contact hypergraph + feature hypergraph -> spectral embedding -> HDBSCAN
--coarse-method hgvae      # experimental ablation: HG-VAE latent -> HDBSCAN
```

The recommended route is spectral coarse plus HG-VAE embeddings for refine:

```text
--coarse-method spectral --hyperedge-embedding
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

The replacement `refinement/ContactIndex` does not expose this ablation
branch: it always caches the hypergraph-native weight shown above.
`--contact-weight-mode original` affects coarse ablation runs only.

### Pairwise baseline

`--pairwise-baseline` runs a separate comparison route:

```text
Pore-C hyperedges
  -> raw clique-expanded pairwise graph
  -> length correction + ICE-style balancing
  -> Leiden resolution sweep
```

This pairwise route is intentionally simple and is not mass-conserved. It is meant to show what happens when high-order Pore-C contacts are flattened into ordinary pairwise contacts. Its normalized pairwise weights are not used to define hypergraph weights.

### HG-VAE embedding evidence

`--hyperedge-embedding` trains an unsupervised feature-anchored hypergraph VAE
and writes latent vectors for replacement refine. In the recommended mainline,
spectral still produces `coarse/bins.tsv`; HG-VAE only supplies similarity
evidence for split, merge, and recruit.

`--coarse-method hgvae` is retained as an experimental ablation that clusters
the HG-VAE latent space with HDBSCAN.

For contig `i`:

```text
x_i = [TNF136_i, log1p(coverage_i)]
z_i = Encoder(x_i)
xhat_i = Decoder(z_i)
```

The VAE part uses two standard terms:

```text
L_feat  = 0.5 MSE(TNF, TNFhat) + 0.5 MSE(cov, covhat)
L_prior = mean_i KL(q(z_i | x_i) || N(0, I))
```

If coverage is not available, `L_feat` falls back to MSE over the available TNF/feature vector.

Each Pore-C contact remains a hyperedge. We use one reusable quantity for a hyperedge's weighted dispersion:

```text
Var_e(Y) = sum_i alpha_ie ||y_i - ybar_e||^2
ybar_e   = sum_i alpha_ie y_i
```

This same definition is used twice:

```text
Var_e(X) = TNF/coverage disagreement inside contact e
Var_e(Z) = latent-space disagreement inside contact e
```

The final contact strength is:

```text
s_e = W_e * c_e
```

where `W_e` is the hypergraph-native contact weight and `c_e` is the feature-compatibility factor:

```text
c_e = 1 / (1 + Var_e(X) / median_positive_VarX)
```

The hypergraph term is:

```text
L_contact = sum_e s_e Var_e(Z) / sum_e s_e
```

During minibatch training, hyperedges are sampled proportional to `s_e`, and the batch loss is the average `Var_e(Z)`. This keeps high-confidence contacts important without multiplying their influence twice.

`L_feat` makes the ML representation learn TNF and coverage directly. `L_contact` uses high-order Pore-C contacts to pull compatible contigs together in latent space. `s_e` prevents contact evidence from blindly overriding sequence/coverage evidence.

The total training objective is:

```text
L = L_feat + beta * L_prior + lambda * L_contact
```

In the experimental HG-VAE coarse ablation, the resulting latent matrix is the
coarse clustering surface:

```text
Z_hgvae -> HDBSCAN -> coarse/bins.tsv
```

In the recommended route, the same `coarse/hyperedge_embedding.tsv` is used
directly by replacement refine for HG-VAE compatibility checks.

### Replacement refine design

The replacement refine mainline is:

```text
SCG-guided split -> conservative merge -> Pore-C/HG-VAE agreement recruit
```

Its action logic is intentionally small:

- duplicated canonical SCGs trigger split diagnosis
- HG-VAE proposes or validates latent compatibility
- original Pore-C hyperedges verify local structural support
- merge and recruit cannot create a new duplicated SCG marker
- missing or contradictory evidence leaves the assignment unchanged

The replacement mainline has no action classifier or secondary embedding
scorer. It records the direct ordered-gate evidence for every considered
action.

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

# run the recommended spectral + HG-VAE embedding-assisted mainline
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --coarse-method spectral \
  --hyperedge-embedding \
  --out run_out

# run the optional pairwise Leiden baseline without affecting the hypergraph main method
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --coarse-method spectral \
  --hyperedge-embedding \
  --pairwise-baseline \
  --out run_out

# fast spectral-only ablation without HG-VAE refine evidence
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --coarse-method spectral \
  --no-hyperedge-embedding \
  --out run_out

# run the pure HG-VAE coarse ablation
porebin bin \
  --contigs contigs.fasta \
  --contacts run_out/evidence/contacts.parquet \
  --coverage-tsv run_out/evidence/coverage.tsv \
  --coarse-method hgvae \
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

Candidate genome bins from the coarse discovery stage. These are not final
bins. In the recommended route, they come from `--coarse-method spectral`:
joint spectral embedding plus HDBSCAN. With the experimental
`--coarse-method hgvae` ablation, they come from HG-VAE latent embedding plus
HDBSCAN.

### Pairwise baseline outputs

When `--pairwise-baseline` is enabled, `coarse/bins.pairwise_leiden.tsv` contains the separate pairwise Leiden comparison. `coarse/pairwise_leiden_sweep.tsv` records the Leiden resolution sweep and selected resolution. These files are not used by the hypergraph main method or by refinement.

### `coarse/hyperedge_embedding.tsv`

When `--hyperedge-embedding` is enabled, this file contains feature-anchored
HG-VAE latent vectors for contigs. The model learns from TNF/coverage
reconstruction and Pore-C hyperedge regularization. In the recommended route,
replacement refine uses these vectors directly while `coarse/bins.tsv` remains
spectral.

This file is written automatically when `--coarse-method hgvae` is used for
the experimental ablation.

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

Accepted, rejected, and abstained split, merge, and recruit records. Each row
keeps direct gate evidence in `note`; there is no combined confidence score.
For recruit actions, `note` records the Pore-C target, the HG-VAE target,
whether the two targets agree, the raw latent distance, the
radius-normalized HG-VAE score, and the radius gate used by the policy.

### `final/refine_stage_log.jsonl`

Immediately flushed stage records for input validation, SCG, initialization,
split, merge, recruit, finalization, completion, and errors. This is a
progress log only. Refine writes no checkpoint or resumable assignment state.

### `final/refine_meta.json`

Stage-level engine identity, candidate/action counts, evidence settings, and
final unbinned counts. The `embedding_evidence` block states that HG-VAE is
used as similarity evidence, records the recruit score formula, and names the
split, merge, and recruit rules used in action notes.

## Method boundaries

- coarse discovery does not promote HDBSCAN noise into bins by component-majority postprocessing
- the pairwise Leiden baseline is a separate comparison route and does not feed pairwise weights into the hypergraph main method
- the default hypergraph main method uses only Pore-C hyperedge-native quantities (`q_e`, `alpha_ie`, `k_eff`) for contact weights
- HG-VAE learns TNF/coverage directly and uses Pore-C contacts as hypergraph regularization; in the recommended route, it is refine similarity evidence rather than the coarse clustering source
- `--coarse-method hgvae` is retained as an experimental direct-clustering ablation
- replacement refine uses only split, merge, and recruit with ordered SCG, Pore-C, and HG-VAE evidence
- candidate actions must use incident-edge indexes instead of rescanning all hyperedges
- unresolved contigs remain explicit instead of being forced into bins
- SCG veto is enabled by default in refine and requires external `prodigal` and `hmmsearch`

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
