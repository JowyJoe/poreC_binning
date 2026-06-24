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
  refinement/
  qc/
  export/
```

### `evidence/`

Builds canonical `contacts.parquet` and `coverage.tsv` from a queryname-sorted BAM.

`evidence/scg/` owns the fixed 107-marker panel and the complete SCG evidence
pipeline:

```text
contigs.fasta
-> Prodigal ORFs
-> hmmsearch against bundled marker.hmm
-> canonical marker aliases
-> per-contig marker presence and ORF counts
-> contig_scg_index.json
```

This layer does not decide split, merge, or recruit actions.

### `coarse/`

Implements the default coarse discovery mainline:

- hypergraph-native contact incidence construction
- hypergraph-native contact weighting from `q_e`, `alpha_ie`, and `k_eff`
- TNF136 + coverage features
- feature kNN incidence
- joint operator construction
- spectral embedding
- HDBSCAN clustering

Outputs are candidate genome bins only.

The default hypergraph contact weight is:

```text
k_eff,e = 1 / sum_i(alpha_ie^2)
W_e     = q_e / max(k_eff,e - 1, 1)^eta
```

`W_e` is the evidence budget for one Pore-C hyperedge. The high-order contact structure itself remains in the incidence values `alpha_ie`.

`coarse/` also contains an optional pairwise baseline:

- raw clique expansion of Pore-C hyperedges into pairwise contacts
- length correction and ICE-style balancing
- Leiden resolution sweep

The pairwise route is a comparison baseline only. It does not feed normalized pairwise weights back into the hypergraph mainline.

`coarse/` now has one recommended stable method and one experimental ablation:

```text
--coarse-method spectral
  recommended stable route:
  contact hypergraph + feature hypergraph -> spectral embedding -> HDBSCAN

--coarse-method hgvae
  experimental direct-clustering ablation:
  TNF/coverage + Pore-C hyperedges -> HG-VAE latent -> HDBSCAN
```

The recommended mainline is:

```text
--coarse-method spectral --hyperedge-embedding
```

In this route, spectral writes `coarse/bins.tsv`, and feature-anchored HG-VAE
embeddings are trained only as refinement similarity evidence.

HG-VAE uses:

```text
x_i = [TNF136_i, log1p(coverage_i)]
z_i = Encoder(x_i)
L = L_feat + beta * L_prior + lambda * L_contact
```

where `L_feat` reconstructs TNF/coverage with block-balanced TNF and coverage losses, `L_prior` regularizes the VAE latent space, and `L_contact = sum_e s_e Var_e(Z) / sum_e s_e` keeps contigs from reliable Pore-C hyperedges close in latent space. `Var_e` is the alpha-weighted dispersion inside one hyperedge, and `s_e` combines contact strength with TNF/coverage compatibility.

The pure `--coarse-method hgvae` route remains available for ablation, but it
is not the recommended default because current CheckM2 results show many more
low-quality bins than the spectral mainline.

### `refinement/`

Implements the replacement refine architecture:

- SCG-guided split
- deterministic duplicate-marker seed selection and HG-VAE seeded k-means
- one-pass mutual-best contact-supported merge
- version-safe Pore-C/HG-VAE agreement recruitment of unbinned contigs
- exact action-local contact deltas
- synchronized profile updates and rollback
- ordered structural, SCG, Pore-C, and HG-VAE gates
- abstention when evidence is missing or contradictory
- immediate JSONL stage logging without checkpoint state

The replacement deliberately has no independent reassign/release stage and no
opaque action classifier. TNF and coverage remain in profiles and final QC,
while HG-VAE is the action-level representation in the recommended
spectral-plus-HG-VAE route.

Split generation is already implemented in `refinement/split.py`. It selects
the most repeated canonical SCG, uses its carrier contigs as explicit HG-VAE
k-means initial centers, keeps the longest child under the source bin ID, and
returns an immutable proposal. Contact and acceptance logic remain in the
shared evaluator and policy.

Merge candidate generation is implemented in `refinement/merge.py`. One
intentional hyperedge scan accumulates raw pair support, independent supporting
edge counts, and contact-mass-normalized support. Only mutual-best eligible
bins become proposals. The evaluator then checks each proposal through local
source-bin incidences, SCG safety, and overlap of the two robust HG-VAE regions.
Accepted disjoint decisions are combined into one versioned contact/profile
transaction, preserving the one-scan stage semantics.

Recruit candidate generation is implemented in `refinement/recruit.py`.
Only current unbinned contigs are considered. Pore-C selects a best target from
incident hyperedges, HG-VAE independently selects the stable bin with the
smallest radius-normalized distance, and only agreement creates a proposal.
Ties abstain. Accepted recruits are applied sequentially; later ready contigs
are refreshed locally after a state change so SCG, contact, centroid, and
radius evidence cannot become stale.

SCG discovery uses the bundled `evidence/scg/marker.hmm` panel. Its 111 raw
profiles are normalized to the 107-marker order embedded in
`evidence/scg/panel.py`. Marker aliases are canonicalized before
contig-level deduplication. Prodigal, hmmsearch, and parser outputs have
separate fingerprints. The refine API exposes no custom HMM or marker-order
override: all runs use the bundled panel. A changed bundled HMM digest reuses
valid ORFs, while a parser or alias-schema change reuses valid domtblout
results.

The refinement contact index always uses the hypergraph-native reliability
definition shared with coarse discovery.

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
- `final/refine_stage_log.jsonl`
- `final/refine_meta.json`

`final/refine_actions.tsv` keeps action evidence in the `note` JSON. Recruit
notes include Pore-C target support, HG-VAE radius-normalized target score,
target agreement, and the final embedding radius check. `final/refine_meta.json`
contains the run-level `embedding_evidence` block describing the same rules.

Optional outputs:

- `coarse/bins.pairwise_leiden.tsv`
- `coarse/pairwise_leiden_sweep.tsv`
- `coarse/hyperedge_embedding.tsv`

## Replacement Refine Semantics

- actions are limited to split, merge, and recruit
- split is triggered by duplicated canonical SCGs
- merge and recruit cannot create a new duplicated SCG marker
- recruit requires Pore-C support and HG-VAE compatibility
- decisions use ordered gates rather than a combined score
- low-confidence contigs remain explicit in `unbinned.tsv`
- candidate evaluation cannot rescan all hyperedges

See [REFINE_DESIGN.md](REFINE_DESIGN.md) for formulas and stage contracts.

## Scope of this branch

This branch keeps the genome-centric hypergraph mainline plus optional audit/comparison routes. Legacy host-centric code, legacy CLI paths, compatibility bridges, and historical refactor notes are not part of the active public surface here.
