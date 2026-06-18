# Porebin Refine Design Contract

Updated: 2026-06-12

This is the authoritative design for the replacement refine engine in
`porebin_genome/refinement/`. Read this file before changing refine behavior.
Historical discussions are not implementation contracts.

## 1. Implementation Boundary

```text
porebin_genome/evidence/scg/  fixed SCG evidence generation
porebin_genome/refinement/    active refinement engine
```

The refinement engine contains:

- consumption of contig-level evidence from the fixed 107-marker SCG panel;
- a compact Pore-C `ContactIndex`;
- exact incremental contact coherence;
- versioned assignment state with apply and rollback;
- HG-VAE, TNF, coverage, and SCG bin profiles;
- one proposal/evaluator/policy interface for `split`, `merge`, and `recruit`;
- deterministic SCG-guided HG-VAE split candidate generation;
- one-pass mutual-best merge candidate generation;
- version-safe recruitment of unbinned contigs;
- the `split -> merge -> recruit` orchestrator;
- final assignment, QC, action, metadata, and stage-log writers.

The public CLI calls `porebin_genome.refinement.orchestrate`. SCG discovery is
not an action implementation: it runs Prodigal and hmmsearch, canonicalizes the
fixed panel, and writes contig-level evidence. All action logic lives only in
`refinement/`. There is no legacy refine package or compatibility route.

## 2. Design Principles

1. Original Pore-C contacts remain hyperedges.
2. Pairwise clique expansion is a separate Leiden comparison only.
3. Pairwise weights never define or modify hyperedge weights.
4. Refine uses three actions only: `split`, `merge`, and `recruit`.
5. SCG protects biological purity and initiates split diagnosis.
6. Pore-C supplies local structural support.
7. HG-VAE supplies the unsupervised similarity representation.
8. TNF and coverage remain available for profiles and final QC, but are not
   repeated action gates.
9. Decisions use ordered gates, not a weighted sum or opaque classifier.
10. Missing or contradictory evidence causes rejection or abstention.
11. No bin or action candidate may rescan every hyperedge.
12. Every reported quantity must have one direct algorithmic purpose.

## 3. Mainline

```text
coarse assignments + Pore-C index + HG-VAE embedding + SCG index
    |
    v
initialize incremental state and bin profiles
    |
    v
SCG-guided split of contaminated bins
    |
    v
conservative merge of contact-supported compatible bins
    |
    v
recruit unbinned contigs when Pore-C and HG-VAE agree
    |
    v
final QC, action log, assignments, and unresolved contigs
```

There is no independent `reassign` or `release` stage.

- A mixed coarse bin is corrected by `split`.
- An over-split genome is corrected by `merge`.
- An uncertain coarse assignment is not repeatedly moved among bins.
- An unbinned contig is added only by `recruit`.

This contraction keeps the algorithm conservative and prevents a large
candidate-action search from dominating runtime.

## 4. One-Time Contact State

For hyperedge `e`, member `i`, and bin `b`:

```text
alpha[e,i] = normalized evidence share of contig i in edge e
q[e]       = upstream read/contact weight
k_eff[e]   = 1 / sum_i(alpha[e,i]^2)
r[e]       = q[e] / max(k_eff[e] - 1, 1)^eta
```

`r[e]` is calculated once when the contact index is built.

The mass of edge `e` assigned to bin `b` is:

```text
m[e,b] = sum_{i in e, assignment(i)=b} alpha[e,i]
```

Bin contact coherence is:

```text
N[b] = sum_e r[e] * m[e,b]^2
D[b] = sum_e r[e] * m[e,b]
C[b] = N[b] / (D[b] + eps)
```

Purpose:

- `m[e,b]` stores how much of one high-order contact belongs to a bin.
- `D[b]` is the reliable contact mass received by the bin.
- `N[b]` rewards contacts concentrated inside the bin.
- `C[b]` is an auditable contact-coherence summary.

The index stores `contig -> incident edge IDs`. Evaluating a proposal visits
only the union of incident edges of changed contigs. It computes exact
`delta N` and `delta D`, then applies or rolls back the change transactionally.

## 5. SCG Quantity

Let `n[b,g]` be the number of distinct contigs in bin `b` carrying canonical
marker `g`. Multiple ORFs for the same marker on one contig count once.

```text
R[b] = sum_g max(n[b,g] - 1, 0)
```

`R[b]` is the SCG duplicate burden.

Purpose:

- `R[b] > 0` identifies a biologically suspicious bin;
- a split should reduce total duplicate burden;
- merge and recruit must not create new duplicate markers;
- marker absence is uninformative, never positive evidence.

The bundled `evidence/scg/marker.hmm` is the only HMM panel. Its 111 raw
profiles are canonicalized to the fixed embedded 107-marker order. There is
no user-supplied marker manifest or custom marker-order interface.

## 6. HG-VAE Quantities

Let `z[i]` be the normalized HG-VAE embedding of contig `i`. For bin `b`:

```text
mu[b]  = mean_{i in b} z[i]
d[i,b] = ||z[i] - mu[b]||_2
med[b] = median_{i in b} d[i,b]
mad[b] = median_{i in b} |d[i,b] - med[b]|
rho[b] = med[b] + 3 * mad[b]
```

Purpose:

- `mu[b]` is the latent center of a bin;
- `med[b]` is its robust latent dispersion;
- `rho[b]` is the direct recruit compatibility radius;
- before/after median dispersion protects split from making child groups less
  coherent;
- merge compares center distance with `rho[a] + rho[b]`.

The decoder reconstruction is used while training HG-VAE. Refine uses the
learned latent vectors `z`, not reconstructed TNF/coverage values.

## 7. Pore-C Contig Support

For contig `i` and bin `b`:

```text
S(i,b) = sum_{e containing i}
         r[e] * alpha[e,i] * m[e,b excluding i]

P(i,b) = S(i,b) / (sum_c S(i,c) + eps)
```

Purpose:

- `S(i,b)` is direct high-order contact support from `i` to `b`;
- `P(i,b)` is the share of assigned-bin support pointing to `b`;
- both are computed through `i`'s incident-edge list only.

## 8. Ordered Decision Gates

Every proposal passes through the same order:

```text
1. structural validity
2. SCG safety
3. Pore-C support
4. HG-VAE compatibility
```

There is no combined action score.

- Structural invalidity is a rejection.
- A newly created SCG duplicate is a rejection.
- Insufficient contact or embedding evidence is an abstention.
- Only a proposal passing all required gates is accepted.

TNF and coverage are still computed for final bin profiles and scientific
audit. They are not additional per-action vetoes because HG-VAE already learns
from them and repeated gates make the logic redundant and difficult to tune.

## 9. Split

### 9.1 Candidate bins

Only bins with `R[b] > 0` enter the primary split queue. This makes SCG the
diagnostic trigger instead of testing every bin.

### 9.2 Number and seeds

Choose the duplicated marker with the largest distinct-contig count:

```text
k = max_g n[b,g]
```

The `k` contigs carrying that marker are deterministic seed candidates. Ties
between markers are resolved by canonical marker ID. If required embeddings
are missing or a valid child cannot be formed, abstain.

Run deterministic seeded k-means in HG-VAE space inside this bin. It assigns
non-marker contigs without introducing a second ML model.

Implementation rules:

1. the canonical marker with the largest distinct-contig count is selected;
2. marker ties are resolved by canonical marker ID;
3. carrier contigs are ordered by integer contig ID;
4. their normalized HG-VAE vectors are passed as explicit Lloyd k-means
   initial centers with `n_init=1`;
5. every source-bin contig must have an HG-VAE embedding;
6. identical seed vectors, an empty child, or marker seeds that finish in the
   same child cause explicit abstention;
7. the child with greatest total contig length keeps the source bin ID;
8. a retained-child tie is resolved by smallest contig ID;
9. remaining children are ordered by smallest contig ID and receive the next
   contiguous bin IDs;
10. the generator returns only changed contigs in the `Proposal` and never
    mutates assignment state.

The generator records marker ID, seed contigs, child memberships, k-means
inertia, iteration count, and a stable reason code. Inertia is diagnostic only;
it is not an acceptance score.

### 9.3 Contact separation

For the proposed child bins, calculate from affected local edges:

```text
within = sum_e r[e] * sum_child m[e,child]^2
cross  = sum_e r[e] * sum_{a<b} m[e,a] * m[e,b]
Qsplit = within / (within + 2 * cross + eps)
```

`Qsplit >= 0.5` means within-child support is at least as strong as the
cross-child term under this normalization.

### 9.4 Accept split

A split is accepted only when:

1. every child satisfies minimum contig count and total length;
2. total SCG duplicate burden decreases;
3. `Qsplit >= 0.5`;
4. aggregate HG-VAE median dispersion does not worsen.

The purpose chain is direct:

```text
SCG detects contamination
-> HG-VAE proposes coherent children
-> Pore-C verifies structural separation
-> SCG confirms biological improvement
```

## 10. Merge

Generate candidate bin pairs in one pass over hyperedges:

```text
T(a,b) = sum_e r[e] * m[e,a] * m[e,b]
```

Also record the number of supporting hyperedges:

```text
L(a,b) = number of hyperedges with positive mass in both a and b
```

Raw support favors bins with more total contact mass, so candidate ranking
uses one direct normalization:

```text
A(a,b) = T(a,b) / sqrt(D[a] * D[b] + eps)
```

`D` is the contact-mass denominator already cached for coherence. No new
latent quantity is introduced.

Merge objects are complete, current, non-empty bins. A bin is eligible only
when:

1. its SCG duplicate burden is zero;
2. every member has an HG-VAE embedding;
3. its embedding centroid and robust radius are available.

Unbinned contigs, empty bins, SCG-suspect bins, and bins without complete
embedding evidence are not merge objects.

Candidate generation keeps a pair only when:

1. `T(a,b) > 0`;
2. `L(a,b) >= 2`;
3. `b` is the highest-`A` neighbor of `a`;
4. `a` is the highest-`A` neighbor of `b`.

Normalized-support ties are resolved by smaller neighbor bin ID. Mutual-best
pairs cannot share a bin, so the number of proposed pairs is bounded by half
the number of eligible bins.

For each pair, the bin with larger total contig length keeps its bin ID. A
length tie keeps the smaller bin ID. Every contig in the source bin appears in
the immutable merge proposal.

HG-VAE merge compatibility uses the existing robust bin regions:

```text
d(a,b) = ||mu[a] - mu[b]||_2
```

The regions are compatible when:

```text
d(a,b) <= rho[a] + rho[b]
```

This replaces the earlier zero-worsening rule. Two valid fragments may have a
larger dispersion after union even when they occupy compatible latent regions;
requiring their robust regions to touch is the more direct merge question.

A merge is accepted only when:

1. the pair has positive support from at least two hyperedges;
2. the merged bin creates no new duplicated SCG marker;
3. the two HG-VAE robust regions overlap;
4. the output bin satisfies structural bounds.

This is intentionally conservative. Merge repairs clear over-splitting; it
does not search broadly for ways to maximize bin size.

All candidates from one scan share one state version. They are evaluated
against that same snapshot. Accepted, bin-disjoint decisions are combined by
`prepare_merge_batch()` and committed once through `apply_merge_batch()`.
Rollback restores both assignment/contact state and profiles. The orchestrator
must use this transaction instead of sequentially applying stale proposals or
repeating the global scan once per candidate.

## 11. Recruit

Recruit operates only on current unbinned contigs:

```text
assignment[i] == -1
```

It never moves a contig that already belongs to a bin.

A target bin is stable enough for recruitment only when:

1. it is non-empty;
2. its SCG duplicate burden is zero;
3. every member has an HG-VAE embedding;
4. its HG-VAE centroid and robust radius are available.

For each unbinned contig with an HG-VAE embedding, compute `S(i,b)` only from
its incident edges.

Let:

```text
b_contact = argmax_b P(i,b)
b_latent  = argmin_b d(i,b)
```

Generate a recruit proposal only when `b_contact == b_latent`.

Candidate-generation rules:

1. the Pore-C best target is selected by raw `S(i,b)`;
2. an exact support tie causes abstention rather than bin-ID tie breaking;
3. the best Pore-C target itself must be a stable target bin;
4. the target must be supported by at least two incident hyperedges;
5. `P(i,b_contact) >= 0.5`;
6. the HG-VAE nearest target is calculated exactly over stable bin centroids;
7. an exact nearest-distance tie causes abstention;
8. Pore-C and HG-VAE targets must be identical.

The generator records raw support, runner-up support, support share, margin,
supporting edge count, latent distance, target radius, both target IDs, and a
stable reason code. These values are evidence records, not a combined score.

The radius check remains in the shared policy rather than the generator. This
keeps the responsibilities separate:

```text
generator: do the two independent target selectors agree?
policy:    is the agreed target biologically and geometrically safe?
```

Accept it only when:

1. target support uses at least two incident hyperedges;
2. `P(i,b) >= 0.5`;
3. target support is not below the runner-up support;
4. adding the contig does not create an SCG duplicate;
5. `d(i,b) <= rho[b]`;
6. target contact coherence does not decrease.

The logic is:

```text
Pore-C selects the structurally supported bin
-> HG-VAE independently selects the nearest feature/contact representation
-> agreement produces one candidate
-> SCG prevents biological contamination
```

Initially ready candidates are ordered deterministically by:

```text
target share descending
-> supporting edge count descending
-> contig ID ascending
```

Recruit changes a target profile, so proposals cannot all be applied from one
old snapshot. `run_recruitment_pass()` generates the initial list once. If an
earlier acceptance changes the state version, it refreshes only the current
contig through its incident edges and current bin centroids before evaluation.
Initially abstained contigs are not revisited in the same conservative pass.
Thus the pass remains linear in tested local incidences and never performs a
global hyperedge scan.

## 12. State Updates

Evaluation is side-effect free:

```text
Proposal
-> ActionEvaluator
-> ActionEvidence
-> ActionPolicy
-> Decision
```

An accepted decision is applied as one synchronized transaction:

1. apply contact `ActionDelta`;
2. apply affected bin-profile update;
3. increment both state versions.

Rollback occurs in reverse order. Stale proposals abstain rather than being
silently reevaluated against a changed assignment.

## 13. Performance Contract

Forbidden in candidate loops:

```python
for edge in store.edges:
    ...
```

Required access patterns:

- split: edges incident to members of the suspect bin;
- merge generation: one global edge pass per merge stage;
- merge evaluation: edges incident to the two bins;
- recruit: edges incident to the unbinned contig;
- profile update: members of affected bins only.

Expected shape:

```text
one-time initialization: O(total incidences)
split work:              O(incidences of suspect bins)
merge generation:       O(total incidences)
recruit work:            O(incidences of tested unbinned contigs)
```

The number of candidates must never multiply the full hyperedge count.

## 14. Observability And Outputs

The replacement orchestrator writes:

- stage name and elapsed time;
- number of suspect bins and candidates;
- accepted/rejected/abstained counts;
- local edges visited;
- state version;
- stage-complete marker.

The log is created when refine starts and flushed after every record:

```text
final/refine_stage_log.jsonl
```

Its stage order is:

```text
validate_inputs -> scg -> initialize -> split -> merge -> recruit -> finalize
```

An exception writes both `stage_error` and `run_error` before it is raised.
The log contains progress and evidence counters only. It never contains a
serialized assignment, candidate cache, or recoverable algorithm state.

There are deliberately no refine checkpoints, resume tokens, or partial
assignment files. An interrupted refine run starts again from coarse
assignments. Existing Prodigal/HMMER output reuse is an evidence cache, not a
refine checkpoint.

Required final outputs remain:

```text
final/bins.refined.tsv
final/unbinned.tsv
final/bin_qc.tsv
final/refine_actions.tsv
final/refine_stage_log.jsonl
final/refine_meta.json
```

The replacement has no action classifier and does not write action-feature,
action-score, or embedding-score tables.

## 15. Implementation Sequence

Completed:

1. fixed SCG panel and cache semantics;
2. compact contact index and local incidence queries;
3. exact incremental assignment state;
4. independent bin evidence profiles;
5. contracted three-action evaluator and ordered policy;
6. deterministic SCG-guided HG-VAE split candidate generation;
7. one-pass mutual-best merge candidate generation and direct HG-VAE region
   compatibility, including one synchronized transaction for accepted
   disjoint merges;
8. Pore-C/HG-VAE agreement recruit generation with version-safe local refresh.
9. observable orchestrator and final output writers without checkpoints;
10. complexity guards and replacement orchestrator end-to-end tests;
11. public CLI switch to `refinement/`.
12. SCG discovery and fixed panel moved to `evidence/scg/`;
13. legacy refine package, scorers, reassign route, and legacy-only tests
    removed.

## 16. Required Tests

- incremental deltas equal brute-force recomputation on small fixtures;
- proposal evaluation performs zero full-edge scans;
- split reduces duplicate burden on synthetic mixed bins;
- split generation is deterministic and performs no contact scan;
- missing embeddings and indistinguishable marker seeds abstain explicitly;
- merge generation performs exactly one full edge scan;
- merge candidates are deterministic, mutual-best, and bin-disjoint;
- merge evaluation performs no full edge scan;
- accepted disjoint merges apply and roll back as one state version;
- merge rejects new duplicated markers and incompatible HG-VAE regions;
- recruit considers only current unbinned contigs and stable targets;
- recruit performs no full edge scan;
- recruit abstains on contact ties, latent ties, and target disagreement;
- recruit rejects newly duplicated markers and points outside `rho[b]`;
- later recruit candidates refresh locally after an accepted state change;
- runtime counters scale with local incidences, not `E * candidates`;
- interrupted stages leave a readable stage/error log;
- no checkpoint or partial assignment is written;
- final output contracts remain stable after the CLI switch.
