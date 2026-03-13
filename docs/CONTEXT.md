# porebin Context

This file is a short entrypoint.

The authoritative repository reference is:

- [`docs/PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md)

## Current status

- Main evidence source:
  BAM-derived `contacts.parquet`
- Canonical contact-row semantics:
  `porebin/contact_hypergraph.py`
- Main coarse path:
  joint contact-feature hypergraph spectral clustering
- Main refine path:
  `refine_bins_parquet(...)` with soft-gated host-assignment inference
- Main export path:
  `porebin/export.py`

## Main CLI path

1. `porebin bam2contacts`
2. `porebin build`
3. `porebin cluster --method spectral`
4. `porebin refine`
5. `porebin export`

Or:

1. `porebin run-bam`

## Legacy paths still present

- PPL `.contacts` normalization in `porebin/normalize.py`
- pairwise baseline in `porebin/pairwise_baseline.py`
- legacy refine path `refine_bins(...)` in `porebin/refine.py`

For the current mathematical definitions, file inventory, CLI interfaces, and legacy-path details, use:

- [`docs/PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md)
