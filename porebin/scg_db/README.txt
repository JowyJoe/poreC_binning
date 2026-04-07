Place the built-in SCG resources for refine here.

Expected files:
  - manifest.json
  - one or more *.hmm files referenced by manifest.json

Recommended manifest fields:
  - marker_set_id
  - db_version
  - marker_hmm
  - expected_markers

Current bundled resource:
  - core_bacterial_scg.hmm

If marker_hmm is omitted from manifest.json, the loader falls back to:
  - marker.hmm, if present
  - otherwise the only *.hmm file in this directory, if there is exactly one
