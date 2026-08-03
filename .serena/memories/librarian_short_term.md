# Librarian Short-Term Observations

## Run: 2026-08-03 — build audit for step-2/step-4 script rewrites

### Scope

Work packages: "Update step 2 script parameters and fix step 8 allowlist" and
"Rewrite step 4 script for geometric tiling-coverage verification".

Changed files:
- `scripts/measure_tube_fraction.py` — parameter grid update (brief specs: 5+4 gammas,
  12-step linspace fractions, n_gamma=1/n_u=6/n_theta=6/w_nodes_per_decade=6)
- `scripts/measure_far_zone_crossover.py` — full rewrite: Born-carrier-accuracy measurement
  → geometric tube+far-field tiling coverage verification at C8 boundary
- `cogwheel/tests/test_lensing_part0_mechanical.py` — new mechanical tests
  (test_no_retired_names_in_live_docs, TestNoDocstringAbsorberLanguage, self-falsification companions)
- Agent state files and memories (no doc action)

### What was stale

**Nothing.** All changed files are in `scripts/` or `cogwheel/tests/`:
- `scripts/` changes: not `cogwheel/` source modules, so no module list, API, or overview staleness.
- `cogwheel/tests/` changes: test-only → skip entirely per triage rules.
- No new serialization artifacts in `scripts/`.
- No `pyproject.toml`/`environment.yaml` changes.

`sync_derived_docs.py` ran cleanly. The 4 test-file-only consumer warnings
(`LensAmplificationSurrogate.load` in `test_lensing_surrogate.py`) are pre-existing and
excluded by convention (test-only callers stay off DATA_CONTRACTS.yaml consumer lists).
Script's "some issues auto-fixed" produced zero git diff — confirmed internal state flush.

### Files changed in this run

- `.serena/memories/librarian_short_term.md` — this file only (no doc edits needed)

### Pattern

`scripts/` rewrites (even large ones like measure_far_zone_crossover.py +188 lines) don't
touch doc surfaces unless they introduce new serialization artifacts or change the cogwheel
public API. A complete scripts/ rewrite that stays in `scripts/` is a legitimate no-op for
the librarian.

### Fragile cross-references (carried forward)

- F060 cites `cogwheel/lensing/surrogate.py` for the far-field chart d-axis — watch for
  module reorganization.
- `_DD_PRODUCT_MARGIN = 58.0` still duplicated in surrogate.py and surrogate_training.py.
- `lensing_remaining_coverage_gaps.md` items 2 and 3 remain open (`[→ spec]` and
  `[research]`) — watch commits touching ppGO paths or interior cell certification.
- The `[housekeeping]` items in lensing_remaining_coverage_gaps.md (sidecar callback
  and xdist tree-gate) have no `[→ spec]` tag — they don't drive doc updates when closed.
