# Librarian Short-Term Observations

## Run: 2026-08-02 — post-commit audit after interior SACR-C / C8 / C5 builds

### What went stale and the pattern

**SPEC.md leading-summary label vs body** (the fix made this run):
The engine row's table cell opens with a compact module list (one parenthetical
per file). The C8 build updated the Born rung _body_ from "Born far-annulus
carrier" to "BORN EXTERIOR RUNG" / "exterior-to-caustic region (rho > 1)" but
left the _opening summary phrase_ unchanged: "the Born far-annulus carrier
(`_born.py`)". The body and the fragment description both used the new
language, so it looked complete until reading the literal opening sentence.
**Pattern to watch**: the opening module-list phrase is a _separate_ text
token from the long description — both must be updated when a concept is
renamed. Future renames of `_born.py`'s role should touch both locations.
Fixed: "Born exterior rung carrier (`_born.py`)" in both places.
Fragment: `.claude/spec/spec_changelog.d/2026-08-02_c8-born-summary-label.md`

### Surfaces checked and found clean

- DATA_CONTRACTS.yaml `certified_ppgo_map`: consumer list (likelihood.py,
  surrogate_training.py, ppgo_map.py) is current. The `annulus_rho ->
  caustic_rho` rename in ppgo_map.py doesn't touch the contract (it's a helper,
  not a disk-artifact accessor). No new disk artifact from `born_residual_chart`
  (BornResidualChart is in-memory; wiring is in place but no trained artifact
  shipped yet).
- docs/source/overview.rst: lensing section pitches at public API level
  (ChangRefsdalChannels, LensedWaveformGenerator, LensedRelativeBinningLikelihood);
  no internal carrier/rung names cited. No change needed.
- surrogate_training.py: `interior_w_nodes_per_decade = 15` is a training
  config constant, implementation detail — not at SPEC architecture level.
- ppgo_map.py: `_EXTRAP_ALPHA_MAX` 1.5→3.5 and envelope smoothing are internal
  to _measure_cell. Not in SPEC.
- spec_changelog.d/contracts_changelog.d: no stub markers from pre-commit hooks.
- sync_derived_docs.py warnings: test-only consumers for lens_amplification_surrogate
  (stay off by convention).

### Fragile cross-references to watch next run

- SPEC.md engine row leading summary phrase and body are now consistent, but
  any future Born rung concept rename (e.g. when BornResidualChart gets a
  trained artifact and the "pending training" status changes) will need to
  update both places again.
- "residual chart pending training" is a status sentence in SPEC.md that will
  go stale the moment the Born residual chart is trained and attached.
