# Foreman Short-Term Notes

- INS-3-001 fix: surrogate_training.py's train() computed per-parity
  `dropped` sliver lists (from stable_gamma_bands) but only wrote them into
  parity_reports (the JSON training report), never into the surrogate's
  serialized provenance. Fixed by threading a flat `all_dropped_slivers`
  list (accumulated across both parity loop iterations) into
  `_build_provenance(box, config, charts, dropped_gamma_slivers=...)`,
  which now stores `provenance['dropped_gamma_slivers']` as flat
  `[[lo, hi], ...]` pairs -- matching the shape `_normalize_slivers` and
  `dropped_slivers_from_training_report` already expect. No change needed
  on surrogate_census.py's read side (`_dropped_slivers_from` already
  defaulted to provenance) -- only updated its stale docstring NOTE that
  described the now-fixed discrepancy as still-open. Verified via
  ast.parse, import + signature check, and a manual
  `_dropped_slivers_from(FakeSurrogate, None)` smoke probe; existing
  Serialization/MultiChart provenance round-trip tests (6) still pass
  unchanged since that fixture builds provenance by hand, not via train().
- INS-1-003 handled per explicit driver decision (item 3): "ACCEPTED AS
  DEFERRED — no build action." Declined per Librarian-owned convention.
- INS-1-001 (commissioning a Test Developer to author
  test_lensing_surrogate_census.py) is NOT trivial-fix scope for
  Foreman-Lite (requires coordinating another agent) and was not in the
  top-level findings list handed to me — left untouched/out of scope,
  flagging for orchestrator routing to a Test Developer agent.