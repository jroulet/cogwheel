# Foreman-Lite Short-Term Observations

- INS-1-002: yet another Librarian-tagged spec-sync finding (SPEC.md row
  ~55, COVERAGE_DESIGN.md retired-estimator narrative, plus a stale
  test-module docstring phrase in test_lensing_caustic_cusps.py ~line 106)
  dispatched to Foreman-Lite despite its own text saying "not this build's
  job" / "Do not edit canonical surfaces in a WP" / "Librarian reconciles
  SPEC.md / COVERAGE_DESIGN.md". Declined per standing rule, zero files
  touched. This is the same recurring class as INS-1-001/INS-1b-004 —
  the orchestrator pre-filter for stripping "-> Librarian"-tagged findings
  before Foreman-Lite dispatch still does not exist. Recommend once more,
  loudly: add that pre-filter upstream rather than relying on per-pass
  declines.
