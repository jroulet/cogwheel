# Foreman-Lite Short-Term Observations

- INS-2-001 (2026-08-14, test_lensing_saddle_rho_guards.py): docstring-only
  fix. CensusBandSplitMirrorIntegrityTestCase's class docstring, section
  header comment, test_corridor_source_no_band_split docstring, and the
  _M_LENS inline comment all described the REMOVED SITE 4
  (surrogate_census rho=None) guard as the reason the test passes. Rewrote
  all four to state the true mechanism: SITE 4/1 were removed, saddle
  rho<1 is no longer suppressed, and the corridor test passes because
  w_trust (28.746, finite, Cell 1 allowlisted) sits above the test's tiny
  w-band (max ~1.24 for f=[20,100]Hz, M=100 Msun) — not because rho was
  nulled. Also fixed the sibling test_lobe_interior_source_no_band_split
  docstring (same stale "SITE 4 does NOT fire" phrasing) for internal
  consistency even though the finding only named the class docstring +
  corridor test — leaving one stale SITE-4 reference next to the freshly
  corrected ones would have been an obvious residual inconsistency.
  Verified via search_for_pattern (only remaining "SITE 4" mention is
  explicitly "former"), ast.parse, and a live pytest run of the class
  (2 passed) since the finding's own numeric claims (w_trust=28.746,
  rho=0.175, w-band [0.2476,1.2379]) needed a green re-run to trust, not
  just a docstring-consistency check.
