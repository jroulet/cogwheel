# Test Dev Short-Term Observations

## 2026-08-14 (F080 ppGO saddle rho<1 band-scope + census mirror shard, test_lensing_ppgo_map.py)
- Extended the SAME suite (prior shards did exact-floor/margin/gamma-adjacent):
  +6 tests, 0 retired, 44->50 passed in 5.9s. Three Architect specs:
  (1) F073 preserved for GENERIC off-band saddle rho<1 -> added ONE method
  test_generic_offband_saddle_low_rho_refuses to the existing
  ShippedMapSaddleRelaxedCellTestCase (cell2/cell3 gamma-adjacent already
  witnessed by test_non_allowlisted_certified_saddle_low_rho_refuses --
  parsimony, did NOT re-witness them); premise pins
  shipped_map._saddle_rho_relaxed_floor(...) is None so refusal is F073 not
  grid-miss. (2) Band-scoped -> new BandScopedRelaxationTestCase: rho sweep
  0->0.6 at cell-1 gamma asserting served<0.5 / UNKNOWN>=0.5 with
  transition-exactness (max(served_rho)<edge<=min(refused_rho)); gamma
  neighbors +/-1e-3 across the frozen grid edge -> UNKNOWN all 3 methods;
  saves output/band_scoped_relaxation_rho_transition.png. (3) served==counted
  -> new CensusLikelihoodBandSplitMirrorTestCase.
- SPEC-3 DECISIVE SUBSTITUTION (census band-split floor is NOT observable):
  characterize_sample consumes w_trust INTERNALLY to narrow the chart band
  then discards it -- the returned SampleRecord never surfaces it, so no
  engine run can read it back. Mirror = REAL likelihood _ppgo_band_split
  (bound onto a 2-attr _BandSplitProbe carrier, runs genuine shipping code
  via likelihood-module globals) vs a _census_band_split helper reproducing
  characterize_sample's inline rule THROUGH the same shipping primitives
  (caustic_rho + shipped_map.w_trust). In-box CLEAN cell -> both serve
  frozen 28.746; off-band -> both None; empty-allowlist patch flips BOTH to
  None together (teeth). Process global installed in setUp / restored in
  tearDown (F078 hygiene, save/restore around each test).
- FIXTURE DERIVED FROM LIVE REACH: in-box/off-band saddle sources built as
  |y| = rho * caustic_geometry(gamma,0)[0], y2=0, so caustic_rho realises
  the target rho exactly (0.25 in-box, 0.30 off-band) -- no pinned |y|.
- BACKWARD-COMPAT AUDIT: build only ADDS an allowlist serve path
  (monotone-toward-serve for one cell) + a new import of likelihood/
  surrogate_census into the test module. No pre-existing test flipped; all
  prior 44 still green. New top-level imports resolve at collection (verified
  via --collect-only). _BandSplitProbe binds unbound methods whose __globals__
  is likelihood's module dict, so get_certified_ppgo_map/caustic_rho/np/
  LensDomainError all resolve without a real Likelihood instance.


## 2026-08-14 (F080 ppGO saddle rho<1 per-cell relaxation, test_lensing_ppgo_map.py)
- Authored F080 CLEAN-cell relaxation suite: +3 classes / +8 tests, 0 retired
  (git diff = +286 lines, all insertions). Net +8; new invariants:
  (a) shipped-map in-box saddle rho<1 serves frozen floor 19.164305537818887
  at rtol=1e-9 (ShippedMapSaddleRelaxedCellTestCase); (b) margin rule
  w_trust==max(1.5*floor,floor+2.0)==28.746 propagates to w_trust AND the
  NEW w_ceiling saddle-rho<1 gate, w_ceiling finite & >=w_trust;
  (c) governance tripwire pins the single live _SADDLE_RHO_RELAXED_CELLS
  entry to FROZEN literal copies (not live reads → non-vacuous);
  (d) self-falsification: empty allowlist → UNKNOWN across all 3 methods,
  raised effective_floor moves w_cert+w_trust off the frozen literals.
- BACKWARD-COMPAT AUDIT FINDING: NO pre-existing test needed flipping. Every
  pre-existing saddle rho<1 test uses a SYNTHETIC map whose gamma edges are
  [0.0, gamma_max] (e.g. via _synthetic_ppgo_map / _SADDLE_BAND) which NEVER
  equal the shipped allowlist Cell-1 edges (1.1572945272629378,
  1.3393306228327468). _saddle_rho_relaxed_floor keys on EXACT float64 edge
  equality, so those synthetic queries still return UNKNOWN and remain valid
  F073-preserved witnesses (test_uncertified_cell_reads_unknown,
  test_narrowing_a_saddle_would_move_the_cell). The CLEAN box is reachable
  ONLY via the SHIPPED map (CertifiedPpgoMap.load()), which no legacy test
  queried at saddle rho<1. Lesson: an allowlist keyed on exact grid edges
  is inherently backward-compatible against synthetic-fixture suites — the
  new serving path is unreachable from any fixture with different edges.
- w_ceiling gained a NEW saddle rho<1 gate this WP (previously ungated);
  no synthetic w_ceiling test broke because their saddle read-points sit at
  rho>=1 (exterior full-radius). Suite: 44 passed in 7.35s.
- Shipped map loads fine via CertifiedPpgoMap.load() in setUpClass (no
  process global touched — F078 xdist-leak-safe).
