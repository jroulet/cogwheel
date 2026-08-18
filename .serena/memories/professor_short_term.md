# Professor review — c3 band-split / above-ceiling build (2026-08-17)

Reviewed the c3-band-split + per-node above-ceiling build (uncommitted
`likelihood.py`, +357 lines; new tests in `test_lensing_saddle_serve_gate.py`).

## Verdict: CONCERN (physics PASS; collateral stale-mock regression)

### All six spec invariants PASS (real/independent oracles where physics matters)
- Spec1 shared `_band_split_mask` unit: 62/62 in serve-gate file; band_split iff
  w_lo<split<w_hi, null-split below_mask all-True, interior == dense_w<=split. PASS.
- Spec2 `_saddle_c3_split_point`: exact cube-root inversion S*est(w_split)==bar to
  fp; certificate passes at w_split*(1+eps), fails just below; est ~ w^-3 strictly
  decreasing; reference-freq independent; merging pair (est=None) -> None. PASS.
- Spec3 c3 null-split byte-exact: w_split<=w_lo -> whole-band zero-envelope serve
  byte-identical; w_split>=w_hi -> None (engine refuse). PASS.
- Spec4 c3 in-band accuracy (THE calibration guard): served zero-envelope F vs
  INDEPENDENT exact DD oracle f_schwinger over [w_split,55]; clean contract configs
  max|dF| in {6.8e-4,5.5e-5,3.8e-5} <= 1e-3. Frame-lift alignment proven load-bearing
  (unaligned ~0.9). Leaky-gate witness (gamma=2, y=1.1, w_split~9.6) HONESTLY pinned
  at ~3.1e-3 miss in the near-caustic low-split corner (20x safety absorbs leading
  w^-3 but not subleading) — escalated/pinned, not papered over. PASS + good science.
- Spec5 per-node above-ceiling partition: `PpgoAboveCeilingPartitionTestCase` uses
  REAL geometry_partition, spies only the slow Schwinger step, proves clean stitch at
  150 (below=engine sentinel, above=independent fold re-derivation, no leak/double-
  count), near-caustic unresolved -> None. PASS.
- Spec6 Born refactor byte-identity: test_lensing_born_residual_wiring.py fully green
  (43 passed incl. Born). PASS.

### CONCERN — collateral regression in the OLD mock file (not physics)
`cogwheel/tests/test_lensing_ppgo_above_ceiling.py` (UNMODIFIED by build): 7 failed +
6 errors. CONFIRMED via git stash: these PASS at HEAD, fail with the build's
likelihood.py. Root cause: the rewritten split-band `_ppgo_above_ceiling` adds a new
collaborator `self._engine_envelope_below_split` -> `_evaluate_envelope`; the old
tests drive it with `stub = MagicMock()` self, so the mock leaks into
reconstruct_farfield ("operands could not be broadcast (0,) (6,/2,)"). Stale test
SCAFFOLDING, NOT a domain error — the real-geometry spec-5 pin covers the same code
green. Spec explicitly said "re-point the existing pin rather than add a parallel
class"; the build added the new real-geometry pin but left the old mock file red.
Fix owed before ship: update the mock stubs to provide `_engine_envelope_below_split`
(and `_evaluate_envelope`), or retire the file in favor of the new pin (test-parsimony).

Heavy full-sampling validation not run (operator-deferred, per turn budget).
