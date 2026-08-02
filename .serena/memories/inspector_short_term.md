# Inspector Short-Term Observations

## Build 1e-gamma — Log-Reach Gamma Axis Collocation (2026-08-02, re-review)

### Scope
Re-review of WP1: Replace uniform gamma grid in surrogate chart construction
(tube, far-field, lobe) with a log-reach-collocated grid that clusters nodes
near the parity wall (gamma → 1) where caustic geometry changes fastest.

**Files changed (production):**
- `cogwheel/lensing/surrogate.py` — new function `_log_reach_gamma_axis`,
  two call sites changed from `_uniform_axis` to `_log_reach_gamma_axis`
  (FarFieldChart.from_engine, LobeInteriorChart.from_lobe_engine)
- `cogwheel/lensing/surrogate_training.py` — one call site changed
  (`_train_band_charts`), import added

**New test file:**
- `cogwheel/tests/test_lensing_log_reach_gamma.py` — 17 fast-tier tests
  (12 structural + 5 self-falsification), 6 TRAIN_TIER-gated tests
  (comparative accuracy + regression), all skipped in fast tier.

### Findings

**INS-10-001 (trivial, Librarian — STILL OPEN):** `DATA_CONTRACTS.yaml` line
199 says `param_spacing = [mean d gamma, mean d s, mean d d] over the uniform
(gamma, s, d) grids`. The word "uniform" for gamma is now stale — the gamma
grid is log-reach-collocated. The code correctly uses `np.mean(np.diff(gamma_grid))`
regardless of grid uniformity, so the exclusion ball still functions correctly
(the mean step is a reasonable normalization scale). Only the documentation
phrasing is inaccurate. Not fixed this build — no change to DATA_CONTRACTS.yaml.

### Production Code Assessment
The `_log_reach_gamma_axis` function:
- Correctly handles both positive (gamma < 1, reach increasing) and saddle
  (gamma > 1, reach decreasing) parities via ascending-xp detection for np.interp
- Pins exact endpoints after interpolation (no FP drift)
- Passes through `_validate_axis` (strict monotonicity, size ≥ 4)
- Uses 200-sample fine tabulation (adequate for production band widths)
- Error handling for reversed ranges and too-few nodes
- Signature is drop-in compatible with `_uniform_axis`
- Math verified: for saddle band (1.02, 1.40), t_fine is decreasing,
  else-branch uses reversed arrays for np.interp, result is correctly
  increasing in gamma. Defensive sort() is a no-op. ✓

All existing test suites exercising the changed code pass:
- test_lensing_surrogate.py: 69 passed (112.5s)
- test_lensing_surrogate_census.py: 14 passed, 13 skipped (101.8s)
- test_lensing_surrogate_lobe.py: 54 passed (52.0s)
- test_lensing_ppgo_bandsplit.py: 62 passed, 4 skipped (28.4s)
- test_lensing_exterior_windows.py: 73 passed, 1 xfailed, 3 FAILED (pre-existing)

The 3 failures (GhostFrameCollapseTestCase) + 3 errors are PRE-EXISTING on HEAD —
confirmed by previous reviews. Not a regression from this build.

### New Test Suite Assessment
- 17 passed, 6 skipped in 3.12s (well under budget)
- Correctly uses only `_log_reach_gamma_axis` and `_caustic_reach` from the
  module under test (non-circular: tests the ROUND-TRIP property using
  independently computed expected_t values)
- Self-falsification proves: uniform grid fails round-trip, fails clustering,
  raised ranges and too-few nodes are caught
- The TRAIN_TIER tests (Spec 1, Spec 3) are properly gated and well-budgeted
- Anti-vacuity patterns (n_checks, tearDown) applied throughout

### Consumer Impact Check
- `lens_amplification_surrogate` artifact consumers (8 total) are UNAFFECTED:
  the gamma grid is read from the persisted npz at load time, not regenerated.
  The change only affects the training/construction path.
- `_uniform_axis` remains available for s_grid, d_grid, rho_lobe_grid

### Existing Test Survival Check
- `test_lensing_surrogate_lobe.py` line 1973: still uses `_uniform_axis` for
  gamma grid, INTENTIONAL — builds V1/backward-compatible chart manually with
  uniform nodes for comparison. `_uniform_axis` remains available. No breakage.

### Carried Forward (pre-existing)
- INS-10-001: DATA_CONTRACTS.yaml "uniform" gamma phrasing stale (Librarian)
- INS-9-001: Dead allowlist entries in test_lensing_part0_mechanical.py (trivial)
- INS-8-001: test_raising_constant_to_two_refuses_an_admit_config fails (pre-existing)
- INS-5-001: SPEC.md old annulus references — Librarian
- INS-5-003: DATA_CONTRACTS.yaml line 228 'caustic-frame annulus rho' — Librarian
- Pre-existing: GhostFrameCollapseTestCase failures (GhostDomainError, 3 failures)

### Verdict
**PASS** — one trivial documentation finding (INS-10-001, still open from
previous build, Librarian-owned). No bugs or design issues introduced by this
build. The implementation is mathematically correct, all existing suites survive
the change, and the new test suite is well-structured.
