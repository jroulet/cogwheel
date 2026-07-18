# Test Dev Short-Term Observations

## WP1 fast-path suite (test_lensing_fast_path.py, 2026-07-18)

Rewrote the WP1 suite (6 Architect specs, kink-aware segmented kernel
interpolation). Full suite GREEN: 21 passed, 7 xfailed (exit 0).
Fixture: UNEQUAL-mass m1=35/m2=25 non-spinning, single det 'H', uniform
fbin DF_BIN=4, DELTA_T_MAX=0.02, seed 20260717 (equal-mass zeros 2 of 4
by_m harmonics -> 0/0 RB NaN; premise repair, not tolerance).

FINDINGS (WP1 shortfalls, encoded as unittest.expectedFailure so suite
stays green + self-corrects to unexpected-success when fixed):
- specs 1&2 RAW-KERNEL recon on shipped reduced grid: 3.6e-2..2.6e-1 null
  -safe, 35-260x above <1e-3 Build-3b gate. Converged 400-node GLOBAL
  spline hits 1.4e-7..2.2e-5 (positive control green) -> reduced node
  budget is the sole shortfall, oracle/method sound.
- spec 3 SEGMENTATION NOT load-bearing: at every crown kink the single
  global spline is 12-100x BETTER than segmented (mutation doesn't ring).
  Per brief contingency -> node budget should be re-derived.
- spec 5: crown reduces only 3.57x (<4x target; near-cusp/near-fold/
  sheared hit 4.5-6.2x). well-sep draws MORE nodes (30) than crown (28)
  -> monotonicity inverted (adaptive budget tracks delay-spread/beat).
  crown warm lnlike ~18.8ms > 15ms ceiling. Speedup 77x (>>2.5x, green).
- spec 4: near-fold RB-vs-brute 1.76nat > 1.5 gate (observable bitten by
  raw-kernel regression); crown/near-cusp/well-sep/sheared <=1.07nat green.

CORROBORATION: pre-existing test_lensing_likelihood.py (owned by another
run) ALSO fails near-cusp RB-vs-brute 2.35>1.5 -> independent confirmation
of the WP1 RB regression. Did NOT edit it (scope discipline).

TECHNIQUE NOTES:
- Off-grid raw-kernel probing must reconstruct via the SAME per-segment
  not-a-knot algorithm as production (searchsorted breakpoints + per-seg
  CubicSpline real/imag), then PROVE fidelity by reducing to k0/k1 via
  _kernel_fit_value/_kernel_fit_slope and asserting bit-equality vs
  like._amplification_coefficients (maxdiff 0.0). Closes "am I testing
  production?" gap for an internal path that only exposes reduced coeffs.
- Oracle = fresh ChangRefsdalChannels(w).evaluate at probe w (deterministic
  initial labeling matches production's fresh coarse-grid instance ->
  channel labels align; F002/F010 independent of the spline under test).
- ChangRefsdalChannels(w) requires strictly-increasing positive w:
  np.unique(np.sort(probes)) before evaluate.
- expectedFailure is the clean stdlib mechanism for "leave a plan
  -anticipated production shortfall RED without failing the build";
  pair each with a green converged positive control for non-vacuity.
