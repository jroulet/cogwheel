# Professor Short-Term — 2026-08-10

## ExteriorPolarChart rho_log_axis review

Reviewed the Simplifier's build plan for adding a `rho_log_axis` flag to `ExteriorPolarChart`
that trains the spline's rho axis on `ur = log(rho-1)` instead of raw `rho`.

### Physics endorsement
- `log(rho-1)` is physically correct: the far-field envelope decays as a power law in
  `rho-1` (exponent ranging -1.7 to -3.2 depending on theta_c and gamma proximity to
  caustic). NOT the `z^(3/2)` evanescent decay — that's the imaginary-plane steepest-
  descent contour, not the real-axis spatial decay.
- No carrier-like phase winding in rho: only the w-axis had rapid phase accumulation
  from the delta-tau carrier. Rho-axis phase varies smoothly at O(1) scale.
- The 4.5-decade dynamic range compression (→ ~10 additive units in ur) is the real win.

### Accuracy assessment
- 4 nodes on the log axis may be marginal. Theoretical cubic spline bound ~30% at
  effective exponent p≈2.4, but bounds are pessimistic and the log transform's effective
  conditioning is better than the simple power-law error model captures. Ship at 4,
  measure empirically, be ready to bump to 5-6.
- The varying exponent across theta_c and gamma (20× and 6× variation in |E|) means the
  tensor-product spline's cross-derivative structure matters — it's not just a 1D problem.

### Test specifications
Endorsed proposed gates (node-exact round-trip, heldout eps ≤ 1e-3, A/B test, schema
hard-refuse) plus three additions:
1. Monotonic decay guard: |E(rho)| must decrease with rho — no spline overshoot
2. Overlap-region agreement: exterior chart matches tube chart at eta boundary
3. Phase continuity: arg(E) unwrapped must not jump > π/2 between adjacent rho nodes

### Edge cases
None. Exterior gate (eta ≥ 0.05 → rho ≥ 1.05) keeps us away from ill-conditioned log
regime. Training margin down to rho=1.02 is conservative and correct.

### Related observations
- The F016 max-normalized currency (max|dE|/max|E|) should be used for all eps gates
- No impact on reconstruct_farfield or census — both operate on interpolated values
- If rho_log_axis proves successful, the same approach may generalize to the gamma axis
  (which varies |E| by ~6×) but that's a separate build

## 2026-08-10: RhoLogAxis build review (verdict PASS)

Reviewed ExteriorPolarChart rho_log_axis implementation across three test files (test_lensing_surrogate.py, test_lensing_exterior_carrier.py, test_lensing_surrogate_training.py). 96/96 fast tests green, 31 train-tier skipped (engine-backed, gated on COGWHEEL_TRAIN_TIER=1).

### Verified physics:
1. **Node-exact round-trip** at machine precision (<1e-15): ur = log(rho-1) at grid nodes reproduces training values exactly — the tensor-product cubic B-spline + coordinate remap is algebraically consistent.
2. **FromValues gate**: rho_grid[0] ≤ 1.0 → ValueError (log undefined), both `=1.0` and `<1.0` branches covered. Knot bounds correctly match ur_grid for log charts, raw rho_grid for linear.
3. **A/B comparison**: log-axis held-out error strictly smaller than raw-axis by ≥3× at 4 rho-nodes, both parity branches. Diagnostic plot confirms consistent improvement across all rho probes.
4. **Carrier composition**: rho_log_axis=True + carrier_rate≠0 compose correctly — off-grid served values within HELDOUT_BAR (5e-2). The ur remap and carrier demod/re-mod are orthogonal.
5. **Schema hard-refuse**: Old `exterior_polar_carrier_demod_v2` tag raises ValueError at load. New V3 schema loads and serves identically after NPZ round-trip (bit-identical coefficients, knots, axes).
6. **Self-falsification**: Every guard has teeth — log vs linear returns different values, node-exact assertion can fail deliberately, rho≤1.0 gate catches violations.

### Physics endorsement: 
log(rho-1) is physically correct — the far-field envelope decays as a power law in (rho-1), and the log transform compresses the dynamic range so the cubic spline operates on smoother, better-conditioned data. No phase-winding concern in rho (smooth O(1) variation, unlike the w-axis carrier).

### No concerns. 
Train-tier validation (COGWHEEL_TRAIN_TIER=1) is operator-deferred post-build.
