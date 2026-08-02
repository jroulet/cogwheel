# Professor Short-Term Observations

## 2026-08-02: _extrapolate_floor build review — PASS

Reviewed the `_extrapolate_floor` implementation and its test suite
(`test_lensing_extrapolate_floor.py`). All 10 fast-tier tests pass; 4 engine-backed
tests correctly gated by `COGWHEEL_TRAIN_TIER`.

### Physics verification:
- **Power-law fit**: Result 2114.86 vs analytic 1994.74 (ratio 1.06, within 30% tol).
  The ~6% overshoot is expected from beat-induced positive bias on the log-log fit.
- **Slope bounds [0.75, 1.5]**: Physically motivated by fold (~1/w) and cusp
  (~w^{-5/6} to w^{-4/3}) decay laws in the unresolved diffraction regime.
- **R² > 0.9**: Rejects random scatter (measured R²=0.04 for uniform noise).
- **MAX_RATIO = 5.0**: Prevents extrapolation beyond ~0.7 decades — appropriate
  conservatism for a power-law extrapolation of an oscillating envelope.
- **Deflation factor 2.0**: Provides a safety margin by reducing the certified w.
- **Interior-only guard (rho_center < 1.0)**: Physically correct — only 4-image
  interior cells have the fold/cusp decay structure that justifies power-law
  extrapolation; 2-image exterior cells have different error geometry.
- **floor > w_ceiling logic**: For interior cells, extrapolated floors can validly
  exceed w_ceiling (the ceiling bounds measurements, not the decay law itself).

### Test design quality:
- Anti-vacuity tearDown prevents silent green on broken impl.
- SelfFalsificationTestCase proves all guards are load-bearing.
- Excessive-extrapolation test uses alpha=0.8 (inside bounds) so ONLY the ratio
  guard triggers — good isolation.
- Positive test uses [10, 2000] grid (not spec's [1, 60]) to keep ratio=0.5 < 5.
  The [1, 60] grid with same C/bar/alpha would give ratio=16.6 — correctly refused.

### Broader suite:
- `test_lensing_ppgo_bandsplit.py`: 62 passed, 4 skipped (engine-gated).
- `test_lensing_ghost.py`: 31 passed, 1 xfailed (expected).

Heavy full-sampling validation is operator-deferred.
