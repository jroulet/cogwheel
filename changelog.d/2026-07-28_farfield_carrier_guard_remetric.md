---
date: 2026-07-28
---

### Fixed: the far-field carrier guard measured phase, but the chart splines re/im

`surrogate._assert_farfield_carrier_continuity` rejected an exterior tile whose
frame-invariant label `E_tilde` wound by `>= pi/2` in `arg` between adjacent
spatial nodes. `FarFieldChart`, however, stores `envelope_real` and
`envelope_imag` and splines them as SEPARATE REAL FIELDS. The two disagree
exactly at an amplitude NULL: the label passes close to the origin, so `arg`
swings by `pi` while `re` and `im` pass smoothly through zero. The guard was
watching a quantity the interpolant never sees.

This was not cosmetic. Nulls are generic in an interference pattern, and
`surrogate_training` responds to `CarrierDiscontinuityError` by subdividing
ONCE (`_train_band_charts`); a subdivided child that still raises is recorded
`'result': 'carrier_flip'` and never appended to `charts`
(`_subdivide_farfield_tile`), i.e. it becomes a ladder-served gap. Because
refinement cannot remove a null — the step PINS at `pi` as nodes are added
rather than shrinking like `1/n` — that subdivision never converged, so an
accurate chart was silently downgraded to fallback serving at every null. That
is a systematic zero-quadrature coverage leak sitting directly upstream of the
coverage census.

The guard now measures the complex increment `|E_lead - E_trail|` normalized by
the peak `|E_tilde|` over the WHOLE grid, against
`_FARFIELD_CARRIER_STEP_MAX = 1.0` (replacing `_FARFIELD_CARRIER_WIND_MAX`).
Whole-grid normalization is load-bearing: where the label decays with `w` the
top-of-band slice can be pure floating-point noise, and noise measured against
itself is O(1) while noise measured against the chart is zero.

Calibrated against every known fixture — worst must-pass 0.1997 (synthetic
continuous), must-raise 1.8980 (synthetic 2.5 rad flip at unit magnitude), a
9.5x gap with the bound placed at 1.0. Two alternatives were measured and
rejected: top-SLICE normalization (margin collapses to 1.24x) and scanning ALL
`w` slices (1.38x, because accurate charts genuinely carry large mid-band
increments). The bound is tuning-free to state — the label changed by more than
the entire chart's peak magnitude across one node gap — and at full amplitude
corresponds to `pi/3` of winding, i.e. STRICTER than the retired `pi/2` where
the label is strong, permissive only where it has decayed.

Four separate test-side bypasses existed for this one defect and all are
removed: `_skip_carrier_guard=True` in the census, band-split and
exterior-admission fixtures (and the kwarg itself is deleted from
`from_engine`), plus a `_from_engine_without_carrier_guard` mock-patch helper in
`test_lensing_surrogate.py`. Its reachable-red control
(`test_unpatched_positive_box_build_raises_carrier_discontinuity`) is deleted
too: it asserted that the false positive occurs, which promoted the bug to a
specification and made fixing the guard look like a regression. The guard's
genuine teeth remain certified by `FarfieldCarrierContinuityGuardTestCase`.

Verified with the guard LIVE against real engine-built charts: train tier 187
passed / 1 xfailed / 0 failed; full fast suite 834 passed / 126 skipped /
6 xfailed / 0 failed. See FINDINGS F022.
