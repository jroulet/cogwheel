---
date: 2026-08-12
section: Backlog
---

### A degenerate exterior band is now recorded instead of vanishing

Closes `todo.d/lensing_rho_outer_goes_negative_near_parity_boundary.md`.

`rho_outer_region = 1 + y_outer_region - coordinate_radius_min` goes `<= 1`
when `coordinate_radius_min` diverges, which it does for the macro saddle as
`det A -> 0`. The exterior interval `(1, rho_outer]` is then EMPTY,
`_farfield_tiles` returns `[]` on its `rho_outer <= rho_inner` guard, and the
region used to disappear with no error, no log and no record — silent
degradation, where this repo's convention is a loud one (cf. `beyond_w_cap`,
recorded rather than dropped).

MEASURED: the shipping trainer reports `rho_outer = -5.470619` with
`coordinate_radius_min = 9.470619` against `y_outer_region = 3.0` for the
topology-stable sub-band at `gamma ~ 1.005`; the census mirror measured
-4.147 with its cruder box corner, so production is worse than the estimate.
121 of 1742 macro-saddle census draws sit in such a band.

`_train_band_charts` now appends a `chart_<label>_exterior_band_degenerate`
record carrying `rho_outer`, `coordinate_radius_min`, `y_outer_region`,
`served: False` and a reason string — both inputs, so the diagnosis needs no
rerun. Behaviour is otherwise unchanged: the region is still empty, it just
says so.

Certified by `test_lensing_surrogate_training.py::
DegenerateExteriorBandIsRecordedTestCase`, which drives the SHIPPING
`_train_band_charts` with only the two engine chokepoints stubbed (the
caustic sweep, admissions, tiling and windows all run for real). Positive
case at `gamma ~ 1.005`, control at `gamma in (1.30, 1.34)` asserting no
false alarm. Verified by falsification: deleting the guard reds the positive
test while the control stays green.

NOT settled here, and left open deliberately: whether the formula is
meaningful at all near `gamma = 1`, and what rung should serve that
neighbourhood. `gamma = 1` is already a measure-zero named refusal; its
neighbourhood is not. This change makes the degeneracy visible so that
question can be answered from evidence rather than rediscovered.
