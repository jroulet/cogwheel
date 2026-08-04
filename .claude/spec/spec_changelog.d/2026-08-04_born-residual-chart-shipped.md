---
bump: minor
---

### Born residual chart trained and shipped — update SPEC.md

Both SPEC.md locations that described the Born residual chart as a pending
TRAIN_TIER artifact have been updated to reflect that the trained artifact
is now shipped (commit `849e580`):

**Engine row (table cell row 54):**
- "BORN EXTERIOR RUNG" banner updated from "carrier + wiring infrastructure
  landed; residual chart pending training" to "carrier + wiring + trained
  residual chart shipped".
- Added description of `cogwheel/data/born_residual_chart.npz` (≈ 8 KB,
  package data): 3-D tensor-product cubic spline of
  `R(w; gamma, rho) = F_exact_demod(w) − F_carrier_demod(w)` over a
  7 gamma × 5 rho × 10 w sparse grid, min-relative delay frame.
  Loading via `BornResidualChart.load(...)` completes the zero-quadrature
  exterior serve path for `rho > 2`.
- Low-w flat extrapolation paragraph added in the same location.

**Conventions bullet (lines 134–141):**
- Replaced "The chart itself is a TRAIN_TIER artifact (not yet trained);
  once trained, attaching it completes the serve path." with the current
  reality: artifact shipped as package data, zero-quadrature serve path
  complete for `rho > 2`.

**spec_version** bumped from 0.31.2 → 0.32.0 (two new descriptive sections).
