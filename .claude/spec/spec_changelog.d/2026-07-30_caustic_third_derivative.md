---
date: 2026-07-30
bump: minor
---

### `geometry.caustic_third_derivative` (y''') joins the public name list

Doc sync for commit `b9c3ed6` (build 1c). `caustic_third_derivative(gamma,
theta, *, kappa=0.0, branch=1)` extends the analytic caustic-derivative
cascade one order beyond `caustic_derivatives`, returning the closed-form
third theta-derivative `y'''` of the caustic curve via the shared private
`_caustic_cascade` helper. Added to the microlensing-engine row's geometry
public-name list beside `caustic_derivatives`/`caustic_speed`/
`caustic_curvature_radius`/`fold_opening_direction`. No consumer wires it yet
(owed to the F040 cusp-window step); it ships and certifies the primitive.
New public API, hence a minor bump.
