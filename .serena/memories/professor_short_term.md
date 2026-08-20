# Professor Short-Term Observations

## 2026-08-20 — low_w_diffractive_chart rung review (verdict PASS)

Reviewed LowWDiffractiveChart + its `_low_w_diffractive_chart_serve` Rung-P
intercept (commit 3a56e97). 36/36 fast tests green in 28 s; census +
diffractive suites 76 passed / 3 skipped, no regressions. Physics verified
numerically, not just green: (1) node-exact re-modulation matches
`_engine_reference_kappa` to 1e-16..1e-14 (kappa=0.2 mass-sheet round-off at
the high end), confirming prefactor_c and the mass-sheet phase are each
applied exactly once and 1/lam is folded into sqrt_mu_full via the
sqrt(mu_macro)/sqrt(mu_pure) ratio; (2) DC anchor |F_serve|/sqrt(mu_macro)
= 1.00079 -> 1.00786 over w=1e-3..1e-2 (the O(w)~0.79*w correction), arg
~-0.024 rad — anchor is sqrt(mu_macro), never 1; (3) r_pure =
f_pure*sqrt(1-gamma'^2)/C(w) is bounded/smooth near the wall (->1 as w->0
at gamma'=0.9/0.95, genuinely varies to 0.48-0.88 at w=1) so the residual
strip removes the parity-wall divergence as designed; (4) unit-derate cubic
overshoot measured 1.5785 on the fixture set, conservative derate 0.576
restores one-sidedness (self-falsification has teeth). Notes: (a) the
shipped artifact cogwheel/data/low_w_diffractive_chart.npz does NOT exist yet
— auto-load falls to None (pure engine) with RuntimeWarning, so the rung is
DORMANT until the driver bake; (b) the scalar de-rate guarantees one-sided
conservativeness only on the calibration set (grid nodes + theta midpoints);
arbitrary 4-D off-grid points are the full-bake margin report's job (operator
deferred); (c) low_w_diffractive_chart.py's `_WALL_GAMMA_PRIME` docstring
slightly conflates the w_low_fit SERIES calibration ceiling (0.5) with the
chart's OWN gamma grid ceiling (1-DELTA_GAMMA_P ~ 0.995) — a doc wording
issue, the wall-band clause is LIVE not dead.

## 2026-08-20 — diffractive certificate wall-approach over-serve ruling

Ruled on the `w_low_fit` deep-interior-ceiling-serve over-serve near the
positive-parity wall (Rung P, order-16 operator series, `_diffractive.py`).

Physics finding: the fence discriminator `rho = |y'|/|y_c(theta)|` is
CAUSTIC-RELATIVE, while the honest ceiling is governed by the ABSOLUTE small
parameter `gamma' * s * w / 2` (s = |y'|**2). Near the wall `|y_c| -> inf`
(F036), so `rho < RHO_LO` does NOT imply small `s`; the two coincide only
where `|y_c| = O(1)` (away from the wall). A "deep interior" source at
gamma'=0.98 can have O(1) absolute offset, making the order-16 series
dishonest at w=60.

Second, INDEPENDENT mechanism (dominant near the wall): the shear-operator
series has convergence radius EXACTLY the parity wall (gamma'=1); its resummed
value carries `sqrt(mu_macro) = 1/sqrt(1-gamma'^2)` which diverges there, so
the order-M truncation has relative error ~ gamma'^(2M+2)/(1-gamma'^2) — O(10)
at gamma'=0.98, M=16 even at w->0. So the series is dishonest at ALL w,
independent of rho. This is distinct from the small-parameter (w-axis)
convergence; DELTA_GAMMA_P=5e-3 is a convergence-radius margin, NOT a
truncation-accuracy fence (dishonesty sets in ~gamma'>0.8, far wider than 5e-3).

Ruling: remove the `rho < RHO_LO -> ceiling(60)` branch; serve the interior
via the fitted honest ceiling (which saturates at 60 only where honest).
Decline to the engine above the fit's calibration ceiling (gamma grid stops
at 0.5) — the wall-approach band gamma in [0.9, 1) is extrapolated AND
dishonest-at-all-w, so it must route to the exact Schwinger engine (no
coverage loss, only a ~6% performance cost). Combined wall-margin +
small-parameter condition; implement as the fitted ceiling + a
calibration-domain fence, never a hard gamma constant or hard-coded 60.
