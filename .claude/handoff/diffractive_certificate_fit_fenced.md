# Build: fenced w_low_fit — fit the smooth region only, route the near-fold shell out

## Mission

The prior representation (smooth basis over the WHOLE domain + global
de-rate) is SUBOPTIMAL: a single non-analytic |dtheta|^(2/3) dip at the
directional caustic (the fold) forces a global de-rate of 0.50, so the entire
surface serves at ~half the honest ceiling to protect one corner. The corner
is a region the weak-deflection diffractive rung fundamentally should not
serve — the near-fold band belongs to the uniform Airy fold arm
(`_airy_fold.fold_amplification`) and the exact engine.

This build FENCES the near-fold shell OUT of the fit's domain: draws inside a
shell around the directional caustic are NOT certified by `w_low_fit` (they
fall through to the fold arm / engine), and the fit covers only the smooth
region where it is good. The de-rate then returns toward the 0.85 hard floor
and tightness recovers — because the fit is no longer paying for a corner it
shouldn't own.

## The fence discriminator (already in the working tree)

The stranded build shipped the parametric caustic feature
`log(|y'| / |y_c(theta)|)`:
- `geometry.caustic_point(gamma, theta, *, beta, kappa)` (geometry.py:1365) —
  O(1) parametric caustic curve, closed-form trig+sqrt, pure python.
- `_diffractive.py`: `_DIFFRACTIVE_FIT_CAUSTIC_COEFF` and the feature in
  `_fit_features`.
This feature is EXACTLY the fence discriminator: `rho = |y'| / |y_c(theta)|`
is 1.0 ON the caustic, > 1 outside (smooth region), < 1 inside. The fold
dip sits at the near-fold shell rho ~ just-above-1. A fence
`log(rho) > -log(1 + eps)` (equivalently rho > 1 + delta) excludes it.

## Design (owner ruling)

1. FENCE: `w_low_fit` refuses (returns None) inside the near-fold shell
   `rho = |y'|/|y_c(theta_source)| < 1 + delta`, where delta is a small
   tunable (order 0.1-0.3, chosen so the excluded shell covers the
   non-analytic dip and nothing more). Inside the shell the consumer's
   existing fall-through routes the draw to the fold arm / engine — the
   diffractive rung simply does not serve there. This is the Professor's
   fallback (b): "route |log rho| < eps shell to the engine/fold
   asymptotics" — NEVER a global de-rate-floor drop.
2. FIT the smooth region only (rho >= 1 + delta): the even-harmonic
   cos(2k theta) k=1..7 basis + degree-2 log-log poly (the stranded build's
   representation, which is GOOD in the smooth region — the 2.06x over-
   prediction was purely the fold corner). Re-bake at --scale full on the
   fenced domain.
3. DE-RATE target returns to >= 0.70 (worst raw over-prediction <= 1.43x),
   certified OFF-GRID (theta midpoints) on the FENCED domain. The de-rate
   no longer needs to absorb the fold.
4. Served series stays exact order-16; refusal semantics (wall via
   `_reduced_shear`, degenerate None) unchanged; the fence refusal is a NEW
   near-caustic refusal distinct from the wall.

## Scope

IN:
- `_diffractive.py`: `w_low_fit` gains the fence (refuse when
  rho < 1 + delta); `_DIFFRACTIVE_FIT_FENCE_DELTA` baked constant; the fit
  surface, caustic feature, and harmonic basis stay (re-baked on the fenced
  domain). `_fit_features` unchanged; the fence is a refusal gate on top.
- `scripts/fit_diffractive_certificate.py`: `_grid_points`/`_off_grid_points`
  restrict to the fenced domain (drop rows with rho < 1 + delta, or weight
  them out); margin report reports the fenced-domain conservative/tight
  distribution. Smoke bake in-build (< ~10 min, PROVISIONAL coefficients);
  full bake is a DRIVER step.
- `likelihood.py` / `serve_route_census.py`: the consumers already fall
  through on `None` — the fence refusal needs NO consumer change beyond what
  exists (verify: `w_low_fit -> None` inside the shell routes to the fold/
  engine path exactly as a wall refusal does). Confirm this is byte-identical
  fall-through.
- Tests: re-scope the corner pin (INS-1-001's ~1.99x bar) to assert the
  FENCE removes the corner from the fit's domain (inside the shell,
  `w_low_fit` returns None; just outside, the fitted surface is
  conservative). D2 symmetry test stays (the even-harmonic basis is still
  correct in the fenced region). Off-grid midpoint oracle stays, now on the
  fenced domain.

OUT (do not touch):
- The uniform arms / fold / engine serving.
- The served series.
- Rung S / macro-saddle engine-host.
- Any surrogate-chart or campaign work.

## Acceptance

- INSIDE THE SHELL (rho < 1 + delta): `w_low_fit` returns None (the diffractive
  rung declines; the draw routes to the fold arm/engine). Asserted on
  fixtures at the fold dip (gamma=0.41, r=0.55, theta midpoint 2.454) and
  nearby.
- OUTSIDE THE SHELL (fenced domain): conservative AND tight. De-rate >= 0.70
  (worst raw over-prediction <= 1.43x), median >= 0.6, off-grid tightness
  >= 0.5x on >= 80% — measured on the theta-midpoint witness set of the
  FENCED domain. If the fence delta needs tuning to hit 0.70, tune it and
  report the achieved de-rate + the excluded-shell prior-mass fraction (must
  be small, e.g. < 10% of residual demand).
- The fence refusal falls through byte-identically to the existing
  fold/engine path (consumer behavior verified, no new exception).
- PROVISIONAL smoke coefficients committed with the '# PROVISIONAL' marker;
  full bake + paste + full on/off-grid validation is the DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build smoke bake < ~10 min.
- Spec/TODO: `[→ spec]`.
- Mirror-fidelity: production `w_low_fit` and the bake script share the basis
  + fence builders; never two copies.
- The full bake is a DRIVER step (bulk sweep, NOT in-build — the SDK 1200s
  ceiling killed the last in-build bake).
