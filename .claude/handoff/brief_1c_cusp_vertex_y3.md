# Build Brief: 1c — Serving-Path Cusp Vertex + Third-Order Derivatives

## Mission

Replace `_pearcey_cusp._cusp_vertex`'s numerical speed-scan with an analytic
root of `|y'(theta)| = 0`, and extend `caustic_derivatives` to third order
(`y'''`). After this, nothing in the package's caustic geometry is estimated
rather than derived.

## In scope

- Replace `_cusp_vertex`'s 129-point scan + golden-section refine with an
  analytic root-find of `|y'| = 0` (cusps are exactly where caustic speed
  vanishes). Use the closed-form `caustic_speed` from build 1a and `brentq`
  or analytic root as appropriate.
- Extend `geometry.caustic_derivatives` to return `y'''` (third-order
  derivative of critical_point w.r.t. theta). This is the same Taylor
  expansion already used for `y'` and `y''`, extended one order.
- The cusp-exclusion half-width `w^{-1/4}` with coefficients in `|y''|` and
  `|y'''_perp|` (F040) — derive the formula; implementation of the
  cusp-window schema itself is OUT OF SCOPE (deferred per TODO).
- Verify `y'''` against the same two-stage oracle at the F038 tolerance.
- Verify served Pearcey values unchanged to the F016 envelope bar.
- Verify cusp angles pinned to the analytic root at 1e-10.
- Verify O(1) geometry calls per serve instead of ~258.

## Out of scope

- Cusp-window schema changes (deferred per TODO — `_CUSP_SPEED_REL_FRAC`
  literal survives until that build).
- Any training runs or artifact generation.
- Far-field or tube coordinate changes (already done in 1e builds).
- The interior-admission distance optimization (separate measurement step).

## Measured facts

- `caustic_derivatives` currently returns `(y', y'')` from
  `cogwheel.lensing.chang_refsdal.geometry` (build 1a, commit `1a82046`).
- `_cusp_vertex` lives in `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`.
- It uses `delta = 1e-4` hardcoded central difference, 129 thetas over a
  pi window (~258 `critical_point` calls), then golden-section refine.
- The F016 envelope bar is the acceptance tolerance for served Pearcey values.
- The F038 oracle is a two-stage (coarse FD + fine FD) independence check.

## Acceptance

- Served Pearcey values unchanged to the F016 bar.
- Cusp angles pinned to the analytic root at 1e-10.
- O(1) geometry calls per serve instead of ~258.
- `y'''` verified against the two-stage oracle at atol=5e-13, rtol=1e-11
  (same as F038 for `y'` and `y''`).
- No production path in `cogwheel/lensing/` differentiates
  `geometry.critical_point` by differencing or sampling.

## Constraints

- Fast tests only (no COGWHEEL_BRUTE_ACCURACY, no COGWHEEL_STRICT_TIMING).
- Do not train any chart artifacts.
- This is on the SERVING PATH — changes must not regress any served values.
- Follow AGENTS.md and the spec/TODO workflow.
