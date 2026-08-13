# Build Brief: Serve the deltoid exterior cusp window — surrogate-extension vs Pearcey (design decision)

## Mission

Close the serving gap just OUTSIDE the deltoid (saddle) cusp tip, AND the
astroid exterior cusp window's mid-w exact-engine flashback — with the
driver's mandate that the exact engine is NEVER the serving rung in the
cusp neighbourhood.  Documented in
`.claude/spec/todo.d/lensing_deltoid_exterior_cusp_gap.md`.

## Measured facts (at HEAD c24dee4)

- DELTOID (gamma=1.3): sources just outside the cusp tip (rho 1.0-1.2, on
  and off the lobe axis: (-2.0,0.05), (-1.8,0), (-1.72,0)) refuse on EVERY
  rung at w up to 5000: cusp arm (R<r_ppgo_min, uniform doesn't certify),
  fold arm (refuses even at fold_dist=0.086 inside the band), and the full
  ladder `operator.F_op` REFUSES (SchwingerCertificationError) — NOT served
  by the exact engine either.  Only FAR sources (rho~1.46) serve via ppGO
  at w>=500.
- ASTROID (gamma=0.5): exterior cusp window serves via ppGO at high w
  (quadrature-free) but has a MID-W exact-engine flashback: at radius_min <
  R < r_ppgo_min the uniform form does not certify, so F_op falls to the
  exact engine (served WITH quadrature — the "bad exact flashback" the user
  flagged).  Within `_CUSP_ARM_COVERAGE=0.07 rad` of the cusp, pearcey
  serves only at w=500 (ppGO); w=100-300 flash to exact or refuse.
- DESIGN (measured, in the code): the demodulated/ghost-subtracted exterior
  surrogate charts DO NOT cover the cusp neighbourhood — `_exclude_near_cusp`
  drops tiles within `_CUSP_EXCLUSION_DISTANCE = 0.35` of ANY cusp vertex
  ("near-cusp tiles induce oscillatory E_ff labels that a polar chart cannot
  resolve", surrogate_training.py:2154).  The intended owner is the CUSP
  ARM: `_CUSP_ARM_COVERAGE = 0.07 rad` (astroid), `_SADDLE_CUSP_ARM_COVERAGE =
  0.0` (saddle — explicitly zero, never calibrated).
- Surrogate w-ceilings: `_POSITIVE_W_CEILING=480`, `_SADDLE_W_CEILING=148` —
  once trained, the surrogate COVERS the mpmath band (60-148 for saddle,
  up to 480 for astroid) EXCEPT the carved-out cusp windows.

## THE DESIGN QUESTION (user mandate — the Professor MUST adjudicate, do not silently pick)

Who should serve the exterior cusp window (both parities, rho 1.0 - ~2.1,
on/near axis)?

(A) EXTEND THE DEMODULATED/GHOST-SUBTRACTED SURROGATE to the cusp windows:
    the fold-carrier `rho_u_carrier` demodulation + ghost-subtraction
    machinery was built for the region OUTSIDE THE FOLDS and already avoids
    exact-engine flashback there.  Can it resolve the near-cusp E_ff labels
    (the `_CUSP_EXCLUSION_DISTANCE` blocker) with a cusp-adapted `u=d^{2/3}`
    axis?  This is the user's preferred direction (they built that
    machinery) and would give quadrature-free serving.

(B) MAKE THE PEARCEY ARM SERVE THERE: calibrate the saddle cusp arm coverage
    (currently 0.0) and close the astroid mid-w gap (extend the geometric-
    limit serve below r_ppgo_min, or a table-only uniform serve).  The
    saddle calibration is speculative (F018: deep-interior images can sit
    arbitrarily close to the cusp).

(C) HYBRID per parity/region.

ACCEPTANCE regardless of choice: the exterior cusp window (both parities,
rho 1.0-~2.1, on/near axis) is served QUADRATURE-FREE by a fast path across
the w band; NO exact-engine serving in the window (driver mandate); no
serving gap (the deltoid must serve, not refuse); refusal-conservative.

## The build must ALSO resolve
- The deltoid exterior currently serves NOWHERE (not even exact).  Whatever
  the fast path chosen, the deltoid window must end up served.
- If (A): the surrogate extension is a training artifact change — the build
  must produce and certify the extended chart(s) and register them.  This
  may be the largest of the three builds.
- If (B): `_SADDLE_CUSP_ARM_COVERAGE` recalibration (scripts/measure_saddle_
  cusp_arm_coverage.py is the stub) + the astroid mid-w closure.

## Constraints
- Fast tests only; brute/training sweeps are post-build driver steps (but
  the chart(s) must be produced and the fast-tier serving tests green).
- Refusal-conservative.  No live quadrature in the fast path.
- Do NOT weaken the existing surrogate serving contract, the ppGO gates, or
  the one-home pin.
- This build DEPENDS ON the fold-arm/cusp-on-axis control work (the on-axis
  degeneracy affects the exterior window too) — sequence after
  `lensing_fold_pair_drops_third_cusp_image` and
  `lensing_saddle_interior_cusp_serving`.
