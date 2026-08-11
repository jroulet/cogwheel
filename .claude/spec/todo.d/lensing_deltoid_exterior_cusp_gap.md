---
section: Backlog
depends_on: []
---

- **Deltoid (saddle) exterior cusp region just outside the tip is UNSERVED — no rung certifies**
  `[→ spec]` — measured 2026-08-11 by the driver (user-flagged exterior coverage).

  For gamma=1.3, sources just OUTSIDE the deltoid cusp tip (rho 1.0-1.2, on
  and off the lobe axis, e.g. (-2.0, 0.05), (-1.8, 0), (-1.72, 0)) refuse on
  EVERY rung at w up to 5000:
  - cusp arm: R < r_ppgo_min (71) so ppGO does not fire; the uniform form
    does not certify (measured R=26 at w=500, 44 at w=1000 for (-2.0,0.05)).
  - fold arm: outside the fold band / not a fold.
  - full ladder `operator.F_op`: REFUSED (SchwingerCertificationError) at
    w=200/500/1000 — NOT served by the exact engine either.
  Only FAR deltoid sources serve (rho ~ 1.46, |src-tip| large enough that
  R >= 71) via ppGO at w >= 500.

  Compare ASTROID exterior (gamma=0.5): the mid-w band (radius_min < R <
  r_ppgo_min) falls through to the EXACT ENGINE and is served (measured
  F_op = 1.50/0.99/1.49 at w=200/300/500 for (1.5,0.05)) — served WITH
  quadrature but not a hard gap; high w serves via ppGO quadrature-free.

  So the deltoid exterior-cusp-neighbourhood has a genuine coverage hole:
  near the tip it is neither served by a fast path NOR by the exact engine.

  ROOT (hypothesis to verify): the saddle exact path (w > 150 above the QD
  ceiling, or the mpmath band) refuses because `f_schwinger` for the saddle
  at these near-cusp-exterior parameters does not certify (or w*delta_min
  < RHO_END so geometric refuses, and the arms decline).  Needs a serving-
  ladder investigation: which rung SHOULD own the deltoid just-outside-tip
  region, and why it refuses.

  ACCEPTANCE: deltoid exterior sources just outside the cusp tip (rho 1.0
  to ~1.2) are served by SOME rung (fast path preferred: ppGO or the
  exterior surrogate / Born rung / fold-ppGO; exact engine acceptable as
  fallback but must serve, not refuse) across the w band; no live
  quadrature in the fast path; refusal-conservative.

  ## Driver verification (2026-08-11): the ppGO gate is NOT the blocker

  The build's ppGO fold-band gate (nearest.distance >= 0.3) does not cause
  this gap: at (-2.0,0.05) nearest.distance = 0.290 < 0.3, so the gate
  correctly defers to the fold arm — but the FOLD ARM ALSO REFUSES (all
  tested sources, even fold_dist=0.086 inside the band).  The cusp uniform
  form does not certify at these R, and the exact engine (operator.F_op)
  refuses (SchwingerCertificationError).  Pre-existing, not build-caused.

  ## DESIGN CROSSING (driver, 2026-08-11): the surrogate does NOT own this region

  The demodulated/ghost-handled exterior surrogate charts do NOT cover the
  cusp neighbourhood: `_exclude_near_cusp` drops tiles within
  `_CUSP_EXCLUSION_DISTANCE = 0.35` of ANY cusp vertex (both parities),
  because "near-cusp tiles induce oscillatory E_ff labels that a polar
  chart cannot resolve" (surrogate_training.py:2154).  The intended owner
  of the exterior cusp window is the CUSP ARM, per the design comment at
  surrogate.py:295-308:
  - astroid: `_CUSP_ARM_COVERAGE = 0.07 rad` (small window, cusp arm owns it)
  - saddle:  `_SADDLE_CUSP_ARM_COVERAGE = 0.0` (explicitly zero; "saddle
    deep-interior images can be arbitrarily close to the cusp (F018)")

  So the deltoid exterior cusp window is a design hole: the surrogate
  excludes it (can't resolve), the cusp arm has ZERO coverage for the
  saddle, the fold arm refuses, and the exact engine refuses.  The astroid
  window is owned by the cusp arm but has a mid-w exact-engine flashback
  (R between radius_min and r_ppgo_min, where the uniform form does not
  certify and F_op falls to the exact engine) — served, but WITH
  quadrature, contradicting the zero-quadrature serving goal.

  This is the same cusp-neighbourhood family as `lensing_fold_pair_drops_
  third_cusp_image` (on-axis) and `lensing_saddle_interior_cusp_serving`
  (deltoid interior): the cusp arm's certified reach does not cover the
  full cusp neighbourhood, and `_SADDLE_CUSP_ARM_COVERAGE = 0.0` documents
  that the saddle was never calibrated.  Fix directions: (a) calibrate the
  saddle cusp arm coverage (scripts/measure_saddle_cusp_arm_coverage.py is
  the stub), (b) extend a cusp-adapted chart (u = d^{2/3}) into the
  exclusion window, or (c) gate the mid-w astroid flashback to keep it
  quadrature-free (ppGO threshold or a table-only uniform serve).

  ## DESIGN QUESTION (user, 2026-08-11): who SHOULD serve the exterior cusp
  ## window — the demodulated/ghost surrogate or the Pearcey arm?  THIS is
  ## the open decision to resolve; do not let it drop.

  The user posed the explicit design choice for the cusp-EXTERIOR region
  (just outside the caustic, on/near the cusp axis, BOTH parities):

  (A) EXTEND THE DEMODULATED/GHOST SURROGATE to serve outside the cusp too:
      the real-demodulated / imaginary-ghost-subtracted exterior charts
      (fold-carrier `rho_u_carrier` + ghost handling) were built for the
      region OUTSIDE THE FOLDS and already avoid exact-engine flashback
      there.  The question: can the SAME machinery serve the cusp-exterior
      window (rho 1.0 - ~2.1, on/near the cusp axis) where it currently
      flashes back to the exact engine (astroid) or refuses (deltoid)?
      The blocker is `_CUSP_EXCLUSION_DISTANCE = 0.35` (surrogate_training
      drops near-cusp tiles: "near-cusp tiles induce oscillatory E_ff labels
      that a polar chart cannot resolve").  If a cusp-adapted coordinate
      (u = d^{2/3}) or the ghost/demodulation machinery can resolve those
      labels, the surrogate could own the window quadrature-free.

  (B) MAKE THE PEARCEY ARM SERVE THERE: the design already hands the cusp
      window to the cusp arm (`_CUSP_ARM_COVERAGE = 0.07 rad` astroid;
      `_SADDLE_CUSP_ARM_COVERAGE = 0.0` saddle — the latter a documented
      placeholder pending calibration).  The arm serves high-w via ppGO
      (quadrature-free) but has a MID-W gap: at radius_min < R < r_ppgo_min
      the uniform form does not certify, so the astroid falls to the exact
      engine (served WITH quadrature — the "bad exact flashback") and the
      deltoid refuses outright.  Measured: astroid just-outside-cusp within
      0.07 rad serves via pearcey only at w=500 (ppGO); w=100-300 refuse or
      flash to exact.  Deltoid (saddle coverage 0.0): nothing serves near
      the tip at any w.

  DECISION REQUIRED: (A) surrogate-extends-to-cusp vs (B) pearcey-fix
  (calibrate saddle coverage + close the mid-w uniform-form gap), vs a
  hybrid.  Whichever wins, the ACCEPTANCE is: exterior cusp window (both
  parities, rho 1.0 - ~2.1, on/near axis) served QUADRATURE-FREE by a fast
  path across the w band; no exact-engine flashback in the window;
  refusal-conservative.  This decision gates the fix, so it must be made
  explicitly (Professor adjudication recommended) — do not silently pick one.

  ## DRIVER MANDATE (user, 2026-08-11): EXACT-ENGINE SERVING IN THE CUSP
  ## WINDOWS IS ACCEPTANCE OF FAILURE — the fast path must own the region.

  The user's directive: the exact engine serving ANYWHERE in the cusp
  neighbourhood (interior or exterior, astroid or deltoid) is a fast-path
  failure — the point of the demodulated/ghost-subtraction surrogate (and
  of the Pearcey arm) is that the exact engine is NEVER the serving rung
  in a region a fast path can own.  The serving ladder must be:
  surrogate (incl. cusp-extended, ghost-subtracted) -> Pearcey arm ->
  ppGO -> [exact engine ONLY as a certified refusal fallback, never a
  regular serve].  The mid-w exact flashback (radius_min < R < r_ppgo_min)
  and the deltoid refusal are BOTH violations.

  ACCEPTANCE (hardened): NO serving path may deliver an exact-engine value
  in the cusp neighbourhood.  Either (A) the demodulated/ghost-subtracted
  surrogate is extended to the cusp windows (real-demodulated +
  imaginary-ghost-subtracted labels resolved there, dropping the
  `_CUSP_EXCLUSION_DISTANCE` carve-out for those tiles via a cusp-adapted
  `u = d^{2/3}` axis), or (B) the Pearcey arm is made to certify the
  mid-w window (ppGO-like geometric limit extended below r_ppgo_min, or a
  non-degenerate on-axis control).  The exact engine must be LEFT as a
  named-refusal fallback only.  This is the acceptance for the build that
  resolves this gap; do not ship a fix that leaves exact-serving in place.
