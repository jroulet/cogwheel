---
section: Backlog
depends_on: []
---

- **Fold arm's `_merging_fold_pair` drops the third coalescing image on the cusp symmetry axis**
  `[→ spec]` — measured 2026-08-11 by the driver (user-flagged).

  On the astroid cusp SYMMETRY AXIS (delta_parallel = 0), an interior source
  with rho ~ 0.42-0.57 has THREE coalescing images (e.g. gamma=0.5,
  src=(0.7,0): tau=1.182 [min], 1.193 [saddle], 1.193 [saddle] — the two
  saddles are the degenerate symmetric pair).  This is the CUSP catastrophe
  (3 comparable images), not a fold.

  `_merging_fold_pair` (`_airy_fold.py`) picks the delay-adjacent
  min/saddle pair (1.182, 1.193) and SILENTLY DROPS the third image (the
  symmetric saddle at the same tau).  The fold arm then certifies a 2-image
  Airy uniform form against a 3-image reality; the uniform-error estimate
  blows up (measured 12.5 vs bar 0.05 at w=200) and it refuses.  The refusal
  is "correct for a fold" but the source is a CUSP — it should be served by
  the Pearcey arm.

  The cusp arm ALSO refuses on the axis: `delta_parallel = 0` makes the
  Pearcey control x = 0, mapping the source to the 1-stationary EXTERIOR
  regime (n_stat = 1), so the interior calibration bypass (len==3) never
  fires.  Measured: on-axis (0.7,0) refuses via both arms at all w; a
  small off-axis kick (0.7, 0.05) serves via both arms.

  So the cusp-symmetry-axis "teardrop neck" is a measure-zero serving gap
  where BOTH uniform arms decline and the node falls to the exact engine.
  Root causes: (a) fold pair selection does not detect a 3-image cusp
  cluster, (b) the cusp arm's x=0 control degeneracy on-axis.

  PRE-EXISTING: the build only touched `_pearcey_cusp.py` calibration
  bypass + ppGO gate; `_airy_fold.py` is untouched, and the cusp-arm
  degeneracy predates the bypass.  Not caused by the build.

  ACCEPTANCE: an interior source on the cusp symmetry axis (the teardrop
  neck) is served by a fast path (Pearcey arm, with a non-degenerate
  control mapping for delta_parallel ~ 0), OR the fold arm correctly
  detects the 3-image cluster and declines to the cusp arm, OR the gap is
  documented as an accepted exact-engine fall-through with a value-pinned
  test.  Refusal-conservative; no live quadrature.

  ## MORAL FINDING (driver, 2026-08-11): the on-axis cusp SHOULD serve, but the
  ## current control mapping CANNOT represent it — the refusal is protective.

  The user pressed the moral case: an interior source on the cusp symmetry
  axis (3 coalescing images) should be served by the Pearcey arm.  Verified
  with a forced-serve experiment: building the cusp uniform form at x=0
  (bypassing the n_stat refusal) gives |F| = 4.33 vs exact 3.52 at w=40
  (23% error), 2.54 vs 1.92 at w=60 (32%), and the reduced cluster-matching
  FAILS (matched = 0/4: the single real stationary phase -1.641 matches no
  image delay; the cluster phases are -0.14/+0.27/+0.27).

  ROOT: `delta_parallel = 0` on the axis makes the Pearcey control
  `x = delta_parallel·w^{1/2}/sqrt(|C4|) = 0`, forcing P(0, y) onto its
  1-stationary EXTERIOR branch — structurally incapable of representing the
  3-image interior cluster.  So the cusp arm's refusal is CORRECT given the
  mapping; the MAPPING is the defect.

  FIX DIRECTION (follow-up): a non-degenerate on-axis control — e.g. the
  cusp-adapted angular coordinate `u = d^{2/3}` used by the surrogate's
  wedge/lobe charts (`_wedge_cusp_axis_map` / `_lobe_cusp_axis_map`), or a
  2nd-order/rotated control that keeps `x` off zero when
  `delta_parallel ~ 0`.  Acceptance: on-axis interior cusp sources serve
  with the same tolerance as the off-axis generic case.
