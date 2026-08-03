---
section: Backlog
---

- **RETIRE THE GRATUITOUS NUMERICAL DERIVATIVES** `[→ spec]` — **this is STEP 1
  of [[lensing_caustic_relative_coordinates]]**, and runs before every
  coordinate change in it. That fragment carries the ordering and the reason it
  is first; this one carries the inventory, the rule, and the protected list.
  Builds 1a / 1b / 1c there map onto the targets below.

  The Chang-Refsdal geometry is closed form end to end, yet five places compute
  its derivatives by differencing or sampling. One of them is on the SERVING
  path. Each imports a step-size or threshold constant that then grows its own
  safety factor, and each is a substitute for an algebra step nobody took.

  ## The rule, stated precisely — this matters more than the list

  **In the IMPLEMENTATION, numerically differentiating a closed form is a
  defect. In an ORACLE it is a virtue.** A test that checks an analytic
  Jacobian against a central difference is doing exactly the right thing: the
  numerical route is INDEPENDENT of the analytic one, which is the whole point
  of an oracle. Do not "clean up" those.

  PROTECTED, do not touch:
  - `test_lensing_prior.py` — central-difference Jacobian oracle for
    `ln_jacobian_determinant` (`H_REL = 1e-7`, `JAC_TOL = 1e-5`, both derived
    from the truncation/round-off tradeoff in that file's header).
  - `test_lensing_ghost.py` — Richardson-extrapolated central FD of the complex
    Fermat potential as the `det H_c` oracle. It carries an AST guard whose
    PURPOSE is to prove the oracle is independent of the implementation.
  - `test_lensing_fast_path.py`, `test_lensing_exterior_admission.py` — FD used
    to assert a VALUE (caustic speed; `drho/d|y| == 1` to 1e-12).

  ALSO NOT A TARGET: `likelihood.py`'s per-bin `(value, slope)` reduction and
  its `kernel_subsamples = 2` bin-edge secant. That is a PROJECTION of the
  kernel onto a per-bin linear basis — which is what relative binning IS — not
  an estimate of a derivative. The secant is the correct object there; a
  tangent would be wrong.

  ## The actual targets (surveyed 2026-07-29)

  All four compute derivatives of `geometry.critical_point`'s closed form, and
  all four fall out of the SAME cascade, which is why build 1a exports `y'` and
  `y''` themselves rather than only the scalars derived from them.

  **Extend the cascade to `y'''` (build 1a-bis or folded into 1b).** F040
  shows the cusp-exclusion half-width is not a measurement either: it is
  `w^{-1/4}` with coefficients in `|y''|` and `|y'''_perp|`. 1a delivers only
  the first two orders, so the third is owed before any cusp-window work. The
  same Taylor tail also supplies the cusp LOCATION (`y' = 0`) and the fold
  direction (`y''`), so third order closes the set: after it, no quantity in
  this package's caustic geometry is estimated rather than derived.

  1. **DONE (2026-07-30, commit `b9c3ed6`, build 1c). `_pearcey_cusp._cusp_vertex` — SERVING PATH, highest priority.** Finds
     the cusp vertex by computing caustic speed with a hardcoded central
     difference `delta = 1e-4`, scanning 129 thetas over a `pi` window (~258
     `critical_point` calls), then golden-section refining the minimum. A cusp
     is exactly `|y'(theta)| = 0`. With `y'` closed form this is a ROOT, found
     directly. Note the conditioning: near a cusp the speed vanishes linearly,
     so minimising an FD-computed speed is the worst case for both accuracy and
     cost, on a path that serves.
  2. **`surrogate_training._branch_speed_profile`** — `np.gradient` (and a
     rolled central difference on the periodic branch) over sampled caustic
     points, to get the same `|y'(theta)|`. Feeds `_find_cusps`.
  3. **`surrogate_training._find_cusps`** — consequence of 2: cusps as sampled
     speed MINIMA below `_CUSP_SPEED_REL_FRAC = 0.2` of the median, with
     `_CUSP_WIDTH_SAFETY` and an absolute `_CUSP_MIN_HALFWIDTH = 0.05` floor,
     plus the wider saddle variants. Once cusps are roots, the relative
     threshold and both safety factors lose their reason to exist. Its own
     docstring already concedes the threshold had to be relative "because the
     measured dip depth scales as ~caustic_size/n_samples" — a sampling
     artifact, described as if it were physics.
  4. **DONE (2026-07-29, commit `00bf8ae`, build 1b).** `surrogate_training._probe_arc_side` / `_PROBE_ETA` — F039.
     Retired alongside the other training-path consumers.

  5. **DONE (2026-07-30, commit `145cec3`, build 1d).**
     `surrogate_training._tube_normal` (added 2026-07-29 — MISSED by the
     original survey, found only when build 1b's `inward_sign` was measured
     against it; F041). It builds the caustic tangent as
     `critical_point(theta + 1e-6) - critical_point(theta)`: a forward
     difference of a closed form, hardcoded step, on the SERVE-consistency
     path. Replace with `tangent = y' / |y'|` from `caustic_derivatives`. The
     survey missed it because it greps as a subtraction of two function calls,
     not as `np.gradient` or a `/(2*h)` idiom — worth remembering when
     checking that this class is really gone.

  Already correct, and the model to copy: `geometry.r_caustic` samples only to
  BRACKET and refines every root with `brentq` to `4*eps`;
  `nearest_caustic_point` uses analytic Newton (`_squared_distance_derivatives`);
  `_schwinger._log_derivative` says "in closed form (never a finite
  difference)". The engine layer got this right — the training and cusp-seeding
  layers did not.

  ## Carried forward from build 1b — do not let these go quiet

  Both are open by DESIGN, not by oversight. Written down because a thing kept
  only in the driver's head dies at the next compaction.

  - **`_CUSP_SPEED_REL_FRAC = 0.2` survived 1b as an inlined local.** 1b
    deletes the module constant by name but keeps the literal `0.2` inside
    `_find_cusps`, because it no longer drives cusp DETECTION (that is now an
    analytic root) — it only measures the dip half-width that sets the cusp
    exclusion WINDOW, which 1b deliberately leaves byte-identical. That is
    honest ONLY while the window is explicitly deferred. When the cusp-window
    schema build lands (F040: the width is `w^{-1/4}` from `|y''|` and
    `|y'''_perp|`, so it is a FUNCTION and the stored
    `(theta_cusp, delta_theta)` pair cannot express it), this literal goes with
    it. A deleted constant that survives as an inlined magic number is the same
    constant with worse provenance.

  - **The exact interior-admission distance is 91-180x slower than the cloud,
    and the cheap fix is a derived bound, not the old fudge.** Measured
    2026-07-29: `nearest_caustic_point` costs 0.536 ms (positive) / 1.060 ms
    (saddle) per call against 0.0059 ms for the 200-point vectorised cloud
    minimum, at 15 probes per `admits()` call (`_INTERIOR_EDGE_SAMPLES` x 3
    band gammas). 1b ships the exact call everywhere, which is CORRECT and is
    the right default; whether it matters depends on the tile count in a full
    training run, which is unmeasured.
    MEASURE FIRST — total training wall-clock before and after 1b. Only if the
    cost is real, apply the bracket: the cloud distance always OVERSHOOTS the
    true distance (that is the fact `_CLOUD_MARGIN_FRAC` was compensating), so
    it is a one-sided bound, and the discretization gap is now computable as
    `h = caustic_speed * dtheta / 2` rather than measured. Then
    `cloud < eta_max` rejects cheaply, `cloud - h >= eta_max` admits cheaply,
    and only the narrow band between needs an exact call. That is the same
    "sample to BRACKET, refine exactly" idiom `r_caustic` already uses, and it
    replaces a measured fudge with a derived bound rather than reinstating it.
    Do NOT pre-optimise: shipping a fudge factor ahead of a measurement is how
    `_CLOUD_MARGIN_FRAC` got there in the first place.

  ## Acceptance

  - No production path in `cogwheel/lensing/` differentiates
    `geometry.critical_point` by differencing or sampling. The oracles above
    still do, deliberately, and their tests still pass unchanged.
  - Cusp angles agree with the incumbent detector to within the incumbent's own
    resolution, and are then pinned to the ANALYTIC value at 1e-10 — a root, so
    there is no sampling error left to tolerate.
  - `_cusp_vertex` costs O(1) geometry calls per serve instead of ~258, and the
    Pearcey arm's served values are unchanged to the F016 envelope bar.
  - `_CUSP_SPEED_REL_FRAC`, `_CUSP_WIDTH_SAFETY`, `_CUSP_MIN_HALFWIDTH` and the
    saddle variants are deleted or re-derived from geometry — never retuned.
    Any that survive carry a one-line reason that names a physical scale.
