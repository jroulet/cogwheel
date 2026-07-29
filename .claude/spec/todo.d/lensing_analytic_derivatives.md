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

  1. **`_pearcey_cusp._cusp_vertex` — SERVING PATH, highest priority.** Finds
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
  4. **`surrogate_training._probe_arc_side` / `_PROBE_ETA`** — F039. An
     absolute `0.05` step decides which side of a fold carries the image pair,
     and the answer moves with the step. Analytic replacement is `D2y[e,e]`,
     the same cascade as the rest. Build 1b, alongside the other
     training-path consumers.

  Already correct, and the model to copy: `geometry.r_caustic` samples only to
  BRACKET and refines every root with `brentq` to `4*eps`;
  `nearest_caustic_point` uses analytic Newton (`_squared_distance_derivatives`);
  `_schwinger._log_derivative` says "in closed form (never a finite
  difference)". The engine layer got this right — the training and cusp-seeding
  layers did not.

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
