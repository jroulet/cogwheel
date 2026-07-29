---
section: Backlog
---

- **CAUSTIC-RELATIVE COORDINATES — retire every prior-box length from the
  serving design** `[→ spec]` — the coverage map's regions are carved by
  `ANNULUS_INNER_RADIUS = 3.0`, inherited from the PRIOR BOX half-width. F036
  measures that no `|y|` threshold can bound the caustic at all: `r_caustic`
  diverges at the parity wall (19.8 at `gamma = 0.99` vs a 4.2426 box corner),
  so a fixed radius is crossed by the caustic somewhere in every prior. The
  four apparent serving regimes are ONE fixed boundary crossing a caustic whose
  extent varies 28x. In caustic-relative units there are TWO per parity:
  caustic-attached (interior + tube) and exterior.

  Governing principle is COVERAGE_DESIGN Part 0; the constant-by-constant
  audit is its Part IV. This fragment is the ORDER OF WORK; it does not restate
  either.

  ## Why now — a window that expires

  Every constant in the Part IV table is currently INERT with respect to any
  served value (measured 2026-07-29, detail in F036): the Born serve slot
  returns `None`, the eta/cusp constants are train-time only, and no trained
  chart artifact is shipped. So this is a pure-source change — no migration, no
  retraining, no value churn, no byte-identity gate. **It stops being true the
  moment anything is trained. Do not train until step 9.**

  ## Steps, in series

  1. **Pointwise curvature radius — CLOSED FORM.** Add
     `geometry.caustic_curvature_radius(gamma, theta, *, kappa, branch)`
     computed ANALYTICALLY: the caustic is an exact parametric curve, so
     differentiate it, do not sample it. Chain rule through `u -> r -> y`,
     then `R_c = |y'|^3 / |y1' y2'' - y2' y1''|`. It belongs in `geometry`
     because the caustic does. Then DELETE the three-point circumradius inlined
     in `surrogate_training._min_curvature_radius` and re-express that as a
     minimum over exact values.
     This step is load-bearing well beyond curvature: the same `y'`/`y''`
     cascade retires the fold-side probe (step 3b) and every target in
     [[lensing_analytic_derivatives]], including the one on the serving path.
     Export the derivatives, not just `R_c`.
     ACCEPTANCE: agrees with an independent high-precision oracle to 1e-12
     (measured 4.4e-13 over 42 cases, F038), on both parities and branches,
     including `kappa != 0`, near-axial `theta`, and near the parity wall; the
     astroid limit `R_c -> 3*gamma*|sin 2th|` holds to its own `O(gamma^2)`;
     and the consumer decision `eta_max > 0.5 * r_min` flips on NO production
     band. Do NOT assert byte-identity with the incumbent and do NOT assert its
     5-10% bias margin — both would enshrine a discretization artifact. A flip
     is a finding to report, never a number to tune.

  2. **DRIVER MEASUREMENT — the tube fraction.** Sweep held-out envelope eps
     against the DIMENSIONLESS `eta / R_c`, across gamma, both parities. Find
     `f_max` where eps crosses `TrainingConfig.tube_eps_max = 5e-2`, and
     `f_floor` likewise. NOT build work: the result goes inline into the step-3
     brief. Do NOT choose `f` to reproduce the incumbent `0.05`.

  3. **C6 — tube shell becomes curvature-relative.** `_DEFAULT_ETA_MAX` and
     `_DEFAULT_ETA_FLOOR` become `f * R_c`. Then DELETE the foot-of-normal skip
     guard: with eta_max a fixed fraction of `R_c` it is vacuous by
     construction, so it becomes an assertion, not a branch. This converts a
     refusal into a serve.
     ACCEPTANCE: no chart skipped for curvature at any gamma in the prior;
     held-out eps under bar at both gamma extremes; the same `f` serves every
     gamma. NOT in scope, and must not be claimed: the small-gamma collar is
     only PARTLY C6's. F037 measures it as three stacked causes — C6 closes the
     foot-of-normal skips (`0.0281..0.0462`, `0.0644..0.1550`) and cannot touch
     the dropped topology slivers (`< 0.0281`, `0.0462..0.0644`), which are a
     served-side detection instability, not a length scale. Acceptance is
     "tubes now serve every gamma that `stable_gamma_bands` yields a band for".

  3b. **`_PROBE_ETA` — the fold-side probe is an absolute length** (F039; this
     is F037's second cause, now diagnosed). `_probe_arc_side` steps an
     absolute `0.05` off the caustic to decide which side of a fold carries
     the image pair. When that step exceeds the local caustic half-extent the
     4-image probe fails its reconstruction check and the arc is SILENTLY
     labelled 2-image on the wrong side. Measured: shrinking the step alone,
     at fixed `n_samples`, takes `stable_gamma_bands((0.01, 0.30), +1)` from
     4 bands / 2 dropped slivers to 1 band / 0 dropped.
     DELETE the probe; do not retune it. The side is ANALYTIC (F039): at a
     critical point `J e = 0` for the soft eigenvector `e`, so the fold opens
     along `D2y[e,e]`, which is closed form from `critical_point`'s `.image`
     and `.soft_axis`. Verified 31/32 against a direct image count; the one
     miss was the image COUNTER failing to resolve a merged pair, not the
     direction. `f * R_c` is a trap: F039 measures `0.25 * R_c` flipping
     `(sign, image_count)` at gamma 0.15, 0.3 and 0.7 — bands that train fine
     today — because a curvature radius is not a caustic THICKNESS. No step
     length works, which is the signature of a question that should never have
     been asked numerically.
     ACCEPTANCE: `_PROBE_ETA` and `_probe_arc_side` are gone; the served side
     agrees with an independent image count wherever that count is
     well-conditioned; zero dropped slivers over `(0.01, 0.30)`. There is no
     step-size parameter left to be stable under.

  4. **DRIVER MEASUREMENT — the far-zone crossover.** Sweep carrier / ppGO /
     chart node cost INWARD in `rho` from the box corner, per gamma, both
     parities, for the real P5/P6 crossover `rho*`. This is an engine run:
     quote unit count x measured per-unit cost before launching.

  5. **C8 — far zone becomes caustic-relative; the annulus is retired.**
     `ANNULUS_INNER_RADIUS` -> measured `rho*`. DELETE `GAMMA_FENCE = 3/4` and
     the saddle fence `1.0502342`; both are consequences of the annulus radius,
     not independent physics. Do not port them, do not replace them with new
     fences. Re-express `surrogate_census`'s 6-way MECE breakdown in `rho`.
     RENAME `ppgo_map.annulus_rho` in this same build — a public symbol named
     for a retired concept is how the concept survives its own deletion.
     ACCEPTANCE: no gamma in the prior yields a "boundary cuts the caustic"
     region; regime count is TWO per parity; coverage-map regions 6-9 collapse
     to one row.

  6. **C5 — ghost decay gate.** Independent of the coordinate work and owed on
     both branches regardless (F027): the ghost branch needs a decay gate, not
     just a separation test.

  7. **`_GHOST_SEPARATION_MIN = 0.7` — the suspect.** Ask the Part 0 question.
     F027 showed it never binds on the saddle. Re-derive as relative, or
     delete. The test-heavy step: 22 references across
     `test_lensing_ghost_gate.py` and `test_lensing_exterior_windows.py`.

  7b. **The remaining sampled estimators.** Surveyed 2026-07-29 while briefing
     step 1; each computes by sampling something that has a closed form or an
     exact solver already in the package. Ask the Part 0 question of the
     METHOD, not just the constant.
     - `_CLOUD_MARGIN_FRAC = 0.10` — a round number inflating a refusal
       threshold to cover a MEASURED ~8% overshoot of the discrete 200-point
       `_caustic_points` cloud, when `geometry.nearest_caustic_point` is exact
       (9.3e-12) and already imported in the same file. Its own docstring says
       the margin buys not "densifying the cloud or spending extra oracle
       calls". It is applied INTERIOR-ONLY; the exterior path carries the same
       slop uncompensated, protected only by a larger margin — two paths, one
       corrected, an F019-shaped trap.
     - `_find_cusps` — cusps as sampled caustic-SPEED minima below a relative
       threshold, with `_CUSP_WIDTH_SAFETY` and an absolute
       `_CUSP_MIN_HALFWIDTH = 0.05` floor. A cusp is exactly `|y'(theta)| = 0`,
       and `y'(theta)` is closed form after step 1, so cusp angles are roots
       findable to machine precision and the safety factors lose their reason
       to exist.
     - `_caustic_inradius` — `min |y(theta)|` over the same cloud; likewise a
       closed-form minimisation.
     The MODEL to copy is already in the package: `geometry.r_caustic` samples
     only to BRACKET, then refines every root with `brentq` to `4*eps`, and
     says so in its docstring ("bracketing density does not set the returned
     radius accuracy"); `nearest_caustic_point` uses analytic Newton;
     `_schwinger._log_derivative` is "in closed form (never a finite
     difference)".
     ACCEPTANCE: no constant in `surrogate_training` exists to compensate for
     a discretization error. Deleting `_CLOUD_MARGIN_FRAC` changes no
     admission decision, because the distance it corrects is now exact.

  8. **Make Part 0 mechanical.** A test asserting that no length-unit float in
     `cogwheel/lensing/` traces to the prior box, and that no live document or
     public symbol names a retired concept. Extend it to the METHOD form of
     the question: no constant in the geometry or training path may exist to
     absorb a discretization error, and no decision with a closed form may be
     taken by stepping. This bug class arrived by accretion, one plausible
     constant at a time; only a test stops it returning.

  9. **Then train — once**, in final coordinates, on the final engine and chart
     set. Cost estimate first; full-suite gate green first.

  ## Standing rules for whoever executes this

  These exist because the last pass changed code and left the prose, and the
  prose is what the next agent reads.

  - **Surfaces are build SCOPE, not cleanup.** Each brief names its spec and
    test surfaces. A build that moves a boundary and leaves a live document
    describing the old one FAILS its acceptance, exactly like a red test.
  - **Default is DELETE, not re-point.** For a test pinning a boundary that no
    longer exists as a concept, changing `3.0` to `rho*` IS the scar — it
    preserves the shape of the wrong idea. Re-point ONLY if the test asserts a
    value that survives the coordinate change. Expect deletion to dominate in
    `test_lensing_born.py` (52 tests, 70 lines on doomed names).
  - **Archive is off-limits.** `COMPLETED.md`, `CHANGELOG.md`,
    `SPEC_CHANGELOG.md`, `DATA_CONTRACTS_CHANGELOG.md` and their `.d/`
    fragments record what was true when written. Never edit them to match the
    new design. `FINDINGS.md` is the middle case: findings stay, but a
    superseded SCOPE gets a pointer to its successor (see F032 -> F035), never
    a silent edit.
  - **Ask Part 0 of the METHOD, not only of the constant.** The governing
    principle generalises: for every DECISION, ask what determines it, and
    whether that determination is analytic. This geometry is closed form
    end to end — the caustic, its derivatives, its curvature, and which side
    of a fold carries the image pair are all algebra at the critical point.
    A sampled estimator or a probe step in that setting is not an
    approximation of the answer, it is a SUBSTITUTE for having derived it,
    and it drags in a step-size constant that then needs its own tuning,
    safety factor and margin. Three constants died to this in one sitting
    (F038 `_min_curvature_radius`, F039 `_PROBE_ETA`, and the
    `_CLOUD_MARGIN_FRAC` family in step 7b). The tell is a constant whose
    docstring explains a discretization error rather than a physical scale.
  - **Never preserve an incumbent number by construction.** Each replaced
    constant gets a measured or derived value; matching the old one is a
    coincidence to report, not a target.
  - **Never add a fence to make a step pass.** Two fences are being deleted
    here precisely because they were consequences of a bad boundary.
  - Slow tests never run inside a build; steps 3 and 5 each get a fast in-build
    gate plus a post-build driver sweep.

  ## Live surfaces in scope (measured 2026-07-29)

  Tests: `test_lensing_born.py` (epicentre), `test_lensing_surrogate_training.py`,
  `test_lensing_ghost_gate.py`, `test_lensing_exterior_windows.py`.
  Spec: `SPEC.md`, `COVERAGE_DESIGN.md` (Parts I and IV), `DATA_CONTRACTS.yaml`,
  `data_registry.yaml`, and the fragments [[lensing_coverage_map]],
  [[lensing_born_b1_derivation]], [[lensing_saddle_born]],
  [[surrogate_component-representation-8hb]].
  Docs: `docs/source/generated/cogwheel.lensing.ppgo_map.annulus_rho.rst`
  follows the rename in step 5.
