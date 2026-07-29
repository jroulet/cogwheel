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

  1. **Pointwise curvature radius.** Add
     `geometry.caustic_curvature_radius(gamma, theta, *, kappa, branch)` — the
     three-point circumradius currently inlined in
     `surrogate_training._min_curvature_radius`, evaluated AT a point rather
     than minimised over a band. It belongs in `geometry` because the caustic
     does; re-express the band-min as a thin wrapper.
     ACCEPTANCE: matches an independent symbolic/high-dps curvature oracle
     (sympy 1.14.0 and mpmath 1.3.0 are both in the env) to 1e-8; the
     small-gamma astroid limit `R_c -> 3*gamma*|sin 2th|` is a scale/sign
     check only, good to 4.4e-5 — F038. The rewritten band-min
     does NOT reproduce the incumbent: F038 measures the circumradius estimator
     biased HIGH by 4.9-9.6% on production bands, because a three-point stencil
     cannot reach the arc endpoints where the true minimum sits. Assert instead
     that the exact band-min is BELOW the incumbent by that measured margin and
     that the consumer decision `eta_max > 0.5 * r_min` flips on NO production
     band. A flip is a finding to report, never a number to tune.

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

  3b. **The dropped topology slivers** (new, from F037). `band_caustic_structure`
     reports the arc's `(inward_sign, image_count)` flipping between band edges
     at `gamma < 0.07` — identically at `n_samples` 200, 800 and 3200, so it is
     not a resolution problem. Bisection recurses to `min_gamma_band` and drops
     the sliver silently-but-loudly. Find the served-side detector's small-gamma
     failure and fix it, or establish that the flip is real physics and the
     drop is correct.
     ACCEPTANCE: `stable_gamma_bands((0.01, 0.30), +1)` returns zero dropped
     slivers, or a stated reason why a drop is the right answer.

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

  8. **Make Part 0 mechanical.** A test asserting that no length-unit float in
     `cogwheel/lensing/` traces to the prior box, and that no live document or
     public symbol names a retired concept. This bug class arrived by
     accretion, one plausible constant at a time; only a test stops it
     returning.

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
