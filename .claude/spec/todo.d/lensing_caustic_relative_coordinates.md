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

  1. **THE ANALYTIC SWEEP — go through the geometry with a fine-toothed comb
     and derive, rather than estimate, everything that has a closed form.**
     This is FIRST, ahead of every coordinate change, and it is a phase of
     three sequential builds rather than one. Detail, inventory and the
     implementation-vs-oracle rule live in [[lensing_analytic_derivatives]];
     this entry is only the ordering and the reason.

     WHY FIRST. Every later step measures or re-expresses something in terms
     of caustic geometry, so each one either inherits exact derivatives or
     re-derives them badly. Two of the plan's own steps were already mis-
     specified because the geometry underneath was numerical: this step's own
     first draft demanded byte-identity with a biased estimator (F038), and a
     separate step proposed retuning a probe step that should not exist at all
     (F039). Both dissolved once the algebra was done. Doing this last would
     mean re-opening finished steps; doing it first means every later
     acceptance is stated against exact geometry.

     1a. **DONE (2026-07-29, commit `1a82046`).** Shipped as `geometry.py`'s
        `caustic_derivatives` (`y'`, `y''`), `caustic_speed`, and
        `fold_opening_direction`; `caustic_curvature_radius` derived from
        `caustic_derivatives` per the acceptance below. Measured against a
        two-stage oracle (F038's single-stage version was circular): `y'`
        worst relative 4.39e-13, `y''` worst relative 2.56e-14, 0 failures at
        atol=5e-13 + rtol=1e-11 over 110 configs on both parities and
        branches, including `kappa != 0`, near-axial `theta` and near the
        parity wall; fold direction 16/16 to the correct side, unit to 1e-12;
        `|y'|` at the astroid cusp = 1.3e-16 (cusps are now exact roots). See
        `completed.d/2026-07-29_analytic_caustic_derivatives_1a.md`.

     1b. **The training-path consumers.** Retire, against 1a:
        `_min_curvature_radius`'s three-point circumradius and its
        `area2 < 1e-30` guard; `_branch_speed_profile`'s `np.gradient`;
        `_find_cusps`'s speed-minimum detection with its relative threshold and
        two safety factors; `_probe_arc_side` and `_PROBE_ETA` entirely;
        `_caustic_inradius`'s cloud minimum; and `_CLOUD_MARGIN_FRAC`, which
        exists only to absorb the discreteness of a caustic cloud that
        `nearest_caustic_point` already resolves exactly.
        ACCEPTANCE: the `eta_max > 0.5 * r_min` decision flips on NO production
        band; `stable_gamma_bands((0.01, 0.30), +1)` drops zero slivers AND
        every stable band it returns yields `len(arcs) > 0`; deleting
        `_CLOUD_MARGIN_FRAC` changes no admission decision, because the
        distance it corrected is now exact. Do NOT assert byte-identity with
        any incumbent estimator — that enshrines its discretization.
        The arcs-non-empty clause is not redundant (F041): the first 1b
        attempt satisfied "no dropped slivers" more nearly than before while
        returning ZERO arcs on two small-gamma bands, because
        `_make_arc`'s new `abs(dot) > 0.1` guard is an ABSOLUTE cut on a
        quantity measured to scale as `~1.5*gamma`. A band that exists but
        produces nothing is unserved exactly like a dropped one; the
        acceptance must say so or it is satisfiable while the collar stays
        shut. And `|dot|` is the WRONG QUANTITY, not merely a mis-scaled one:
        it measures how TRANSVERSE the fold opening is, which is a legitimate
        gamma-dependent geometric fact, not a pathology. What the guard wanted
        is cusp proximity. Use a DIMENSIONLESS RATIO OF LOCAL QUANTITIES:
        since `theta` is dimensionless, `|y'|` and `|y''|` both carry length,
        so `|y'| / |y''|` IS the angular distance to the cusp (where
        `y' = 0`), and dividing by the arc half-span makes it fully
        arc-relative. Measured, that ratio is gamma-STABLE — 0.307, 0.310,
        0.313, 0.324, 0.319 at matched cusp proximity across
        `gamma = 0.02..0.9` — while `|dot|` swings 33x over the same range.
        A dimensionless O(1) constant on a scale-free ratio is the shape every
        guard in this package should have; an absolute cut on a
        dimensionful local quantity is the bug class this whole step exists to
        remove.
        The sign needs no protection at all: swept over the prior the minimum
        `|dot|` is 4.4e-3 at `gamma = 0.01`, twelve orders above the float
        noise floor, so it never approaches ambiguity. Deleting the guard
        outright is defensible, since the arc bounds already exclude the cusp
        windows — measured `|y'|/|y''|` over arc half-span stays >= 0.39
        everywhere sampled. If a guard is kept, it must be the ratio.

     1c. **The serving path, plus third order.** `_pearcey_cusp._cusp_vertex`
        locates a cusp by differencing caustic speed at a hardcoded
        `delta = 1e-4` over a 129-point scan plus a golden-section refine. A
        cusp is `|y'| = 0`, a root. Separate build because it SERVES: it needs
        the F016 envelope bar, not just a geometry tolerance.
        Extend `caustic_derivatives` to `y'''` in this same build — it is the
        cusp build, and F040 shows the cusp-exclusion half-width is
        `w^{-1/4}` with coefficients in `|y''|` and `|y'''_perp|`. Third order
        closes the set: location (`y' = 0`), fold direction (`y''`) and width
        (`y'''`) then all come from one Taylor tail, and nothing in the
        caustic geometry is estimated rather than derived.
        ACCEPTANCE: served Pearcey values unchanged to the F016 bar; cusp
        angles pinned to the analytic root at 1e-10; O(1) geometry calls per
        serve instead of ~258; `y'''` verified against the same two-stage
        oracle at the F038 tolerance.

     1d. **DONE (2026-07-30, commit `145cec3`).** `_WEDGE_EPS = 1e-3` —
        DELETE it, and close the estimator
        inventory. This entry asked for a conditioning bound
        (`Δθ >= |y''|_max^{-2/3}`) on the theory that the wedge edge is a
        singularity to stand off from. Driver measurement 2026-07-30 (F044)
        says it is not: the edge is a REGULAR point of the caustic — `y` and
        `dy/ds` are both finite and nonzero in `s = sqrt(θ_max − θ)` — and
        the `Δθ^{-1/2}` / `Δθ^{-3/2}` divergences are artifacts of the `θ`
        parametrization alone. `critical_point` serves the edge exactly and
        refuses by name immediately outside it, so the standoff adds no
        safety; what it does add is an excised sliver that leaves
        `_lobe_winding_loop` open by 0.279 at `gamma = 1.05` (9.3% of lobe
        reach) while `_winding_number` uses it as the saddle interior
        admission test. Sampling the wedge CLOSED takes that gap to exactly
        zero and changes nothing else. So: no derived constant — deletion,
        per the standing rule's default.
        Fold in the last estimator on the inventory,
        `_tube_normal`'s `dth = 1e-6` forward difference (target 5 in
        [[lensing_analytic_derivatives]]). It is the F041 surface, so
        `inward_sign` is the tripwire.
        ACCEPTANCE (met): `_WEDGE_EPS` gone with no inlined replacement, loop
        closure gap exactly 0.0, no shrink in cusp/arc/reach/arc-span at any
        saddle band; `_tube_normal` analytic with `inward_sign` unchanged on
        every production arc. After this, nothing in the caustic geometry is
        estimated rather than derived — the claim step 1 exists to earn.
        See `completed.d/2026-07-30_wedge_edge_deletion_1d.md`.

     1e. **The interpolation COORDINATE — full detail in
        [[lensing_collocation_from_local_scales]].** Moved here from a step-9
        prerequisite (F042): the `theta` axis is gridded and splined in an
        ABSOLUTE coordinate, so held-out eps depends on the grid, not just the
        geometry — measured 0.059 uniform-theta vs 0.027 arc-length at the
        same nodes, and knife-edge under a small arc-bound shift. This MUST
        precede step 2, because step 2 (and step 4) MEASURE held-out eps / node
        cost, and on a uniform-theta grid they would pin `f_max`/`f_floor` and
        the far-zone crossover to placement artifacts. It is not node placement
        but a COORDINATE CHANGE (spline IN the coordinate where the demodulated
        envelope is smooth, as `u = sqrt(eta)` already does for the fold axis;
        the principled coordinate is the catastrophe-uniformizing one, arc-
        length its cheap stand-in). Capstone of the analytic phase: the last
        place the cascade's geometry is baked into the surrogate's own
        coordinate system, before any measurement runs on it.
        ACCEPTANCE: held-out eps insensitive to a small arc-bound shift (F042
        knife-edge gone) — the property node placement alone cannot achieve.
        SIX sub-builds, not three: 1e-tube (DONE), 1e-farfield (DONE
        2026-07-31), 1e-lobe cover
        the SPATIAL axis only. The `eta`, `w` and `gamma` node measures are
        equally part of step 9's prerequisite and were scheduled nowhere until
        2026-07-30; they are now 1e-eta, 1e-w and 1e-gamma in
        [[lensing_collocation_from_local_scales]], which carries the ordering.
        1e-gamma runs AFTER step 5 (C8), since it uses the caustic-relative
        coordinate C8 establishes.

  2. **DRIVER MEASUREMENT — the tube fraction.** DEPENDS ON 1e: run only after
     the interpolation coordinate is settled, or the eps this sweeps is a grid
     artifact (F042). Sweep held-out envelope eps against the DIMENSIONLESS
     `eta / R_c`, across gamma, both parities. Find `f_max` where eps crosses
     `TrainingConfig.tube_eps_max = 5e-2`, and `f_floor` likewise. NOT build
     work: the result goes inline into the step-3 brief. Do NOT choose `f` to
     reproduce the incumbent `0.05`.

  3. **DONE (2026-08-01). C6 — tube shell becomes curvature-relative.**
     `TrainingConfig.eta_max`/`eta_floor` renamed to `f_max`/`f_floor`;
     per-arc `eta_max = f_max * R_c` computed in `_train_band_charts` and
     threaded to `_build_tube_chart`, `_tube_heldout_samples`,
     `_interior_admission`, `_saddle_lobe_admissions`. The foot-of-normal skip
     guard deleted; replaced by `assert f_max < 0.5`. No chart is skipped for
     curvature — the old branch becomes vacuous by construction.

  4. **DRIVER MEASUREMENT — the far-zone crossover.** Sweep carrier / ppGO /
     chart node cost INWARD in `rho` from the box corner, per gamma, both
     parities, for the real P5/P6 crossover `rho*`. This is an engine run:
     quote unit count x measured per-unit cost before launching.

  5. **DONE (2026-08-02, commit 65eebcb). C8 — far zone becomes caustic-relative; the annulus is retired.**
     `ANNULUS_INNER_RADIUS` -> measured `rho*`. DELETE `GAMMA_FENCE = 3/4` and
     the saddle fence `1.0502342`; both are consequences of the annulus radius,
     not independent physics. Do not port them, do not replace them with new
     fences. Re-express `surrogate_census`'s 6-way MECE breakdown in `rho`.
     RENAME `ppgo_map.annulus_rho` in this same build — a public symbol named
     for a retired concept is how the concept survives its own deletion.
     ACCEPTANCE: no gamma in the prior yields a "boundary cuts the caustic"
     region; regime count is TWO per parity; coverage-map regions 6-9 collapse
     to one row.

  6. **DONE (2026-08-02, commit e456871). C5 — ghost decay gate.** Independent of the coordinate work and owed on
     both branches regardless (F027): the ghost branch needs a decay gate, not
     just a separation test.

  7. **DONE (2026-08-02, commit e5b56b8). `_GHOST_SEPARATION_MIN = 0.7` — cleared through Part 0 (Build 7):** Ask the Part 0 question.
     F027 showed it never binds on the saddle. Re-derive as relative, or
     delete. The test-heavy step: 22 references across
     `test_lensing_ghost_gate.py` and `test_lensing_exterior_windows.py`.

  8. **Make Part 0 mechanical.** A test asserting that no length-unit float in
     `cogwheel/lensing/` traces to the prior box, and that no live document or
     public symbol names a retired concept. Extend it to the METHOD form of
     the question: no constant in the geometry or training path may exist to
     absorb a discretization error, and no decision with a closed form may be
     taken by stepping. This bug class arrived by accretion, one plausible
     constant at a time; only a test stops it returning.

  9. **Then train — once**, in final coordinates, on the final engine and chart
     set. Cost estimate first; full-suite gate green first.
     PREREQUISITE: [[lensing_collocation_from_local_scales]]. Training bakes
     the node layout into a shipped artifact, and today every chart axis is a
     uniform grid in a coordinate that does not know the local scale — a bare
     count against the wrong measure. The scales are all closed form from the
     step-1 cascade (arc length and curvature for `theta`, the Airy argument
     for `eta`, the envelope's own variation for `w`, caustic-relative for
     `gamma`), so this is derivation, not a sweep. Getting it wrong does not
     cost speed, it costs COVERAGE: nodes uniform in the wrong measure starve
     the region that sets the held-out eps, and the historical response to
     that has been to widen an exclusion window and serve less.

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
    safety factor and margin. This is why step 1 comes first and is a phase,
    not a task. The tell is a constant whose docstring explains a
    discretization error rather than a physical scale — `_CLOUD_MARGIN_FRAC`
    and `_CUSP_SPEED_REL_FRAC` both do, in as many words.
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
