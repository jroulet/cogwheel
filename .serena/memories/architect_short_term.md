Last session: 2026-08-04 production batch. Clean.

2026-08-06 brief_wedge_cusp_axis_and_subdivision [PLAN DONE]: 2 Coder WPs +
  domain tests. Replace wedge chart's cusp-singular ARC-LENGTH angular axis
  with per-tile cusp-adapted u=d^(2/3); give _wedge_interior_tiles >=2 angular
  columns (split at pi/4) + adaptive subdivision in u on eps fail.
  KEY DESIGN (Prof+Simplifier concur):
  - REUSE existing theta_to_s (2,N) field/np.interp/spline/serialize/serve
    machinery to store the u-map (theta_fine, u_fine); serve is GENERIC in
    theta_to_s (zero serve-code change). Only map CONTENTS + schema string +
    load-tightening change. (Simplifier LEAN; new field = 4 touch points for 0
    correctness.)
  - Build u from PURE ANGLE, NO caustic_speed (Prof): lower tile [.,pi/4]
    u=theta^(2/3); upper [pi/4,.] u=-(pi/2-theta)^(2/3); then u-=u[0] (affine
    shift -> validator-clean: strictly incr, starts~0). Gamma-INDEPENDENT ->
    kills the rep_gamma approximation the arc map had. Existing
    _validate_theta_to_s passes UNCHANGED (theta_fine[0]==theta_grid[0],
    s_fine[0]~0). Side chosen by theta_wedge_range midpoint vs pi/4.
  - Schema bump _WEDGE_AXIS_SCHEMA v1->v2 + shrink _KNOWN set to {v2} so stale
    s-axis artifacts HARD-REFUSE at load (mechanism already wired). Tighten
    wedge load: theta_to_s REQUIRED (not optional) under new schema.
  - Node placement: STAY uniform-theta (Simplifier TRIM; Prof: uniform-u is
    2nd-order nice-to-have, uniform-theta reaches 3.42e-4 floor @9 nodes).
  - Subdivider: NEW wedge fn (mirror _subdivide_farfield_tile, DON'T generalize
    -246-line ff subdivider is exterior-(rho,theta_c)-specific). Split at
    u-midpoint -> theta_mid=inverse(u_mid) (lower theta_mid=u_mid^1.5; upper
    theta_mid=pi/2-(u_mid_d)^1.5), 2 children each rebuilt via _build_wedge_chart
    with own theta_wedge_range (auto-gets correct side map). Angular-only
    (defect is provably angular; radial rows already pass 3.82e-4). Single-level;
    re-fail or basin-flip child -> ladder-served gap (mirror lobe).
  - Tiler emits THETA bounds; subdivider converts to u internally (Simplifier).
  - pi/4 CLOSED-HALF seam: own diagonal once, no double/zero serve; add seam-
    continuity test (lower vs upper served value at pi/4 agree within eps).
  TOLERANCES (Prof authority; SACR-C interior ~0.4ms/w-node batched):
    grid n_gamma=2,n_r=3,n_theta in {5,9,17},n_w~12,n_heldout~40 -> seconds.
    Accuracy gate p90<=3.42e-4 (ffin baseline); CONTRAST eps_s/eps_u>=50 on
    same samples (proves AXIS is the lever). Subdivision: parent~5e-3
    (parent/bar in [10,30]) fails -> both children<=3.42e-4. Node-exact ON-node
    <=1e-7 (interp-map budget NOT 6.33e-16), OFF-node within eps bar. Report
    p50/p90/max + WORST-SAMPLE LOCUS never bare max.
  WPs: WP1 surrogate.py (from_wedge_engine u-map + schema + load + docstrings;
  serve unchanged). WP2 surrogate_training.py (_wedge_interior_tiles 2 cols +
  wedge subdivider + wire into build loop). WP2 depends_on WP1.
  Test shards (disjoint, each names file, F057): A=test_lensing_wedge_dd_arclength.py
  (axis: u-map shape/monotone/orientation/gamma-indep, contrast, node-exact,
  v1-hard-refuse; PORT existing arc-length tests, keep DD-ceiling). B=
  test_lensing_interior_wedge_chart.py (tiler >=2 cols, subdivision, seam,
  accuracy p50/p90/max). C=test_lensing_ppgo_bandsplit.py (reconcile
  _wedge_interior_tiles consumer to multi-col; derive counts dynamically).
  has_domain_changes=true. DATA_CONTRACTS: axis_schema is on SHIPPED surrogate
  (consumers=8) -> Librarian post-build changelog note (no training in-build).

2026-08-06 brief_mpmath_band_tests [PLAN DONE]: is_test_only=true, 0 Coder
  WPs, 3 mandated shards. Prof levers: (1)DDW dd_cap=58/(r_max*reach_max)
  monotone-dec; r_max<1 hard (interior), so ALSO widen DD_GAMMA_RANGE up
  (reach grows w/ shear, stay <1) to get r_max*reach>1 => dd_cap<60; keep
  DD_W_RANGE[1]=500 above cap so it still binds; keep some nodes unrefused;
  file has STALE cost-budget docstring. (2)marg: lower m_lens_msun<90 (w prop
  M), non-vacuity survives (gamma-driven refusal unchanged), leave margin
  over 200 draws. (3)prior mutation: steer refusal-search to gamma~1 det-A
  STRUCTURAL refusal (no Schwinger call) — cheapest. (4)saddle band-limit:
  try near-fold geom + modest |y| + mass x2-3 so LensedBinningError trips at
  w<=60 (binning cares w*dtau span, Schwinger w*|y| offset — decouplable);
  slow-tier behind COGWHEEL_TRAIN_TIER w/ ~100s cost comment if it can't land
  in 2 iters. Simplifier: guard=Candidate C (patch _f_schwinger_mpmath raise,
  exercise ONLY the 4 offending fixtures, assert calls==0; NOT whole-file
  rerun, NOT subprocess). Cost-comment fix = inline part of each shard, not a
  separate deliverable. No standalone verify-suite WP (driver/Inspector job).
2026-08-06 brief_mpmath_band_tests (PLAN v2, Prof rulings):
- Prof Ruling 1: DDWCeiling geometry-to-cap<60 is likely INFEASIBLE
  (astroid reach_max<~0.7, r_max<1 => min cap ~88, cannot reach <60).
  Give shard1 PRIMARY geometry attempt + PROMINENT FALLBACK = slow-tier
  the setUpClass (COGWHEEL_TRAIN_TIER) w/ corrected ~100s cost comment;
  class purpose intrinsically involves w>60 (brief rule 2).
- Prof Ruling 2 (DECISIVE, verified _schwinger.py:938-980):
  SchwingerCertificationError FIRES in FAST double-double band (w<=60)
  at |a|=|1-gamma'|->0 parity pinch. So shard2 prior test: push gamma'
  ->1 at LOW mass, keep all draws w<=60, still collect the vocabulary.
- Prof Ruling 3: LensedBinningError(w*dtau) vs Schwinger(w,|y|) decouple
  near a fold; try near-fold low-mass first (~2 iters) else slow-tier.
- Prof Ruling 4: keep box straddling certified/refused via gamma'/source
  (mass only sets w) so both finite & -inf present (C7: ~41/59).
- Prof Ruling 5: assert DD product vs chart's OWN interpolated reach_max
  (bilinear O(h^2)); loosen product tol ~1e-3 if oracle recomputes reach.
- Simplifier: trim patch-guard OVERRIDDEN (Acceptance #2 mandates it).
- Route: is_test_only=true, 0 WPs, 3 disjoint shards. Prior notes:
- Four fast-tier tests wander into f_schwinger mpmath band w in (60,150]
  (~85-120s/eval, F061). NO production change (f_schwinger + both ceilings
  frozen). ALL fixes are in TEST FILES (fixtures/geometry/cost comments)
  + one new guard test in test_lensing_schwinger.py -> is_test_only=true,
  zero Coder WPs, per-suite Test Dev shards.
- DDWCeilingTestCase: change GEOMETRY so DD cap (58/(r_max*reach_max))
  lands <60; formula assertions are geometry-independent (its own docstring
  says "verify the FORMULA not the success rate").
- Existing slow-tier mechanism = _MPMATH_TIER_SKIP (skipUnless
  COGWHEEL_TRAIN_TIER) in test_lensing_schwinger.py:1989. Use it if a
  test genuinely needs w>60.
- M_LENS_MSUN=90 shared (marginalized imports into saddle); lens w scales
  with m_lens -> mass lever. Don't prescribe numbers; Test Dev iterates.

2026-08-06 brief_wire_interior_wedge_chart:
- from_wedge_engine + _from_wedge_fixed + wedge NPZ round-trip ALREADY
  complete in surrogate.py (build 56a223a). Brief is STALE claiming it's
  missing. Only surrogate_training.py wiring is genuinely absent.
- _interior_admission MUST BE KEPT: live exterior-tiler dependency
  (surrogate_training.py:3949) + 5 test suites. Brief's "interior-only,
  move/delete" premise is WRONG.
- _farfield_interior_tiles genuinely dead after swap -> DELETE (but ported
  by 2 test suites: exterior_windows, ppgo_bandsplit).
- Professor rulings: 1 angular column [0,pi/2] (carrier smooth through
  pi/4, empirically confirmed by test_lensing_wedge_dd_arclength), uniform
  n_per_side radial rows, r_min>0, r_extent capped below 1 by tube shell
  (leave Airy edge to tube); in-build eps gate = ABSOLUTE floor <5e-2 +
  chart-count (ffin relative baseline is driver post-build since ffin is
  deleted).
- Simplifier trims: no 2D tiler, inline/minimal radial split; no verify-
  only WP.
- Single Coder WP (all edits in surrogate_training.py; multiple WPs on
  _train_band_charts would conflict). _heldout_eps annotation add =
  biggest risk (G3).
- Gated/flip wedge tiles -> ladder-served gap (mirror LOBE, NOT ffin
  subdivision).
