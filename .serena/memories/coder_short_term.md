# Coder Short-Term Observations

## WP1 (Build 8h-b4): per-theta_c-column exterior admission (positive parity)
- surrogate_training.py ONLY. Replaced the scalar exclusion_rho exterior tiling
  test (parity==1 only) with per-column directional admission mirroring the S2-1
  interior tiler. Saddle (parity==-1) path + `_farfield_tiles` +
  `ppgo_exclusion_rho` UNTOUCHED; interior `admits` byte-identical to HEAD.
- 4 edit locations (WP "Where" named 3; I added the 4th, see below):
  * `_InteriorAdmission.admits_exterior(center, half, source_magnitude_max)`
    (~L1478): probes INNER rho edge rho_inner=center[0]-half[0] across
    _INTERIOR_EDGE_SAMPLES=5 thetas over the tile theta span; per band gamma
    interp directional radii; ADDITIVE positive-parity reconstruction
    y_mag = radii + rho_inner - 1.0 (rho>1 arm of _from_caustic_fixed). True iff
    EVERY probe: (1) rho_inner>1.0, (2) nearest caustic-CLOUD dist>=eta_max (same
    cloud `admits` uses), (3) y_mag<=source_magnitude_max. NOTE: added
    source_magnitude_max as explicit param (frozen dataclass can't carry the
    per-region prior box extent; brief's sig omitted it).
  * `_farfield_exterior_tiles(rho_outer, n_per_side, *, admission,
    source_magnitude_max)` (~L1554): UNIFORM theta grid over [-pi,pi] (NO
    cusp-alignment per Simplifier), rho rows over [rho_inner_floor=1.0,
    rho_outer]; keep tiles where admits_exterior True. Row-major i(rho inner
    -first)/j(theta) -> tiles[0] = global-min rho_inner (innermost, hardest).
  * exterior region loop rewire (~L3222): parity==1 builds
    exterior_admission=_interior_admission(band,1,reach_scalar,config) + tiles
    via _farfield_exterior_tiles; region_exclusion_rho = MIN admitted per-column
    rho_inner (fed to _farfield_region_window so w_floor/ppGO stays
    conservative). report keeps scalar 'exclusion_rho' (backward compat) + adds
    'region_exclusion_rho'. If parity==1 and zero columns admit -> LOUD
    zero_admission report (window None, admitted_tiles 0), no crash. Saddle:
    region_exclusion_rho=exclusion_rho, tiles=_farfield_tiles(exclusion_rho,...).
  * (4th, NOT in WP "Where" - correctness completion) `_subdivide_farfield_tile`
    (~L2859): the eps-gate corrective subdivision re-admitted EXTERIOR children
    via scalar `child_rho-child_half_rho >= exclusion_rho`. For parity==1
    exclusion_rho IS the cusp-spike (1+reach_max+eta_max-coord_radius_min ~5.94
    @ gamma0.85) - the very quantity WP1 replaces - so it would drop ALL children
    of a legit rho_inner~1.4 tile, defeating subdivision EXACTLY in the high-gamma
    exterior WP1 restores. Fix: added OPTIONAL trailing params
    exterior_admission=None, source_magnitude_max=None; exterior branch now
    if/elif/else: interior->admits; parity==1 (params given)->admits_exterior;
    else (saddle/None default)->scalar floor (BYTE-IDENTICAL). Both call sites
    (L3572 carrier-flip, L3595 gated) pass
    exterior_admission=(exterior_admission if parity==1 else None),
    source_magnitude_max=(y_outer_region if parity==1 else None). Conditional
    expr never evaluates exterior_admission for saddle (unbound-safe);
    y_outer_region bound for both parities.
- VERIFIED (sandbox ran, cheap geometry - no engine build): AST+import OK;
  _subdivide sig has both new params. gamma 0.80-0.90 band:
  _farfield_exterior_tiles admits 12 tiles (scalar path -> 0, the WP1 symptom);
  scalar cusp-spike exclusion_rho=5.942 vs admitted tile rho_inner=1.4; the 4
  subdivision children of tiles[0] -> directional admits 4, scalar floor admits 0
  (proves the defect + fix). Earlier partial already smoke-verified admits_exterior
  vs EXACT nearest_caustic_point 0 violations across 4 bands, reject cases (inside
  caustic / tube shell / out-of-box) all False, interior admits() unchanged.
- CONSUMER CHECK: pipeline_graph consumers_of lens_amplification_surrogate = 8;
  tiles stay (rho,theta_c) rectangles, no schema/serve/npz change -> consumers
  unaffected (per-column rho_inner only differs).
- UNVERIFIED (Test Dev/Inspector, need engine training): held-out eps-gate
  accuracy of per-column exterior tiles; live region report admitted_tiles>0 on a
  real gamma0.80-0.90 build; subdivision exterior-child recovery end-to-end on a
  real gated tile; production per-column rho_inner + window w_floor boundaries;
  full suite green. Saddle byte-identity to HEAD under a real saddle build.

## Findings pass (Build 8h-b3-fin): INS-2-002 FIXED; INS-2-001 -> Test Dev
- INS-2-002 (surrogate_training.py _heldout_eps ~L2187): FIXED. The LOO held-out
  reference was built via farfield_envelope_from_partition(partition) with the
  DEFAULT FARFIELD_KERNEL_SUM for EVERY far-field chart, ignoring
  chart.envelope_definition -> a diffractive-bottom or kernel-sum-minus-ghost
  chart would be probed against the wrong-F reference (latent under/over w-node
  provisioning). Fix: pass chart.envelope_definition to the helper (covers all 3
  S1-2 tags: farfield_full_kernel_sum / farfield_diffractive_bare_total /
  farfield_kernel_sum_minus_ghost; INTERIOR_SACR_C still routes to the
  partition.envelope branch via the unchanged is_farfield_label guard). Wrapped
  the call in try/except geometry.GhostDomainError -> `continue` so a ghost-gate
  refusal (minus-ghost only, w_min*Im tau_c>=2 off principal axes) DROPS that
  held-out point from the LOO accumulation, mirroring the training-time gate
  exactly (no propagate, no substitute). No new metric. Docstring updated.
  VERIFIED: py_compile OK; import no-circular; geometry.GhostDomainError exists
  and IS-A LensDomainError (already in _REFUSAL_ERRORS, but the explicit catch is
  intentional per the finding). Confined to _heldout_eps; training-time labeling
  (S1-2) and serve dispatch untouched.
- INS-2-001 (cogwheel/tests/test_lensing_surrogate.py): NOT done by me - handed
  to TEST DEV. It is genuine certification re-authoring, NOT a mechanical kwarg
  rename, and it blesses the caustic-fixed FarFieldChart migration that lives on
  the coder side of this build -> author-certifier separation forbids me touching
  it (a Coder porting the accuracy certs of its own migrated code = grading own
  homework; risk = green-but-wrong cert masking a defect). SCOPE for Test Dev:
  * _train (~L426): from_engine(gamma_range, y1_range, y2_range, n_y1, n_y2) ->
    caustic-fixed from_engine(gamma_range, rho_range, theta_c_range, n_rho,
    n_theta). Boxes POS_BOX/SAD_BOX are Cartesian (gamma,y1,y2) tuples - need
    re-derivation into (gamma, rho_range, theta_c_range) covering the same
    physical EXTERIOR region so downstream served (y1,y2) probes still land in.
  * _refusal_surrogate (~L448): same from_engine port; preserve the gamma=1
    parity-boundary refusal column + valid box-centre reasoning in caustic-fixed
    coords.
  * _multichart_fixture (~L2042): FarFieldChart.from_values(y1_grid=, y2_grid=)
    -> from_values(rho_grid=, theta_c_grid=) for pos_ff & sad_ff; refused_points
    ([[1.35,0.25,0.15]] gamma,y1,y2) stays physical; MC_QUERIES far-field
    expected-index rows must be re-checked against the new rho/theta_c coverage.
  * 4 affected cert classes now dark: SerializationMultiChart round-trip,
    RefusalPreservation, LnlikeAccuracy, EnvelopeReconstruction. Alternative the
    Inspector offered: migrate these 4 classes into test_lensing_exterior_windows
    .py. REFERENCE caustic-fixed API patterns already in that new suite: L414-420
    (from_values rho_grid/theta_c_grid) and L488-493 (from_engine rho_range/
    theta_c_range/n_rho/n_theta/w_nodes_per_decade/definition).
  * Serve still takes physical (gamma,y1,y2); only the TRAINING axes migrated -
    so a ported fixture must map its intended physical serve points through
    surrogate._to_caustic_fixed to choose covering rho/theta_c ranges.
- INS-2-001 RE-ISSUED (3rd pass, now naming 3 suites: test_lensing_surrogate.py
  21f+9e, test_lensing_ppgo_bandsplit.py L881 from_engine(y1_range=(1.2,1.4)),
  test_lensing_surrogate_census.py L300 _pos_farfield_dense from_engine(y1_range=
  (2.05,2.35),y2_range=...)). DECISION UNCHANGED: Test Dev owns it. NEW HARD
  EVIDENCE it is NOT a mechanical kwarg rename (computed via _caustic_reach/
  _to_caustic_fixed, reach(gamma) values banked):
    reach: 0.30->0.7171, 0.35->0.8682, 0.50->1.4142, 1.10->2.0793, 1.30->1.7144,
           1.50->1.8974, 2.05->2.3477, 2.20->2.4597, 2.35->2.5679.
  POS_BOX ((0.30,0.50),(1.95,2.30),(-0.15,0.15)) physical corners map to
  rho=|y|/reach that VARIES WITH GAMMA: at g=0.30 rho in [2.73,3.21]; at g=0.50
  rho in [1.38,1.63]; CROWN_LENS(g=0.35,y=(2.25,0)) -> rho=2.59. So the SAME
  physical Cartesian box is NOT any single caustic-fixed rectangle -- a
  fixed-rho box maps to a physical wedge that SCALES with reach(gamma). The
  migration deliberately changed the training-region SHAPE. Consequences that
  make this author-certifier-forbidden for the Coder:
    (a) Test Dev must CHOOSE new (rho_range,theta_c_range) per box; there is no
        deterministic coordinate-only translation of a Cartesian rectangle.
    (b) eps-budget tolerances are COVERAGE-CALIBRATED to the old box: POS_RECON_
        TOL=0.20 (measured ~8.4e-2), SAD_RECON_TOL=0.05 (~1.7e-2), LNLIKE_BUDGET
        _TOL=0.5, LNLIKE_ERROR_AMP. A different-shaped box gives different
        measured eps -> tolerances must be RE-MEASURED, not carried over. That
        re-measure/re-calibrate of the accuracy bar for the serve path S1-1
        migrated is precisely grading-own-homework -> Coder must not.
    (c) probe roles (CROWN 'deep' vs 'box-edge = coarsest-fit high-gamma/low-y1
        corner') are fixture geometry -> redefining them in caustic-fixed coords
        is fixture design, not renaming.
  RECIPE for Test Dev (native caustic-fixed, cover the documented physical
  probes at their gamma via _to_caustic_fixed, keep gamma/w ranges + node counts
  identical, keep serve probes physical, RE-MEASURE eps tolerances). Candidate
  POS box: rho_range~(1.35,3.25) theta_c_range~(-0.10,0.10) g(0.30,0.50) covers
  CROWN(2.59)+corners; SAD physical rho small (0.11..0.31 -> INTERIOR not
  far-field: SAD_BOX at g in [1.1,1.5] has rho<1, so the saddle 'far-field' box
  may itself need re-siting to a genuine exterior rho>1 wedge -- flag to Test Dev
  + Professor). OWED, do NOT ship these 3 suites red without Test Dev port.

## WP S1-3 (Build 8h-b3-fin): fixed [w_floor,w_trust] exterior window replaces
## mass strata + per-window LOO w-node reprovision (CONTINUATION - completed)
- Arrived with the previous coder's S1-3 work ALREADY in tree and coherent; my
  job was verify+complete+checkpoint. NO code edits needed - all pieces present,
  compiled, imported, and the pure-arithmetic containment check smoke-passed.
- surrogate_training.py ONLY. Pieces (all present, verified read-only):
  * _farfield_region_w_floor (~L1095): region w_floor = MAX over exterior-inner
    probes of channels.farfield_w_floor (S1-2 physics threshold (RHO_END/2)/
    min|tau_a-tau_b|); fallback = w(f_lo,m_lo) if no finite floor. report loud.
  * _farfield_region_window (~L1160): fixed [w_floor, w_trust]. top = _upper_w_cap
    (w(f_hi,m_hi), parity, rho_outer*reach) then _apply_ppgo_trim(boundary,ceiling)
    -> action drop/cap/keep/empty. 'drop'=whole band ppGO-served (window None);
    'empty'=degenerate w_floor>=w_trust (window None). SAME ppGO trim the strata
    path used (band-split serving live above w_trust).
  * _farfield_window_contains_draws (~L1230): 1e-12 RANGE CHECK. Sweeps 8
    log-masses; each draw's [w(f_lo,m),w(f_hi,m)] clipped to window; asserts every
    non-empty segment subset of [w_floor,w_trust]. Contained BY CONSTRUCTION of
    clip (max_viol=0) -> violation flags window/clip bug; n_overlap=0 = coverage
    note not violation. NO strata bookkeeping re-entered.
  * _reprovision_w_nodes (~L2246): draws held-out set ONCE (rng), descends n_w
    from config.w_nodes_per_decade down, retrains probe tile via
    _build_farfield_chart(w_nodes_per_decade=n_w) on the WINDOW, recomputes SAME
    LOO _heldout_eps (F-normalized, no new metric) vs config.farfield_eps_max=1e-3.
    Returns MINIMAL N_rec still clearing (stops at first n_w where eps>bar;
    eps(N_rec-1)>bar confirms minimality). Statuses: ok/engine_refused/
    bar_not_cleared/floor_reached -> non-decision returns FULL density (never
    guessed). Probes the INNERMOST tile (tiles[0], largest w_floor, hardest fit).
  * _build_farfield_chart gained w_nodes_per_decade: int|None kwarg (None ->
    config default); threads to from_engine(w_nodes_per_decade=...). n_rho/n_theta_c
    ALWAYS config -> spatial density HELD by construction (array_equal trivially).
  * exterior loop in _train_band_charts (~L3028): mass-strata partitioning of the
    EXTERIOR RETIRED -> ONE fixed region window; tiles still admitted by
    _farfield_tiles (geometry, not mass). Records window+N_rec+containment+
    reprovision in chart_{label}_farfield_region report. Each admitted exterior
    tile carries w_nodes_per_decade=N_rec; build loop (~L3294) + subdivision
    (~L2796) consume tile.get('w_nodes_per_decade') (interior tiles lack key ->
    fall back to config). _mass_strata STILL called for INTERIOR w-ranges +
    beyond_w_cap (plan-pinned: strata remain for interior). dropped_strata now
    collects INTERIOR ppGO drops only.
- VERIFIED (read-only + cheap arithmetic): py_compile OK; import no-circular; all
  referenced helpers exist (_upper_w_cap, _apply_ppgo_trim, _stratum_ppgo_boundary/
  ceiling, _scalar_caustic_reach, _from_caustic_fixed, _lens_prior._source_scale,
  dimensionless_frequency from lensing.waveform); TrainingConfig has n_heldout=10,
  farfield_eps_max=1e-3, w_nodes_per_decade=4, n_rho/n_theta_c/n_gamma=4; sig match
  _farfield_heldout_samples(band,center,half,config,rng) & _heldout_eps(chart,
  samples,prov); from_engine accepts w_nodes_per_decade; rng is a _train_band_charts
  param. _farfield_window_contains_draws smoke: wide window contained True viol 0.0
  n_overlap 8; narrow contained-by-clip True viol 0.0 n_overlap 1.
- NOTE on [0.5e-3,1e-3]: routine gates only eps<=bar (clear) + eps(N_rec-1)>bar
  (minimal). Did NOT add a >=0.5e-3 lower gate - minimality is defined by N_rec-1
  failing, not by eps floor; a discontinuous low jump at N_rec is still a valid
  minimal count. The [0.5e-3,1e-3] is the EXPECTED smooth-descent band the Test
  Dev checks on a synthetic tile, not a hard routine gate.
- UNVERIFIED (Test Dev/Inspector - all need engine training, out of Coder scope):
  live _reprovision_w_nodes descent on a synthetic tile hitting eps in [0.5e-3,
  1e-3] at N_rec and >1e-3 at N_rec-1; N_rec actually < config.w_nodes_per_decade
  on a real smoothed window (reduction claim); rho/theta_c array_equal asserted
  over a real before/after build; full suite green; tube path byte-identity to
  HEAD under a real interior build (tube untouched by S1-3 but other WPs in tree
  touched interior); production window boundaries + drop/empty/cap fractions per
  region against real ppGO map.


## Build 8h-b4 findings-fix (INS-4-001 done; INS-2-001 -> Test Dev)
- INS-4-001 (surrogate_training.py _train_band_charts): saddle (parity!=1)
  exterior tile bounds still used OLD multiplicative/division convention
  (exclusion_rho = physical_exclusion_radius/coordinate_radius_min;
  rho_outer_region = y_outer_region/coordinate_radius_min) while surrogate.py
  serve map went ADDITIVE in WP2(a). FIXED: collapsed both ternaries to the
  single additive form `1.0 + <phys|y_outer> - coordinate_radius_min` (parity
  difference is fully encapsulated in coordinate_radius_min from
  _coordinate_radius_bounds: per-angle min r_caustic for parity==1, band-min
  scalar _caustic_reach for saddle). parity==1 BYTE-IDENTICAL (same expression
  evaluated). Added explanatory comments.
- VERIFIED (sandbox ran): AST+import OK; saddle coordinate_radius_min ==
  min scalar _caustic_reach over band edges/mid (True); additive tile bound is
  exact mutual inverse of _from_caustic_fixed at band-min gamma (y_back==3.0);
  rho=1 at |y|=coordinate_radius_min; drho/d|y|=1 by linearity.
- FLAG to Inspector: saddle parity!=1 path in _train_band_charts is exercised
  by _sad_surrogate_ship()=_train(SAD_BOX,...) -> test_lensing_surrogate.py
  :847 (DomainGate box eps), :1920/:2031/:2054 (lnlike/chart-select saddle).
  _farfield_tiles/_farfield_region_window DIRECT-call tests (exterior_windows,
  surrogate_training) build their own exclusion_rho/rho_outer -> unaffected by
  this caller-side change.
- INS-2-001 (four RED suites: test_lensing_surrogate/ppgo_bandsplit/
  surrogate_census/exterior_windows) NOT DONE BY CODER - it is Test Developer
  work: (a) requires AUTHORING/choosing new test-fixture coordinates ("you
  never write the tests"), (b) requires RUNNING the 4 suites to green ("do NOT
  run test suites"). Root cause (Inspector-diagnosed): stale test fixtures pick
  physical sources now outside the trained caustic-fixed domain, e.g.
  BetaEliminationTestCase.setUp self.eig=(0.40,2.15,0.05) -> served=False
  ("anchor beta=0 source out of domain"). RECIPE for Test Dev: pick a TRAINED
  caustic-fixed (rho<=~rho_outer, theta_c) node, map to physical via
  surrogate._from_caustic_fixed(gamma,rho,theta_c), use as the fixture source;
  preserve all assertions/tolerances (POS_RECON_TOL/SAD_RECON_TOL/
  E_INVARIANCE_TOL unchanged). If Test Dev finds the domain gate wrongly
  rejects a legitimately-covered source, escalate BACK to Coder (production
  bug, not fixture).

## WP2 (Build 8h-b4): saddle additive-scalar axis + gamma=1 box-centre guard
- surrogate.py ONLY. (a) parity-else (saddle) arm of _to_caustic_fixed (~L282)
  and _from_caustic_fixed (~L318): switched multiplicative reach-norm to ADDITIVE
  scalar-reach: rho = 1.0 + |y| - _caustic_reach(gamma); inverse |y| =
  _caustic_reach(gamma) + rho - 1.0. KEPT scalar _caustic_reach (NOT directional
  geometry.r_caustic - Prof: r_caustic raises LensDomainError for saddle rays
  missing the two disjoint deltoid lobes -> ill-posed on most exterior rays).
  Both docstrings updated to describe additive scalar-reach (drho/d|y|=1).
  Positive-parity (astroid directional) arm + interior multiplicative arm
  UNTOUCHED (only the `else:` branch bodies changed).
- (b) _box_region_labels (~L1326): wrapped box-centre _from_caustic_fixed +
  geometry_partition in try/except _REFUSAL_ERRORS -> return None,None; return
  annotation now tuple[int|None,int|None]; docstring notes the None case (box
  gamma_c==1.0 hits _caustic_reach parity wall). Fixes chart-construction crash
  when box centre gamma exactly 1.0.
- VERIFIED (sandbox ran): AST+import OK; saddle round-trip exact 1e-12 for
  |y|>=reach (exterior range); drho/d|y|=1.0; rho=1.0 at |y|=reach (continuity
  with astroid arm); box gamma=1 -> (None,None); box gamma=1.5 -> (2,-1).
  NOTE: additive rho goes negative for |y|<reach-1 (below exterior); that hits
  the PRE-EXISTING rho>=0 guard in _from_caustic_fixed - expected, saddle
  exterior charts live at large |y|.
- CONFIRMED present & UNMODIFIED: c28408b node-loop guard (~L1268-1284) already
  wraps _from_caustic_fixed in try/except _REFUSAL_ERRORS + refused.append.
- _REFUSAL_ERRORS = (LensDomainError, CancellationError,
  SchwingerCertificationError) at L110.
- UNVERIFIED (Test Dev/Inspector): full suite green; whether existing saddle
  round-trip tests assumed multiplicative rho=|y|/reach (would now fail - the
  additive form is the intended replacement, tests need updating); held-out
  eps-gate accuracy on additive-axis saddle tiles; interp coupling reduction
  claim on a real trained saddle band.

## WP S2-3 (frozen WP8 amended): whole-interior SACR-C tau_c-demodulated label
- CONTINUATION completion. channels.py/surrogate.py were ALREADY done by prior
  partial (INTERIOR_SACR_C='interior_sacr_c_envelope', KNOWN_INTERIOR_DEFINITIONS
  frozenset, _KNOWN_ENVELOPE_DEFINITIONS union, CarrierDiscontinuityError,
  _assert_carrier_continuity, from_engine interior path storing partition.envelope
  + stamping tag, npz meta persist/validate, serve returns tag). Interior charts
  are FarFieldChart distinguished ONLY by envelope_definition tag; reuse EXISTING
  reconstruct_from_envelope (NO new reconstruction algebra) - label choice, coord
  stays S2-1 caustic-fixed rho (NOT tube u=sqrt(eta)).
- MY completing edits (surrogate_training.py + likelihood.py):
  * surrogate_training.py main tile loop (~L2947) + _subdivide_farfield_tile
    (~L2482): the ESSENTIAL bug was interior tiles built with default
    FARFIELD_KERNEL_SUM. Fixed: definition = INTERIOR_SACR_C if interior else
    FARFIELD_KERNEL_SUM; kind='interior'/'farfield'; build_ff/build_child pass
    definition=; gate=_gate_chart(kind,...); interior bar=config.interior_eps_max.
  * Carrier-flip = reseat-via-SUBDIVISION (Prof R4 "reseat tau_c via assignment
    convention"): wrap _load_or_build in try/except CarrierDiscontinuityError ->
    main loop calls _subdivide_farfield_tile+continue (L2978); child records
    result:'carrier_flip' gap + continue (single level, no recursion, L2549).
  * likelihood.py: import KNOWN_INTERIOR_DEFINITIONS; interior falls to else ->
    reconstruct_from_envelope with geom.switch/geom.critical_delay (functionally
    already correct); made explicit + defensive assert definition is None or in
    KNOWN_INTERIOR_DEFINITIONS at caustic-region branch.
- VERIFIED (sandbox ran, cheap 4^3 grid): interior from_engine(definition=
  _INTERIOR_ENVELOPE_DEFINITION) BUILDS finite - FarFieldChart, tag
  'interior_sacr_c_envelope', real/imag_coeffs bounded ~3 (SACR-C boundedness:
  NO divergence unlike far-field label), 0 refused, all finite. Public save/load
  round-trip preserves tag+coeffs. _validate_farfield_definition accepts interior
  tag, rejects bogus (ValueError). likelihood imports KNOWN_INTERIOR_DEFINITIONS.
  All 4 files py_compile OK, imports no-circular.
- NO near-cusp carve-out, NO fallback, NO Pearcey (Prof R4 clean-swap honored).
- UNVERIFIED (Test Dev/Inspector): held-out eps-gate accuracy of interior SACR-C
  tiles at mid-gamma=0.40 vs old far-field-label eps~6e-2 failure (the WP's
  raison d'etre - only smoke-built at gamma~0.5, not eps-measured); live
  likelihood serve end-to-end through _amplification_coefficients on a stored
  interior npz; carrier-flip subdivision actually TRIGGERING on a real basin-flip
  tile (assertion path present, not exercised - flip didn't occur in smoke band);
  finer near-gamma=1 band refinement fractions; full suite green.


- WP1 ghost-kernel (chang_refsdal/geometry.py): additive-only ghost path
  (451 insertions, 0 deletions -> real-image path byte-identical). New:
  GhostDomainError(LensDomainError), GhostContribution NamedTuple,
  _ghost_candidates/_ghost_delay/_ghost_kernel/ghost_kernel,
  _branch_pinned_amplitude/_wrapped_angle. All bilinear (no conjugation);
  reuses _companion_roots/image_quartic_coefficients/_source_frame and
  _c1/_c2_polynomial only; calls NONE of delay/hessian/magnification/
  morse_index/_saddle_metric/saddle_coefficients/image_kernel.
- KEY GOTCHA: the ghost pair is only a genuine COMPLEX-conjugate u-pair
  when the source is OFF the principal axes. EXACTLY on a principal axis
  (diagonal rotated frame, a12=0) the "extra" pair collapses to the
  removable singularity u=a22 (imag part below root_tolerance -> read as
  real; generic reconstruction is 0/0). So exact-on-axis is REFUSED
  (GhostDomainError). Near-axis is fine: Im tau_c -> 0 continuously and
  |kernel| converges to a finite limit -- matches the brief's on-axis
  "pure oscillation, evaluates finitely" as a LIMIT. Plan pins only the
  generic reconstruction (no axial ghost path), so this is plan-faithful;
  documented in ghost_kernel Notes + flagged UNVERIFIED below.
- Branch pin: reference_amplitude = exp(-0.5j*pi) directly (merged-saddle
  Morse index 1). Only its PHASE (-pi/2) enters +/- sqrt selection, so no
  need to call magnification on a possibly near-critical real image
  (would be fragile exactly where the ghost matters). arg(sqrt|mu|*e^{-i
  pi/2}) = -pi/2 regardless of |mu|.
- UNVERIFIED (Test Dev's oracle job): independent-oracle magnitude+phase
  agreement; exact P1-anchor |C|/phase-vs-E_ff within few %; Morse-
  double-count / branch-cut correctness at the exact anchor source
  angles. Smoke tests only confirmed runs-finite + qualitative anchors
  (Im>0 off-axis, ->0 near axis, negligible at rho=4, inside-caustic
  refused).


## WP S1-1 (Build 8h-b3-fin): far-field tiler -> caustic-fixed (rho, theta_c)
- Migrated ONLY surrogate_training.py (chart/serve/save-load in surrogate.py
  + geometry.r_caustic already done by prior WP; left untouched, verified
  self-consistent: from_engine takes rho_range/theta_c_range/n_rho/n_theta,
  save/load axis_schema='caustic_fixed_rho_theta').
- _farfield_tiles/_farfield_interior_tiles now emit (rho,theta_c) boxes.
  Tile half is a (half_rho, half_theta) 2-tuple (was scalar). theta tiled
  over [-pi,pi] with edges pinned on +-pi (half_theta=pi/n, centers
  -pi+half_theta*(2k+1)) -> NO tile straddles the branch cut by construction.
- Exterior admission uses SCALAR reach: exclusion_rho = 1 + eta_max/reach_scalar
  where reach_scalar=_scalar_caustic_reach(gamma_mid). NOT geometry.r_caustic
  (directional). Near-cusp notch (scalar-rho<1) owned by Slice-2 interior.
- _build_farfield_chart calls from_engine(gamma_range,rho_range,theta_c_range,
  w_range,n_gamma,n_rho=config.n_rho,n_theta=config.n_theta_c,
  w_nodes_per_decade). n_points=n_gamma*n_rho*n_theta_c.
- DEVIATION: plan said rename n_y1/n_y2 -> n_rho/n_theta, but tube already
  owns n_theta; used n_theta_c to avoid silent axis-merge. Threaded through
  budget check + node_counts report.
- Import: aliased surrogate._caustic_reach as _scalar_caustic_reach (local
  5-arg _caustic_reach shadow). Also imported _from_caustic_fixed for
  held-out physical mapping.
- Serve-mirror round-trip verified 8.9e-16 (< 1e-12 pin). Smoke build of
  _build_farfield_chart: FarFieldChart 64 pts, 0 refused, grids match box.
- OWED to Test Dev: tests still call retired sigs -
  test_lensing_ppgo_bandsplit.py:882-883, test_lensing_surrogate.py:430/459,
  test_lensing_surrogate_census.py:302, test_lensing_surrogate_training.py:214/1078,
  test_lensing_farfield_envelope.py:1852 (n_y1/n_y2 or from_engine(y1_range,y2_range)).
- UNVERIFIED (sandbox: no full suite run): held-out eps-gate accuracy of
  migrated tiles; interior origin (rho=0) degeneracy handling (marked
  TRANSITIONAL, Slice-2 replaces); node_counts arithmetic under real strata.


## WP S1-2 (Build 8h-b3-fin): w-windowed 3-class far-field label + serve mirror (atomic)
- channels.py OWNS the tags (single source): FARFIELD_KERNEL_SUM=
  'farfield_full_kernel_sum' (legacy mid-band, real switch=1),
  FARFIELD_DIFFRACTIVE='farfield_diffractive_bare_total' (switch=0, subtract
  nothing, bounded F object), FARFIELD_KERNEL_SUM_MINUS_GHOST=
  'farfield_kernel_sum_minus_ghost' (mid-band minus decaying ghost G).
  KNOWN_FARFIELD_DEFINITIONS frozenset (3) + _FARFIELD_KERNEL_FAMILY (2, the
  switch=1 tags) + _FARFIELD_WINDOW_RADIANS=RHO_END/2. All in __all__.
- New channels helpers: farfield_w_floor(delays,real_mask)=(RHO_END/2)/
  _min_delay_separation (inf if <2 real); _farfield_switch(real_mask,n_w,
  definition) raises ValueError on unknown tag (kernel family real=1 else all
  0); farfield_ghost_term(w,source,matrix) calls geometry.ghost_kernel, GATE
  w_min*Im tau_c>=RHO_END/2 else raise geometry.GhostDomainError, returns
  kernel*exp(1j w tau_c); reconstruct_farfield(w,env,delays,saddle_kernels,
  real_mask,definition)=reconstruct_from_envelope with _farfield_switch +
  tau_c=0. farfield_envelope_from_partition gained definition=FARFIELD_KERNEL_
  SUM (default byte-identical); minus-ghost branch subtracts farfield_ghost_
  term(partition.w,partition.source,partition.matrix).
- surrogate.py: import FARFIELD_KERNEL_SUM,KNOWN_FARFIELD_DEFINITIONS from
  channels; _FARFIELD_ENVELOPE_DEFINITION=FARFIELD_KERNEL_SUM,
  _KNOWN_FARFIELD_DEFINITIONS=KNOWN_FARFIELD_DEFINITIONS (now 3-tag).
  _validate_farfield_definition already generic (frozenset membership) -> now
  accepts all 3; hard-refuses None/unknown at LOAD (_chart_from_npz L1648,
  legacy L1494) BEFORE numerics. _chart_to_npz persists chart.envelope_
  definition generically. NO other surrogate change.
- likelihood.py serve mirror (_surrogate_coefficients far-field branch):
  changed `if definition==_FARFIELD_ENVELOPE_DEFINITION` -> `if definition in
  KNOWN_FARFIELD_DEFINITIONS`. Diffractive+band_split -> return None (switch=0
  gauge has no ppGO telescoping). minus-ghost: rebuild source=[y1,y2],
  matrix=macro_matrix(gamma,beta,kappa) (EXACTLY as partition L672-673), try
  ghost=farfield_ghost_term(chart_w,...) except GhostDomainError: return None;
  envelope_dense[below_mask]+=ghost. All far-field tags now route through
  reconstruct_farfield(...,geom.real_mask,definition). elif band_split / else
  (tube) UNCHANGED. Removed now-unused surrogate _FARFIELD_ENVELOPE_DEFINITION
  import; added GhostDomainError,macro_matrix from geometry + reconstruct_
  farfield,farfield_ghost_term,FARFIELD_DIFFRACTIVE,FARFIELD_KERNEL_SUM_MINUS_
  GHOST,KNOWN_FARFIELD_DEFINITIONS from channels.
- ATOMICITY satisfied: tag frozenset (channels) + reconstruct dispatch
  (reconstruct_farfield + likelihood branch) shipped in ONE edit set; unknown
  tag hard-refused at load (_validate_farfield_definition) before any numerics.
  Mixed-tag legal (no coexistence validation added).
- CONSUMER CHECK: pipeline_graph consumers_of lens_amplification_surrogate ->
  runtime = LensedRelativeBinningLikelihood (edited) + LensedMarginalized
  ExtrinsicLikelihood (composes internal RB engine, calls _engine._
  amplification_coefficients L306 -> same edited _surrogate_coefficients). ONE
  dispatch site covers both.
- VERIFIED (sandbox ran, engine builds cheap): py_compile all 3 OK; imports
  resolve no-circular; KNOWN set=3; farfield_w_floor formula; switch dispatch
  (diffractive all-0, kernel family real=1); unknown/None tag ValueError. REAL
  gamma=0.5 off-axis exterior round-trips: kernel-sum F-norm err 0.0 (byte-
  identical to HEAD path); minus-ghost subtract-G/re-add-G telescopes to F err
  0.0, env_ks-env_g==G to 3e-21, |G| decays 1.2e-4->1.5e-18; diffractive env
  IS F (bounded 1.39, no divergence) reconstruct err 0.0; ghost gate refuses
  at w_min*Im tau_c=1.96<2.0 (just-below-threshold near-axis).
- UNVERIFIED (Test Dev/Inspector): full suite green; likelihood serve path
  end-to-end through _amplification_coefficients under a REAL trained
  minus-ghost chart (only tested channels-level round-trip, not a stored npz
  tile served via surrogate.serve->_surrogate_coefficients); held-out eps-gate
  F-normalized accuracy on minus-ghost/diffractive training tiles; band-split
  minus-ghost above w_trust bare-ppGO collapse in the live likelihood (tested
  the gauge identity at channels level only); production w_floor/w_trust
  window boundaries against real geometry per region.

## WP S2-2 (frozen WP7): per-lobe saddle interiors + winding admission + inter-lobe corridor
- surrogate_training.py ONLY (geometry.py needed NO edit: critical_point already
  serves both saddle lobe branches via +-sqrt + wedge LensDomainError; macro_matrix
  beta=0 -> diag(1-g,1+g) so shear/negative axis = x, lobes on x-axis at +-|centroid_x|,
  equidistance line = y-axis / x=0).
- New consts: _SADDLE_LOBE_CENTERS=(0.0, pi) (lens-plane lobe centres),
  _INTERLOBE_CORRIDOR_ETA_SCALE=1.0 (corridor_half = 1*eta_max).
- New helpers: _lobe_caustic_points(g,lens_center,n) one lobe's source caustic cloud
  (both branches over wedge |sin2t|<=1/|g|); _lobe_winding_loop ORDERED closed deltoid
  boundary (branch+1 fwd over thetas, branch-1 bwd over thetas[::-1]);
  _lobe_cusp_source_angles -> 3 cusp rays as lobe-local atan2 from centroid (via
  _find_cusps on both branch speed profiles w/ saddle windows + critical_point.source);
  _directional_lobe_boundary(points,centroid,n_bins=181)->(centers,r_deltoid)
  angular-binned MAX radius, nan-fill periodic interp; @dataclass(frozen,eq=False)
  _SaddleLobeAdmission (centroid/other_centroid/reach/eta_max/corridor_half/loops/
  caustic_cloud/boundary_theta/boundary_r + _r_deltoid + _probe_points(9-probe 3x3) +
  admits); _saddle_lobe_admissions(band,config)->2 admissions;
  _lobe_interior_tiles(admission,cusp_angles,n_per_side) rows rho_lobe[0,1] x
  _cusp_aligned_theta_tiles.
- ADMITS (all 9 probes): (a) winding +-1 about EVERY band loop
  (abs(_winding_number(loop-p))>=0.5), (b) nearest caustic dist>=eta_max (off-radial
  shell, S2-1 style), (c) corridor |p-centroid|+corridor_half<=|p-other_centroid|
  (nearer-lobe assign, excludes equidistance belt).
- KEY FIX (my scalar-reach draft was WRONG): rho_lobe MUST normalise by DIRECTIONAL
  r_deltoid(theta_local), NOT scalar reach. Lobes are elongated/sheared deltoids -- far
  cusp at dist=reach but near-cusp clearance<<reach, so scalar-reach tiles overshoot
  near-cusp dirs -> ZERO tiles ALL bands even where inradius 0.31>>eta 0.05. Directional
  r_deltoid (rho=1=boundary each dir) fixed it: tiles=114/82/61 for g~{1.05,1.15,1.30}.
  WP explicitly pins r_deltoid(theta) w/ 3 kinks -> plan-faithful, not a deviation.
- SERVE DEFERRED (documented, served=False + serve_note in interior_report): lobe tiles
  RECORDED but NOT packed into `admitted` (not built/served). surrogate._from_caustic_
  fixed is strictly ORIGIN-centred, NO lobe-centroid offset -> lobe-local chart can't be
  placed at true physical location without out-of-scope serve changes (surrogate.py/
  likelihood.py). Follow-on slice. This slice delivers admission+tiling geometry (the
  S2-2 verification target).
- Wiring: _train_band_charts interior block `if parity != 1:` (saddle: build lobe
  admissions, loop 2 lobes, accumulate lobe_records, interior_skip=
  'saddle_lobes_zero_admission' if 0 tiles) `elif not encloses:` `elif reach<=eta:`
  `else:` (astroid loop UNCHANGED). interior_report: admission='per_lobe_winding' saddle
  else 'directional_r_caustic'; +lobes/served/serve_note only parity!=1. ASTROID
  (parity==1) BYTE-IDENTICAL (conditional->'directional_r_caustic', parity!=1 block
  skipped). `admission` var stays None for saddle (no interior tile in `admitted` ->
  _subdivide_farfield_tile never dereferences it; exterior uses region).
- Zero-admission LEGIT for bands where lobe inradius<eta_max (small lobes larger gamma:
  (1.4,1.6) 0.067 borderline sub-tile 0 tiles; (1.9,2.1) 0.047<0.05; (2.9,3.1)
  0.026) -> shell fills lobe, ladder/eps gate covers (mirrors astroid tube_shell_fills).
- VERIFIED (sandbox, cheap geometry): AST+py_compile+import OK; centroids on x-axis
  opposite sides; winding admits lobe-own centroid(-1) refuses other(0)+origin(0);
  3 cusps/lobe; band(1.1,1.2) 46+36 tiles ALL bad_wind=0 bad_eta=0 bad_corr=0
  crossed_x0=0 cusp_edges_aligned=True; synthetic isolate proves corridor BITES
  (on-equidistance probe refused, deep-own-side admitted).
- OWED to Test Dev: NEW gates - per-lobe winding admission; directional r_deltoid vs
  scalar-reach near-cusp tileability; cusp-ray + equidistance no-straddle; real_mask by
  Morse signs across caustic (engine behavior, inherited, NOT exercised since
  served=False); saddle_lobes_zero_admission skip. No retired signatures (all additive;
  astroid untouched).
- UNVERIFIED (Test Dev/Inspector): full suite green; whether mean-of-caustic-points
  centroid is true deltoid star-centre for strongly-sheared lobes (r_deltoid assumes
  star-shaped about centroid -- holds tested bands, verify near g->1 merge and large g);
  production admission fractions; SERVE end-to-end (deferred, no lobe offset); real_mask
  =-2 negative-parity at the served label (not built this slice).

## WP S2-1 (frozen WP6): interior directional-radius admission + cusp-ray tiling
- surrogate_training.py ONLY. Replaced isotropic inscribed-disk interior
  admission (scalar interior_rho_admit = (inradius-eta_max)/reach) with
  DIRECTIONAL admission. New helpers (after _caustic_inradius): _cusp_source_angles,
  _cusp_aligned_theta_tiles, @dataclass(frozen,eq=False) _InteriorAdmission
  (reach/eta_max/theta_axis/rho_boundary/caustic_cloud + .admits), _interior_admission.
  New consts _INTERIOR_BOUNDARY_NODES=181, _INTERIOR_EDGE_SAMPLES=5.
- admits((rho_c,theta_c),(half_rho,half_theta)): probe OUTER rho edge at 5
  thetas across span; admit iff every probe (a) rho_outer < rho_boundary(theta)
  [band-MIN directional r_caustic/reach -> inside caustic for ALL band gammas,
  no band-edge waste] AND (b) nearest-caustic distance >= eta_max [Prof caveat
  ii: near a cusp the nearest caustic is OFF the radial ray, so use point-cloud
  min-distance, NOT the radial gap].
- rho_boundary built by CALLING geometry.r_caustic (the pinned directional
  radius, reused not re-added) at 181 thetas x band {g_lo,g_mid,g_hi}, per-angle
  MIN across the 3 gammas. Cloud = vstack(_caustic_points over same 3 gammas).
  KEY: reach=surrogate._caustic_reach(gamma_mid) is FIXED for the band, so band
  min of r_caustic gives boundary<1 even at cusp direction (0.49-band gave max
  0.858, narrow 0.49-0.51 gave 0.97) -> conservative, correct.
- cusp rays: _cusp_source_angles maps _find_cusps LENS-plane minima -> SOURCE
  atan2 of critical_point.source. Astroid gamma=0.5 -> 4 cusps at axes
  (-pi/2,0,pi/2,pi). _cusp_aligned_theta_tiles: edges = cusps U {-pi,pi},
  each sector split into n_per_side uniform sub-tiles -> NO tile straddles a
  cusp-ray kink OR the +-pi branch cut. Empty-cusp fallback = 1 uniform sector.
- _train_band_charts interior block: encloses gate PRESERVED (via
  _caustic_inradius winding number; saddle still skips 'caustic_not_origin_
  enclosing'). tube_shell skip now keys on reach_scalar<=eta_max (was
  inradius-eta_max<=0, which would've discarded the whole notch). grid_extent
  now min(reach_scalar, y_extent) (reaches full caustic, was min(inradius-
  eta_max, y_extent)). report fields dropped interior_admit_radius/
  interior_rho_admit; added admission/caustic_reach/n_cusp_rays/cusp_angles.
- _subdivide_farfield_tile: param interior_rho_admit -> interior_admission
  (_InteriorAdmission|None); interior child predicate now
  interior_admission.admits((child_rho,child_theta),(child_half...)) [same test
  as tiler, DRY], exterior child predicate unchanged (exclusion_rho). Call site
  passes interior_admission=admission.
- VERIFIED (smoke, sandbox ran): parse+import+py_compile OK; synthetic unit-
  circle admits() -> interior True / near-shell False / outside False; real
  astroid 4-cusps-at-axes, boundary anisotropic (max at cusp, min at diagonal),
  no tile straddles a cusp ray, diagonal-tile refused, NOTCH RECOVERED (cusp-dir
  tile at rho 0.672 between inradius 0.354 and dir-radius 0.97: S2-1 admits True,
  old isotropic False).
- OWED to Test Dev: retired _farfield_interior_tiles signature (old
  (rho_extent, rho_admit, n_per_side) -> new (rho_extent, n_per_side, *,
  admission, cusp_angles)) breaks test_lensing_ppgo_bandsplit.py:560,607,609
  (class ~L544, imports L88). Need new gates: directional admits vs r_caustic;
  cusp-ray no-straddle; nearest-distance eta exclusion; band-min conservatism;
  notch recovery vs old isotropic; encloses gate + saddle skip unchanged.
- UNVERIFIED (Test Dev/Inspector): full-suite green; production eta_max/band
  widths admission fractions; r_caustic cost at 181x3 per band per parity
  (offline, dwarfed by engine builds) not runtime-profiled; interpolation
  conservatism near sharp cusp kinks (steep dropoff can reject on-ray tiles
  right at the cusp -- legitimate, but Test Dev should confirm not over-strict).