# (compacted 2026-07-23: older entries dropped after the E2BIG argv incident — full history in git)

- WP4 (Build 8h-a-fin far-field edge subdivision): surrogate_training.py
  ONLY (surrogate.py/likelihood.py/_build_farfield_chart/_gate_chart/
  _heldout_eps/_farfield_tiles/_farfield_interior_tiles all UNTOUCHED — git
  confirms; likelihood.py/DATA_CONTRACTS mods in tree are prior WP1-3, not
  mine). New module-level helper `_subdivide_farfield_tile` inserted before
  _train_band_charts + a subdivision branch added to the FINAL
  `for tile in admitted:` gated arm. On _gate_chart('farfield')==gated:
  halve tile half h->h/2, form up to 4 children at (cx±h/2, cy±h/2) half
  h/2, iterate row-major (sx outer in {-1,+1}, sy inner) => ci 0..3
  deterministic. Re-admit each child through the PARENT's region predicate
  carried VERBATIM (tile['region'], never re-derived): exterior admit iff
  hypot(max(0,|ccx|-h/2),max(0,|ccy|-h/2))>=exclusion_radius (mirrors
  _farfield_tiles min-corner); interior admit iff
  hypot(|ccx|+h/2,|ccy|+h/2)<=interior_admit_radius (mirrors
  _farfield_interior_tiles max-corner). Disk-excluded child DROPPED silently
  (recorded only in summary result='disk_excluded', packed in neither charts
  nor chart_reports — correct geometry, not a failure). Admitted child
  retrains via _build_farfield_chart on the parent's INHERITED already-
  ppGO-trimmed tile['w_range'] (NO per-child _stratum_w_range/_apply_ppgo_trim
  — Simplifier), samples via _farfield_heldout_samples, eps via _heldout_eps,
  wrapped in _load_or_build (child tag {parent_tag}_c{ci}, resumable like
  parent), re-gated via _gate_chart vs same farfield_eps_max. PASS -> append
  charts + chart_reports (with subdivided_from) like a normal admitted tile;
  STILL-FAIL -> chart_reports gated=True + gate_reason, NOT packed. SINGLE
  level, NO recursion (failing child recorded & left; depth==1). nan_eps
  parent whose child re-nans stays gated (not special-cased). Parent gated
  chart_report gets 'subdivided':True and 'subdivision' summary
  (parent_tag,region,per-child {ci,center,half,admission,eps,bar,gate_reason,
  result}) for ladder census. Parent report appended BEFORE children (list
  order parent-then-children) then summary mutated onto the same dict object.
- Verified: ast.parse OK, import OK, helper callable. Stubbed-engine wiring
  smoke (4 helpers monkeypatched, physics-free): EXTERIOR all-admitted case
  2 packed/2 recorded_gated in row-major ci order, all children carry
  subdivided_from + gate_reason='eps_above_bar'; INTERIOR near-boundary case
  1 admitted + 3 disk_excluded, excluded packed nowhere (reports len ==
  4-excluded). Child summary carries all census fields. NOTE: real
  gate_reason strings are 'eps_above_bar'/'nan_eps' (brief said 'heldout_eps'
  — I pass _gate_chart's actual return verbatim, no hardcoded string).
- UNVERIFIED (WP4): full training-loop run at fixture/production scale (the
  engine-backed _build_farfield_chart path) — smoke used stubbed engine;
  Test Dev/downstream runs the real subdivision on a genuinely gated tile.
  Whether a halved child actually clears farfield_eps_max in practice is a
  physics/measurement claim I did NOT run (structure + gating-wiring +
  admission geometry verified). OWED to Test Dev: author gates pinning
  (a) child row-major determinism + tags, (b) parent-region-carried admission
  (exterior min-corner / interior max-corner) with disk-excluded drop,
  (c) inherited w_range (no per-child ppGO recompute), (d) pass->packed /
  fail->recorded-not-packed / depth==1 no-recursion, (e) subdivision summary
  census fields, (f) surrogate.py + tube byte-identity intact.

- (prior arm) d to the runtime CALIBRATION CERTIFICATE (_calibration_certified:
    each reduced stationary phase must match a distinct geometric cusp-
    cluster scaled delay w*(tau-tau_c) within _CALIBRATION_TOL). At the real
    cusp the reduced stationary phase vs geometric cluster delay differ by a
    factor CONSTANT IN w but VARYING WITH OFFSET (4.26 at eps=0.25, 6.95 at
    eps=0.12, 12.5 at eps=0.05). => the w^{1/2}/w^{3/4} SCALINGS ARE CORRECT
    (mismatch w-independent) but the offset->normal-form-coefficient map
    (b1,b2) uses BARE soft/hard-axis projections and is missing O(1)
    curvature factors (b1=-Delta·e_s times mixed 2nd-derivs of the Fermat
    potential, not the bare offset); also #stationary pts (1) vs geometric
    cluster (3 merging images) => controls land in the wrong Pearcey region.
    The certificate is DOING ITS JOB: refuses the mis-calibrated mapping,
    NEVER serves a wrong number. Broad random sweep (600 configs, seed 0):
    0 exceptions, 4 served, 596 refused — arm is refusal-conservative and
    NOT dead. I did NOT hand-tune the curvature constants (would make the
    certificate pass on an unvalidated amplitude = grading own homework
    against a fitted oracle). OWED to driver/Test Dev: pin & brute-force
    validate the b1,b2 offset->control curvature calibration; the served
    AMPLITUDE is UNVERIFIED (structure/refusal-gating verified; primitive
    verified). Caveat documented in cusp_amplification docstring Notes.
  * Gates (cusp_amplification, in order): input guards (w>0, shape (2,),
    finite, envelope_bar>0) -> geometry LensDomainError -> _cusp_vertex
    (golden-section on caustic-speed min) None -> _soft_normal_form None ->
    F016 radius gate R<R_min=(c_P/bar)^{2/3} -> pearcey None -> per-image
    _leading_geometric None -> _calibration_certified. All return None; no
    raise. Full smoke suite GREEN.

- WP1 (Build 8e corner-scoping census): EXTENDED
  scripts/census_homogenization_corners.py ONLY (engine untouched;
  operator.L_MAX stays 48, select_branch/geometry.py NOT modified —
  git status confirms census script is my sole change). Pure-geometry,
  engine-free, deterministic (same seed -> byte-identical JSON; verified
  n=300 seed=1 twice). Classifies refused high-w corner nodes into 4
  buckets under report['corner_scoping']:
  (a) a_geometric_now = high & geometric-served already (smoke: 1091/3040,
      frac 0.359 +Wilson95).
  (b) b_geometric_under_relaxed_l_max = MEASURED-ONLY: resolved refused
      POSITIVE-parity nodes (L=w*|y'|<=48, blocked only by conservative
      cancellation gate) that L_MAX_RELAXED=60 PLUS image-census guard
      (_image_census_matches: mags.size in {2,4} & merging pair is one
      +min & one -saddle) would move onto geometric. Saddle refusals have
      NO L gate -> never rescued. Reports frac 0.194 + Wilson +
      production_l_max=48 note. select_branch/L_MAX UNCHANGED (raising
      L_MAX would REDUCE geometric since gate is L>L_MAX; (b) is a
      what-if count, not an action).
  (c)/(d) cd_uniform_or_hardcore = refusal & ~relaxed: NO hardcoded
      xi*/R* threshold. Emits full fold (w*Delta_tau, +xi=(0.75*wDtau)^(2/3))
      and cusp (R) argument CDFs as fraction_vs_threshold TABLES over fixed
      grids (FOLD_WDTAU_THRESHOLD_GRID 33pt linspace 0..8;
      CUSP_R_THRESHOLD_GRID 32pt [0]+logspace(-1,2,31)); each row
      fraction_resolvable_ge + Wilson95 + fraction_hardcore_lt; split read
      off at arms' certified thresholds POST-BUILD.
- Geometry-only args per refused node: ONE geometry.find_images solve per
  config (when any w>60); delta_min inline-mirrors
  operator._real_delay_min_separation; Delta_tau=delta_min/2 EXACT (merging
  min(n=0)/saddle(n=1) pair IS the delta_min pair — no re-solve). Topology
  via _classify_fold_or_cusp(delays): 'cusp' when a 3+ cluster of
  consecutive delay gaps within CUSP_CLUSTER_DELAY_RATIO=3*gmin, else
  'fold', else 'degenerate' (<2 real images -> hard-core (d)). Cusp R:
  x=w^0.5*|delta_parallel|, y=w^0.75*|delta_perp| (2/3 law), offsets from
  geometry.nearest_caustic_point(gamma,beta,source,kappa) projected on
  soft_axis(parallel)/hard_axis(perp); guarded try/except LensDomainError
  -> degenerate.
- Memory-bounded: streams cd fold/cusp args into fixed threshold-grid
  histograms (counts_ge += (arg[:,None]>=grid[None,:]).sum(0)) — no storing
  2e5x128 per-node samples; the CDF/table IS the histogram. Wilson95 reuses
  existing wilson_interval helper. partition_check asserts
  n_fold_cd+n_cusp_cd+n_degenerate_cd == cd_node_total (smoke: 1358==1358,
  consistent True). All existing report keys preserved (additive section).
- BUG hit & fixed (recorded lesson): replace_symbol_body on `run` DELETED
  the def+docstring (tool body MUST include the signature line) -> col-0
  IndentationError; repaired via replace_content re-adding def/docstring +
  reindent, then all later replace_symbol_body calls INCLUDED signatures.
  ast.parse + importlib(exec w/ sys.modules registration for KW_ONLY
  dataclass resolution) both GREEN; e2e smoke run GREEN.
- UNVERIFIED (WP1 8e): full N>=1e5/2e5 production run pass/fail + wall-clock
  (smoke n=300 only; each high-w config does one find_images solve +
  possibly one nearest_caustic_point — offline run, owner/downstream
  executes). The reported fractions (a 0.359 / b 0.194 / cd 0.447) and
  topology counts are FIXTURE-scale (seed=1 n=300); owner-facing numbers
  need the full draw. Test authorship (census-extension gates) is Test
  Dev's — declined (code+blessing must not share an author).

- WP3 (Build 8d homogenization-corner census): NEW standalone
  scripts/census_homogenization_corners.py — geometry-only Monte-Carlo
  REPORTING deliverable (NOT a gate), modifies NO engine code (git status:
  only new file; operator.py mod is the pre-existing WP1 change). Draws from
  _LensPriorBox(CombinedPrior) = [FixedLensGeometryPrior, UniformLensMassPrior,
  UniformReducedShearPrior, UniformSourcePositionPrior] via
  generate_random_samples (box READ from prior classes, not hardcoded), mirrors
  surrogate_census idiom but self-contained (all logic in-script per WP "new
  standalone script under scripts/"). Reports: (a) gamma'==0 fraction (==0 in
  smoke, measure-zero) + gamma'<0.01 bonus (tracks analytic 0.01/1.6); (b)
  unresolved-high-w NAMED-refusal corner fraction with Wilson95 interval.
  classify_config mirrors PRODUCTION dispatch per parity: positive
  select_branch gate (resolved w*delta_min>=RHO_END AND L=w*|y'|>L_MAX);
  saddle _saddle_grid gate (resolved AND w>W_CEILING, NO L condition);
  w<=60 -> served-by-Schwinger both parities; w>60 & not-geom-eligible ->
  refusal. delta_min (real-image delay sep, one quartic solve) computed ONCE
  per config and ONLY when any w>60 (perf). macro_matrix/maps/find_images
  LensDomainError (incl. F012 census) -> engine_refused bucket (precedes
  dispatch). gamma_prime read from engine mass-sheet map, cross-checked vs
  analytic gamma/(1-kappa) (smoke: max diff 0.0). Constants imported not
  hardcoded (W_CEILING=_schwinger.W_CEILING_SCHWINGER, RHO_END/L_MAX from
  operator). Smoke n=300 seed=1: gamma'==0=0, gamma'<0.01=1/300, corner
  0.23 [0.186,0.281], engine_refused=0, max_w=441.6<500, xcheck=0.0; JSON
  well-formed (10 top keys). Parse+import+e2e all GREEN.
- UNVERIFIED (WP3): full N>=1e5 production run pass/fail + wall-clock (smoke
  was n=300 only; 2e5 does ~most-config quartic solves for delta_min — offline
  run, downstream/owner executes). Refusal-corner fraction is FIXTURE-scale
  seed=1 n=300 (0.23); the owner-facing number needs the full N>=1e5 draw.

- WP1 (Build 8d homogenize positive parity onto Schwinger): rerouted
  operator.py positive-parity wave branch. RENAMED
  _positive_parity_grid_with_fallback -> _positive_parity_grid (internal;
  only F_op/F_op_grid callers, no test refs to the name). New body is a
  DIRECT dispatch on gamma_prime=_mass_sheet_map(y,gamma,kappa)[2]
  (=gamma/(1-kappa)): gamma'>0 -> per-node _schwinger.f_schwinger with the
  IDENTICAL reduce/rotate/reconstruct copied verbatim from the old 7a
  fallback AND _saddle_grid (z_eig rotate, mass_sheet_phase, /lam); gamma'==0
  (exact, via `not gamma_prime>0.0`) -> legacy _grid_certified (sole legacy
  production exit). Deleted the try/except-CancellationError-then-Schwinger
  fallback structure entirely. Added w_array.ndim!=1 guard (matches
  _saddle_grid). Both F_op(scalar, 1-elem grid) and F_op_grid delegate here
  => single-intercept reroute for both.
- STEP2 refusal symmetry: gamma'>0 above-ceiling now raises
  SchwingerCertificationError (named), NOT the old CancellationError — this
  is the intended homogenization (positive parity == saddle arm behavior).
  No legacy fallback catch. gamma'==0 keeps CancellationError.
- STEP3 oracle alias: module-level `legacy_operator_oracle = _grid_certified`
  inserted right after _grid_certified def, with 'test-only oracle; NOT a
  production path (see Build 8d)' comment. No wrapper/logic.
- STEP4 verified intact (I did NOT touch _grid_certified/_fused_contraction):
  _SERIES_TOLERANCE=2e-12 module global, _fused_contraction.py_func reachable,
  half_sum still a param. Smoke: gamma'>0 F_op order_used=0 & matches direct
  Schwinger recon <1e-14; gamma'==0 order_used=9 (legacy); grid==scalar;
  w=70 gamma'>0 -> SchwingerCertificationError.
- Docstrings updated in-file (F_op, F_op_grid, _positive_parity_grid, module
  MACRO-SADDLE DISPATCH para): removed the now-false 'positive parity ...
  BYTE-IDENTICAL operator/1F1 branch' claim; documented gamma'=0 as sole
  legacy exit + Schwinger for gamma'>0 on both parities.
- DID NOT TOUCH: _saddle_grid (saddle arm), channels.py, select_branch,
  geometry branch dispatch, refusal vocabulary (no new exception classes).
- OWED to Test Dev (Build 8d): tests that pin positive-parity gamma'>0 to
  the LEGACY operator path WILL now go RED and need re-baselining with
  contract-flip witnesses (7b precedent) — NOT my defect, intended flip:
  (1) test_lensing_schwinger.py::PositiveParityBitFreezeTestCase — asserts
  F_op/F_op_grid return frozen literals AND diagnostics.order_used>0 for
  moderate-shear (gamma'>0) configs; both now false (Schwinger value within
  1e-10 of the frozen literal, order_used==0). Re-baseline literals to the
  Schwinger values + flip the order_used>0 assertion to ==0.
  (2) crown byte-identity pin (test_lensing_surrogate.py) and ratio-layer
  cache-determinism pin against the CURRENT positive-parity evaluator WILL
  flip (brief anticipates this); physics tolerances (RB-vs-brute, oracle
  1e-10) must hold. (3) The NEW Build-8d overlap-domain harness
  (Schwinger-vs-legacy_operator_oracle at 1e-10 on the certified overlap,
  refusal-decision identity both directions) + a dispatch mutation test are
  Test Dev deliverables — legacy_operator_oracle is the import handle.
  (4) UNVERIFIED: full lensing suite pass/fail at fixture scale (I only
  smoke-checked import+wiring+one config per branch; downstream runs it).

- WP1 (Build 8c multi-chart surrogate): rewrote
  cogwheel/lensing/surrogate.py into a flat multi-chart emulator — two
  FROZEN dataclasses (TubeChart in (gamma,u=sqrt(eta),theta,log w);
  FarFieldChart in (gamma,y1_eig,y2_eig,log w)), NO hierarchy;
  select_chart() is a plain guard-stack fn keyed on certified physical
  eta+image_count (theta only for cusp-window exclusion, F017). Single-npz
  save/load (chart{i}_* named arrays + one JSON provenance scalar, NO
  manifest); two load paths (package-data default via importlib.resources +
  explicit path override; data_registry fallback left as a TODO). 8a
  backward compat: legacy npz (no `n_charts` key) -> one FarFieldChart via
  _load_legacy_single_box; envelope()/in_domain()/from_engine()/grid attrs
  preserved. Smoke-verified: multi-chart round-trip coeff-exact, serve
  tube+farfield, cusp exclusion + gamma-guard fall-through, legacy load.
- BUG I introduced AND fixed same session: _normalize_refused(None) gave
  array(nan) shape () -> ValueError; None must short-circuit to
  np.empty((0,3)). (np.asarray(None,float) is NOT size 0.)
- channels.py: ONE additive field `caustic_theta: float` on
  ChangRefsdalGeometryPartition, populated `caustic_theta=float(caustic.theta)`
  in geometry_partition (the ONLY constructor; caustic already bound via
  geometry.nearest_caustic_point). Verified populated (2.74 rad sample).
  reconstruct_from_envelope consumed fields untouched.
- likelihood.py: _surrogate_coefficients now does may_serve early-out ->
  geometry_partition -> surrogate.serve(eta=caustic_distance,
  theta=caustic_theta, image_count=int(real_mask.sum())). Removed dead
  _SURROGATE_CAUSTIC_FLOOR const, _surrogate_region_image_count method,
  _surrogate_region_nimg attr (init+getstate pop+setstate+docstring). Kept
  kappa!=0->None guard (INS-8a-001). Intercept still guarded by
  `if amplification_surrogate is not None` => default None byte-identical.
  pyright None-narrowing warnings on serve/may_serve are pre-existing-style
  noise (HEAD had same on in_domain/envelope), not defects.
- WP-CS (Build 8c-cont census tool): NEW cogwheel/lensing/surrogate_census.py
  (importable core, pure compute) + scripts/census_lens_surrogate.py (thin
  arg-parse+run+json.dump CLI, no embedded logic). NO edits to
  surrogate.py/channels.py/likelihood.py/prior.py (all READ-ONLY). Verified
  read-only: parse OK, import OK, guard-predicate signatures match my call
  sites (positional _tube_serves/_farfield_serves, kw select_chart/serve),
  draw_samples yields gamma/m_lens_msun/y1/y2 cols. NO shipped surrogate .npz
  in tree yet => full end-to-end run UNVERIFIED (Test Dev builds fixture).
- Categorization technique: classify_fallthrough toggles ONE guard off on a
  FROZEN chart via dataclasses.replace (cusp_windows=() -> re-call
  _tube_serves for 'cusp-window'; refused_points=empty(0,3) -> re-call
  _farfield_serves for 'refusal-ball'), never re-deriving guard math. Priority
  gamma-guard -> dropped-sliver -> cusp-window -> refusal-ball -> out-of-box.
  Professor Q7 (near-cusp neighbor arc) lands in out-of-box naturally: with
  cusps relaxed the theta-range gate still fails.
- OWED to Test Dev (WP-CS): fallthrough_breakdown reports a 6th bucket
  `engine_refused` BEYOND the brief's 5 categories (samples where
  geometry_partition raised a named refusal BEFORE any surrogate guard — not a
  surrogate decision). Defensive partition assert is
  served + engine_refused + sum(5 cats) == N; the 5 guard cats partition
  (N - served - engine_refused). Hand-computed fixture counts must budget this
  bucket.
- INS-3-001 RESOLVED (verified, no new change by me): surrogate_training.train()
  now accumulates `all_dropped_slivers` across BOTH parities as flat [lo,hi]
  pairs and passes them to _build_provenance(box,config,charts,
  all_dropped_slivers), which writes provenance['dropped_gamma_slivers'] =
  [list(s) for s in ...]. save() serializes the whole provenance via
  json.dumps -> load() json.loads, so the field round-trips. Census
  _dropped_slivers_from(surr, None) reads it by default; _normalize_slivers
  unpacks `for lo,hi in ...` — shapes match. Runtime-verified end to end:
  _build_provenance -> json round-trip -> census read recovers
  ((0.98,1.02),(1.19,1.205)). The finding was written against the EARLIER
  state (see stale note below); threading was added since, so no re-edit.
- INS-1-001 is a Test Developer deliverable (test_lensing_surrogate_census.py)
  — NOT a Coder task; declined authorship (I authored the census module;
  code+its blessing must not share an author). Routed to Test Dev.
- (superseded) UNVERIFIED discrepancy (WP-CS): brief says read dropped_gamma_slivers "from
  the surrogate provenance field" but landed WP3 writes it only to the TRAINING
  REPORT (report['parities'][label]['dropped_gamma_slivers']), NOT
  surrogate.provenance. Handled defensively: _dropped_slivers_from reads
  provenance.get('dropped_gamma_slivers') (empty if absent) AND accepts an
  explicit `dropped_slivers` override; added public helper
  dropped_slivers_from_training_report(report) so the CLI (--dropped-slivers-
  report) stays thin. If a future WP adds slivers to provenance this path
  still works.
- lnL tiers (stage 4) are dependency-injected via lnlike_pair callable
  (default None -> stage skipped/returns None); CLI does NOT wire a likelihood
  (kept thin). assign_tier keys on gamma/eta ONLY (F017, never theta):
  gamma>1 OR gamma'>=0.5 OR eta<=CROWN_CAUSTIC_MARGIN -> strong_saddle else
  crown; rescued via injected best-effort predicate. Held-out eps uses FRESH
  engine_factory(w).evaluate (F002), currency max|Es-Ee|/max(max|Ee|,1e-6).

- WP-REG (Build 8c-cont registration): registered lens_amplification_surrogate
  as a first-class data product across 3 spec/tooling files, NO source touched.
  (a) DATA_CONTRACTS.yaml: new `lens_amplification_surrogate` artifact
  (format npz; producer scripts/train_lens_surrogate.py::main; consumers =
  likelihood.py::LensedRelativeBinningLikelihood,
  marginalized_likelihood.py::LensedMarginalizedExtrinsicLikelihood,
  surrogate_census.py::run; conventions units+array_axes), schema_version
  0.1.0->0.2.0 (MINOR). All function tokens find_symbol-verified BEFORE writing
  so check_data_contracts passes. (b) data_registry.yaml: added `package_data`
  storage_root (path cogwheel/data) + entries.lens_amplification_surrogate ->
  makes pipeline_graph.py report registry_path=yes (verified: `list` shows
  registry_path=yes, `trace` shows all 3 consumers). (c)
  regenerate_consumer_graph.py LOADERS += LensAmplificationSurrogate.load entry
  (idiom-matched, AST+import verified). sync_derived_docs.py --check RC=0.
- UNVERIFIED (WP-REG): CONSUMER_GRAPH.json NOT regenerated — ripgrep (`rg`) is
  not installed anywhere on this machine (not next to interpreter, not on PATH),
  so scripts/regenerate_consumer_graph.py dies at the FIRST loader
  (_files_importing needs rg), unrelated to my new entry. CONSUMER_GRAPH.json
  left UNTOUCHED; LOADERS edit is correct so post-gate doc-sync regenerates it
  once rg is available. This is an env/tooling gap, NOT a denied command.

- OWED to Test Dev: 8a test_lensing_surrogate.py references the OLD single-box
  API attrs (_real_coeffs/_knots/envelope_real/envelope_imag) — these moved
  onto charts[0]; public envelope()/in_domain()/save/load/from_engine kept,
  but any test scraping private single-box internals needs re-targeting at
  charts[0].real_coeffs / .knots. NEW multi-chart tube/farfield selection +
  sqrt(eta) fitting + serialization gates are Test Dev's to author.


- WP-B (Build 8b-levers): fused operator.py `_weight_vectors` +
  `_contract_grid` into ONE njit `_fused_contraction(table, z_powers,
  zbar_powers, abs_powers, half_sum, derivs_scaled, w_array,
  gamma_scaled, max_order, dim)`. Dispatch-only merge: the two loop
  nests are inlined VERBATIM (v/v_abs now internal per-call temporaries),
  so accumulation order is byte-identical. Proved bit-identity vs
  HEAD's two-stage pipeline: 1800 output arrays / 300 random trials,
  0 tobytes mismatches (extract HEAD funcs via git show, strip
  @numba.njit decorators, run .py_func-equivalent bodies — njit
  cache=True cannot exec from a string, must strip decorator). half_sum
  kept an ARG; _SERIES_TOLERANCE/_CONSECUTIVE_SMALL/_MIN_ORDER kept
  module globals referenced by name; .py_func exposed.
- OWED to Test Dev (F010 red-capability, test_lensing_batched_operator.py):
  the two BatchedContractionFalsificationTestCase gates patch the now-
  REMOVED `operator._contract_grid` / `operator._weight_vectors`
  py_funcs (~lines 721, 753) — must be re-targeted at
  `operator._fused_contraction.py_func` (series-tolerance test: patch
  fused->py_func + _SERIES_TOLERANCE; gather test: wrap fused with a
  zeroed-half_sum shim). Also add `_fused_contraction` to
  ORACLE_FORBIDDEN_NAMES (line ~173) and drop the two dead names.
  Until updated these two tests ERROR (AttributeError), not vacuous.
