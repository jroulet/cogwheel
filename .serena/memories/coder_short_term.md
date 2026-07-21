# Coder Short-Term Observations

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
