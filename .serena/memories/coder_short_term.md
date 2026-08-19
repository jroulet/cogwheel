# Coder Short-Term Observations

## 2026-08-19 findings-fix (INS-1-001 DD-ceiling clip in tiling_plan, WP1)
- FIX: `_measured_w_range` (cogwheel/lensing/tiling_plan.py) now clips the
  chart w-axis UPPER edge at the DD ceiling in BOTH branches: the measured
  branch (`w_hi = min(exp(max(log_hi)), ceiling)`) and the
  prior_box_fallback branch (`min(box.w_range(parity)[1], ceiling)`).
  w_lo stays MEASURED (NOT the forbidden blanket [w_floor,60]) — the clip
  only bites the above-ceiling (60,150] engine_residual leak that
  serve_route_census emits (exact_wave route, DrawResult.log_w_max left
  UNCLIPPED across the whole QD/mpmath band). New source tags:
  'measured_clipped_dd' / 'prior_box_fallback_clipped_dd' (unchanged cells
  keep 'measured' / 'prior_box_fallback').
- SINGLE-SOURCE: `w_ceiling_dd` threaded from `build_plan` (read from
  `census['header']['w_band_edges']['w_ceiling_dd']`, == 60) through
  `_plan_region` -> `_plan_band` -> `_measured_w_range`. New engine-free
  lazy resolver `_resolve_dd_ceiling(w_ceiling_dd)` returns the passed
  value, or on None falls back to `chang_refsdal._schwinger.
  W_CEILING_SCHWINGER` (lazy import — NO mpmath pulled). NOTE:
  serve_route_census imports `_schwinger` LAZILY inside a function, so
  there is no `serve_route_census.W_CEILING_SCHWINGER` module attr — import
  from `cogwheel.lensing.chang_refsdal._schwinger` directly.
- BACKWARD-COMPAT DEFAULT: all three helpers gained a TRAILING
  `w_ceiling_dd: float | None = None` param (not a required positional) to
  avoid stranding the Test Developer's existing positional-arity callers in
  test_lensing_tiling_plan.py (their _CELLS fixtures are all sub-ceiling
  w_hi<=60, so None-default keeps them green). `cost_model` dict gained a
  `'w_ceiling_dd'` field.
- `_n_w_nodes` already guards `w_hi<=w_lo or w_lo<=0 -> return 1`, so
  clipping w_hi to 60 cannot crash even in the (non-DD-demand) case w_lo>60.
- VERIFIED (Serena shell, full python path): explicit clip 150->60 with
  w_lo=2.0 preserved (source measured_clipped_dd); default-None resolves to
  60 and clips; sub-ceiling 38 unchanged (source measured); fallback branch
  480->60 (source prior_box_fallback_clipped_dd); `_resolve_dd_ceiling(None)
  ==60.0`; mpmath absent at import AND after helper use; CLI --help rc 0.
- INS-2-001 (pipeline-routed to Coder in a findings-fix pass; normally a
  Test Dev cell, but the finding was fully prescriptive + no production
  change): extended MeasuredWAxisEdgeTestCase in
  test_lensing_tiling_plan.py with 3 boundary-witness cases for the
  INS-1-001 DD-ceiling clip — (1) engine_residual w_hi=150 default-ceiling
  -> got_hi==60 & status 'measured_clipped_dd'; (2) empty records +
  box.w_range=(2,480) -> got_hi==60 & status 'prior_box_fallback_clipped_dd';
  (3) explicit w_ceiling_dd=45 kwarg with w_hi=150 -> got_hi==45 (pins the
  ceiling is config-sourced, NOT the module constant 60). All reuse the
  existing `_residual_record`/`SimpleNamespace(w_range=...)`/`_observe`
  anti-vacuity pattern; asserting the new status strings is the teeth
  (clip removal returns 'measured'/'prior_box_fallback' with got_hi
  150/480 -> all 3 red). Full file 39 passed (was 36).

## 2026-08-18 build (demand-sized tiling_plan module + CLI, WP1)
- Created NEW `cogwheel/lensing/tiling_plan.py` (pure engine-free
  demand-sized tiling + cost predictor, no I/O/print) + NEW
  `scripts/tiling_plan.py` (thin CLI: argparse -> run -> json.dump ->
  print verdict). Follows tiling_census.py's lazy-import pattern
  (`tiling_census._load_production_modules()` -> (st, sg)); NEVER imports
  surrogate_census (engine at module load). SCHEMA='tiling_plan_v1'.
  Output: .claude/handoff/tiling_plan_and_cost_7a2.json.
- ENGINE-FREE VERIFIED (smoke, via Serena shell): import OK; `mpmath in
  sys.modules` == False after import AND after pure-helper use; CLI
  `--help` rc 0. Module docstring carries a BOOBY-TRAP note for the Test
  Developer (mock.patch evaluate entry points, assert mpmath absent).
- LOAD-BEARING SYMBOL AUDIT (all confirmed present at HEAD, do not
  re-doubt): `st._scalar_caustic_reach` IS real — surrogate_training.py
  line 64 aliases `_caustic_reach as _scalar_caustic_reach` and exposes it
  as a MODULE ATTRIBUTE (used at 1305/5176/5290), so `st._scalar_caustic_
  reach(gamma)` is valid for the gamma-axis central-difference. Also
  confirmed: st._min_curvature_radius, st._trim_tube_arc, st._self_estimate,
  st._coordinate_radius_bounds, st.PriorBox.from_prior_classes;
  tiling_census._REQUIRED_CONFIG_FIELDS/_collect_band_contexts/_admissible_
  regions/_count_region/_spatial_nodes_per_tile/_load_production_modules;
  serve_route_census._gamma_band_of/aggregate_cells/residual_demand/run.
- DATA-SHAPE AUDIT (serve_route_census.run report, schema
  serve_route_census_v1): consumed keys VERIFIED against source —
  header.gamma_band_edges (list[float]), records (list of
  DrawResult.as_record with fields route/region/gamma_band/w_band/
  log_w_min/log_w_max — LOG_W ARE NATURAL LOGS, w=exp(log_w)), cells
  (aggregate_cells: each cell has region/gamma_band/w_band/total/routes,
  routes is a dict pre-initialized with ALL SERVE_ROUTES keys so
  `cell['routes'].get('engine_residual',0)` is always present). Demand gate
  keys on the COARSE census gamma-band grid (ppgo_map._gamma_band_edges via
  header), mapping FINE stable-band ctx.gamma_mid through
  serve_route_census._gamma_band_of.
- build_plan MIRRORS tiling_census.run's box/context construction verbatim
  (`box = st.PriorBox.from_prior_classes()`, `tiling_census._collect_band_
  contexts(st, box, parity, config)`, `_admissible_regions(parity, None)`)
  — no independent re-derivation, so any prior-box drift stays in ONE place.
- COST CURRENCY: SECONDS_PER_CALL=0.0903 (per node-label = per call, DD-band
  smoke) reconciled in-code + in emitted cost_model.note against
  tiling_census._SECONDS_PER_LABEL=0.09 (~0.3% smoke jitter). total_calls =
  total_nodes * _LABELS_PER_NODE(8). Three cross-check ratios emitted (vs
  st._self_estimate, vs tiling_census aggregate_call_count, vs 0.4119
  engine_residual ledger). Escalation verdict NEVER raises (records
  should_escalate = calls>5e5 or max_region_share>0.40 + reasons).
- UNVERIFIED (downstream measurement, Test Dev / Inspector): the full
  10k-sample `run()` and hence the actual .claude/handoff JSON artifact
  (total calls, wall-clock hours, per-region share table, escalation
  verdict, the reconciled ~41% engine_residual share, the three
  cross-check ratios) were NOT executed here — that is a census/measurement
  campaign owned downstream, not a Coder self-cert. Code path + every
  delegated signature + every consumed data key verified statically.

## 2026-08-18 build (tube_trainer_subarc_trim / F083 promotion, WP1)
- Promoted the F083 trimmed-sub-arc algorithm into
  `surrogate_training.py` as `_trim_tube_arc(*, band, arc, eta_max,
  parity)` (keyword-only), verbatim from the test fixture's
  `_f083_shared_tube` scan: 80-pt Delta_tau scan at binding corner
  (gamma_hi=band[1], eta_max); constants _TUBE_TRIM_DTAU_FRAC=0.6,
  _TUBE_TRIM_LO_STANDOFF=0.20, _TUBE_TRIM_HI_STANDOFF=0.05,
  _TUBE_TRIM_SCAN_POINTS=80. PARITY GATE FIRST: `if parity != 1: return
  arc` (verified `out is arc` -> byte-identical saddle). Imports added:
  `_frame_delays` from chang_refsdal.channels, `_merging_fold_pair` from
  chang_refsdal._airy_fold, `replace` from dataclasses. Wired into
  `_train_band_charts` per-arc loop AFTER eta sizing (eta_max/eta_floor
  stay full-arc); reassigned loop-local `arc` so the `build_tube` closure
  default `arc=arc` (bound at def-time, AFTER the trim) carries the
  trimmed arc into `_build_tube_chart`/`_tube_heldout_samples`/
  theta_range. arc_r_min/max_eta_max/min_eta_max/tube_w_range all computed
  BEFORE the loop, untouched. Module imports clean.
- UNVERIFIED (engine build, downstream): parity==1 astroid trim produces
  the sweep-consistent eps with refused=0 at production density — needs
  the ~200s spot-check (Test Dev / Inspector / Professor). Fixture
  re-point to the production helper (DRY) is a separate WP, not this one.
