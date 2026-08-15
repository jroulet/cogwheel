# Coder Short-Term Observations

## 2026-08-15 build (F081 Inspector-fix — retired max_tube_arcs in tests)

- Loop-break fix (Inspector explicitly directed) of 3 findings, all in test
  files, all referencing the F081-removed `TrainingConfig.max_tube_arcs`:
  - INS-1-001 test_lensing_surrogate_training.py L6329: dropped
    `max_tube_arcs=4,` from DegenerateExteriorBandIsRecordedTestCase.CONFIG
    literal (class-body execution => collection-time TypeError => whole 6300-
    line suite dead). py_compile does NOT catch this (class body not run);
    `pytest --collect-only` does.
  - INS-1-002 test_lensing_tiling_census.py: replaced the
    `min(_SADDLE_DETECTED_ARCS, config.max_tube_arcs)` assertion with an
    INDEPENDENT re-derivation — rebuild the census's own first-saddle ctx via
    `tc._collect_band_contexts(st, box, -1, config)[0].structure`, expected =
    `len(st._tube_training_arcs(structure, -1))`; non-vacuous because it
    re-runs the selector rather than reading rep.tube_arcs. Teeth: `1 <=
    trained < 6`. Deleted `test_saddle_slice_widens_with_max_tube_arcs`
    (retired knob, `TrainingConfig(max_tube_arcs=2)` TypeError), replaced with
    `test_saddle_folds_strictly_below_detected_while_astroid_pins_one`.
    Updated ArcCensusQ1TestCase docstring + the `_SADDLE_DETECTED_ARCS`
    constants comment (both cited the retired slice). GROUNDING NOTE: did NOT
    hardcode the saddle count (~3) — asserted `== production-selector` +
    strict `< detected`, per the no-hardcode-without-fresh-execution rule.
  - INS-1-003 test_lensing_caustic_cusps.py L1302/L1661: both
    `structure.arcs[:config.max_tube_arcs]` -> `st._tube_training_arcs(
    structure, 1)` (astroid single pi/4 arc; max(arc_r_min) unchanged).
- Remaining `max_tube_arcs` matches tree-wide are INERT PROSE only:
  test_lensing_tube_d2_fold.py L48/L582 (docstrings documenting the knob is
  retired, 2-arg signature) + my own new tiling_census docstring L550.
- VERIFICATION: py_compile OK on all 3 edited files; `pytest --collect-only`
  on all 3 => 255 tests collected, rc 0 (the exact Inspector repro, now
  green). Did NOT run the suites (Coder scope).

## 2026-08-14 build (F081 saddle tube D2 trim + lobe-edge shell, WP1)

- surrogate_training.py: `_tube_training_arcs(structure, parity)` — dropped
  `max_tube_arcs` param; saddle branch now partitions detected deltoid arcs
  into D2 gauge orbits (midpoint match under {theta, pi-theta, -theta,
  pi+theta} within tol=max(1e-3, 0.25*min_width)), returns first rep per
  orbit (count follows partition, NOT hardcoded 3). New helper
  `_circular_angular_distance`. Astroid branch (pi/4-bracket predicate)
  byte-unchanged. `_EXPECTED_ARCS` + F079 guard untouched.
- PART B: `_train_band_charts` computes BOTH max_eta_max=f_max*max(arc_r_min)
  AND min_eta_max=f_max*min(arc_r_min) (fallback f_max*0.05). Routed
  min_eta_max -> `physical_exclusion_radius`(far-field inner edge/exclusion_rho)
  + `_saddle_lobe_admissions(eta_max=)`. KEPT max_eta_max on tube w-cap
  (`_capped_w_range`), astroid interior-skip(L5374), wedge r_extent(L5429),
  `_interior_admission`(L5034). Single-arc astroid => min==max => astroid
  arithmetic byte-identical.
- Removed `TrainingConfig.max_tube_arcs` field+comment. tiling_census.py
  mirror: dropped 'max_tube_arcs' from `_REQUIRED_CONFIG_FIELDS`, dropped
  arg from selector call, added min_eta_max, routed exclusion_rho +
  saddle_lobe_admissions to min_eta_max (kept ctx.max_eta_max on parity==1
  _count_exterior/_count_wedge_interior). scripts/train_surrogate_production.py
  removed `max_tube_arcs=20` + its print. scripts/census_dry_run.py:127 now
  calls `_tube_training_arcs(structure, _SADDLE_PARITY)` + min_eta_max.
- py_compile OK on all 4 files. ACCEPTANCE NUMBERS (nonzero lobe/ff counts,
  tube:-1 node drop from 61,740) are a MEASUREMENT CAMPAIGN (engine-free
  census run) => UNVERIFIED by Coder, left for driver re-run.
- BREAKS EXISTING TESTS (flag Test Developer, NOT my scope): 
  test_lensing_tube_d2_fold.py (TubeTrainingArcSelectionTestCase 3-arg
  `_tube_training_arcs` calls L387/410-412; `test_saddle_reconsumes_max_
  tube_arcs_slice`); test_lensing_tiling_census.py (asserts min(detected,
  max_tube_arcs) slice + `TrainingConfig(max_tube_arcs=2)` L529+, L466-541);
  test_lensing_caustic_cusps.py (`structure.arcs[:config.max_tube_arcs]`
  L1301/1660); test_lensing_surrogate_training.py (`max_tube_arcs=4` L6329).
  All will AttributeError/TypeError on the removed field/param — new saddle
  behavior is "one rep per D2 orbit (~3)", not a max_tube_arcs slice.


## 2026-08-14 build (tiling_census WP1 — engine-free tiling census + CLI)

- NEW `cogwheel/lensing/tiling_census.py` `run(config, regions=None)->dict`
  (schema 'tiling_census_v1') + NEW I/O-only CLI `scripts/tiling_census.py`
  (mirrors scripts/census_lens_surrogate.py; default=TrainingConfig(),
  --regions, --out). Continuation build: prior coder left the module
  complete-and-verified but the CLI missing; this session added the CLI +
  ran all smoke/engine-free checks.
- ENGINE-FREE IS NO-CALL, NOT NO-IMPORT, FOR ANY MODULE INSIDE
  cogwheel/lensing/: importing tiling_census necessarily runs
  cogwheel/lensing/__init__.py (imports prior/posterior/
  marginalized_likelihood) AND chang_refsdal/__init__.py (line 4 imports
  channels -> _schwinger). So `_schwinger`/`channels` module OBJECTS load
  at import time and CANNOT be avoided without editing package __init__
  files (out of scope, 38 consumers). The load-bearing + achievable
  guarantee (Professor's authoritative def = amplitude EVALUATION only;
  Test Dev's "assert zero calls"): NO engine CALL + mpmath NEVER loaded
  even after a full run() (verified `'mpmath' not in sys.modules`). The
  WP verification text "importable without transitively importing
  _schwinger" is structurally impossible here — flagged UNVERIFIED for
  Inspector/Test-Dev adjudication.
- SMOKE RESULTS at default TrainingConfig(): 6 triples (tube:+1,
  exterior:+1, wedge_interior:+1, tube:-1, lobe_interior:-1,
  lobe_exterior:-1); aggregate_call_count 96512; self_estimate_seconds
  806.4; Q1 detected 4/6 arcs vs trained 1/1 (max_tube_arcs=1); Q2
  redesign_needed=True (cusp ray strictly inside a deltoid far-field tile,
  mis_alloc_ratio 1.55); Q3 kink_free=True over 19 near-cusp tiles.
- INS-1-001 FIX (triage round 2, disclosure-only): census omits the
  loop-level ppGO trim (_apply_ppgo_trim installed via
  get_certified_ppgo_map() in every real train()), so counts are a
  conservative UPPER BOUND. Fixed per Inspector option (b): (1) new module
  docstring section "CONSERVATIVE UPPER BOUND (no ppGO trim modeled)"; (2)
  sentence added to run() Returns docstring; (3) new output key
  'ppgo_trim_modeled': False placed alongside aggregate_call_count. NO
  counting-loop / _apply_ppgo_trim modeling (option a explicitly out of
  scope), Q4 ppgo_map usage + PpgoTrimIndependenceTestCase untouched.

(previously: empty — last consolidated by Dreamer on 2026-08-14)
