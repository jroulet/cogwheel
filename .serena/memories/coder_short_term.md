# Coder Short-Term Observations

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
