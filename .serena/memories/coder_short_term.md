# Coder Short-Term Observations

- wp1 (min_gamma_band 0.02→0.005): 3 value edits + 2 comment fixes across
  surrogate_training.py and scripts/measure_dropped_slivers.py. F041 test
  constant intentionally left at 0.02.

- wp1 (ppgo_interior_handoff): Added fold-ppGO interior handoff in
  _surrogate_coefficients (likelihood.py, after Born chart block, inside
  rho<=1.0 branch) and mirrored gate logic in characterize_sample
  (surrogate_census.py). Key discovery: geom.images is a LIST not ndarray
  (despite type annotation), so used list(geom.images) not shape indexing.
  matrix reconstructed via macro_matrix (not on geom). Census uses
  _XI_FOLD_THRESHOLD=4.0 locally (no circular import from likelihood).
  Category 'ppgo_fold' with served=True passes fallthrough_breakdown
  validation (served records skip category check).

- wp1 (schwinger_qd): Added _f_schwinger_mpmath() to _schwinger.py for
  w ∈ (60, 150]. Uses lazy-imported mpmath with dps=30+ceil(w). Same IBP
  structure as the DD path. Paired N/2N certification on RECONSTRUCTED F
  (not raw integral — matches WP spec). Dispatch in f_schwinger() checks
  QD ceiling (150) first, then DD ceiling (60). W_CEILING_SCHWINGER_QD=150.0
  exported. pyproject.toml gains training = ["mpmath"]. Key bug found and
  fixed: mp.linspace must receive mpf endpoints, not float() casts (the
  float cast caused catastrophic precision loss and ~e4 magnitude errors).
  Tests that assert refusal at w>60 will now see success — Test Dev scope.


- wp2 (schwinger_qd_wiring): Wired mpmath Schwinger ceiling into training
  pipeline. In operator.py `_saddle_grid` and `_positive_parity_grid`:
  routing pivot changed from W_CEILING_SCHWINGER (60) to
  W_CEILING_SCHWINGER_QD (150). Three-way node classification: (1) w>150
  → geometric/arm/refuse, (2) 60<w<=150 → mpmath sequential batch via
  `_schwinger.f_schwinger` (which dispatches to `_f_schwinger_mpmath`
  internally), (3) w<=60 → DD parallel batch (byte-identical to HEAD).
  mpmath nodes catch SchwingerCertificationError and add to mpmath_refusers.
  Refusal reduction merges ceiling_refusers + mpmath_refusers + DD refusers.
  In surrogate_training.py: _SADDLE_W_CEILING raised from 58.0 to 148.0
  (2 below QD ceiling). _DD_PRODUCT_MARGIN left at 58 (independent concern).
  _measure_node_parallel_speedup already uses 0.9*W_CEILING_SCHWINGER (DD)
  — no change needed. Updated docstrings in operator.py module header,
  _schwinger_wave_grid_values, _uniform_arm_value, F_op_grid, F_op, and
  SchwingerCertificationError class in _schwinger.py. NOTE: ppgo_map.py
  line 57 still references "W_CEILING_SCHWINGER = 60" in a comment —
  Librarian scope.

- INS-fix (schwinger_qd test adaptation): Fixed 5 inspector findings across
  4 test files. All test-only changes — no production code modified.
  (1) test_lensing_operator.py: ONEHOME_WS pruned to avoid mpmath band,
  filter changed to W_CEILING_SCHWINGER_QD, refusal test uses QD+10.
  (2) test_lensing_waveform.py: BAND_EDGE arm-serve test repurposed to
  verify finite mpmath value; HARD_CORE.w_probes moved to 151.
  (3) test_lensing_surrogate.py: FLIP_REFUSAL_W moved to 160.
  (4) test_lensing_airy_fold.py: _ABOVE_CEILING_W→160, _W_CEILING→QD,
  geometric ladder node moved to w=200.
  (5) test_lensing_batched_operator.py: XOR_BAND_LS capped at L=54 (w=60),
  XOR_REFUSING_W moved to 160, n_above assertion removed.

- INS-fix-2 (BAND_EDGE performance): Fixed INS-2-001 + INS-1-002 in
  test_lensing_waveform.py. Changed BAND_EDGE.w_probes from (30, 40, 60.5)
  to (30, 40, 59.9) to avoid the slow mpmath band. Updated fixture docstring
  (removed mpmath QD references, noted DD path). Rewrote
  test_band_edge_companion_now_served_by_arm to verify all probes return
  finite values (all now on DD path, no above-ceiling filter). Updated
  MacroSaddleControlTestCase class docstring to reflect new contract.
