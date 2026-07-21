# Test Dev Short-Term Observations

## 2026-07-20 build8d test_lensing_surrogate BYTE-PIN RE-BASELINE (WP1)
- WP1 reroutes SHEARED positive-parity F_op/F_op_grid (gamma'>0) from
  legacy operator-series to exact Schwinger; legacy demoted to
  `operator.legacy_operator_oracle` (=_grid_certified, NOT in __all__) +
  gamma'==0 point-lens exit. Positive-parity F changes ~5e-15 (byte flip).
- CRITICAL: the existing CrownByteIdentityTestCase does NOT catch this —
  `_head_likelihood_class()` reloads only HEAD likelihood.py; both cur+head
  import the SAME working-tree operator.py, so it stays green (correctly
  certifies 8a likelihood wiring is additive-neutral). Only operator.py is
  modified this build. So I ADDED CrownContractFlipWitnessTestCase (8 tests)
  witnessing the flip at the F_op level: NEW Schwinger vs OLD
  legacy_operator_oracle (independent algo, F002) on the legacy-CERTIFIED
  overlap, max-normalized 1e-10. `_flip_witness_metrics` collects overlap
  per-node (legacy per-node loop cheap ~0.26s/60; Schwinger ~58ms/node),
  batches F_op_grid on overlap, gates max(metric_re,metric_im)<1e-10.
- Witness table: A/crown,B/two-image,crown,near-fold ~5e-15; sub-critical
  (gamma=0.35) partial overlap (legacy refuses w>17.6) tightest ~4.8e-11 —
  that's the LEGACY oracle's own 1e-10 cert floor near its cancellation
  edge (Schwinger=exact truth), NOT a Schwinger defect; still clears gate.
- Dispatch seams (Python-level even though targets njit): spy
  `mock.patch.object(schwinger_module,'f_schwinger',...)` +
  `mock.patch.object(operator_module,'_grid_certified',...)`. gamma'>0 crown
  -> f_schwinger=n_w, _grid_certified=0; gamma'=0 pointlens (gamma=0.0) ->
  f_schwinger=0, _grid_certified>0. legacy_operator_oracle holds its own ref
  to real _grid_certified so patching operator._grid_certified can't leak.
  F010 mutation: patch f_schwinger*(1+1e-4) -> witness metric 9.2e-5 RED.
  Refusal: F_op(w=68>W_CEILING_SCHWINGER=60, gamma'>0) raises
  SchwingerCertificationError. Full suite 41 passed 1 skipped ~11min.
- NEIGHBOR DRIFT (report, don't touch): test_lensing_operator.py 6 fail +
  4 err — all OLD positive-parity operator refusal contract
  (test_raises_named_error_above_w_ceiling, _former_silent_nan_config_now_
  refuses, _certified_or_named_refusal_across_band, _cancellation_ratio_
  field_matches_independent) that WP1 rerouted to Schwinger. Operator
  suite's own re-baseline run owns these; my change is test-only.

## 2026-07-20 build8c test_lensing_surrogate (TEST 11/12/13)
- TEST 11 (preservation): 8a suite greened vs landed multi-chart surrogate.
  Root cause of 7 reds = `_refusal_surrogate()` box (0.8,1.2) centres on
  gamma=1.0, and `from_engine` calls UNWRAPPED `_box_region_labels` at box
  centre -> LensDomainError. Fix: box->(0.8,1.3), n_gamma=6 preserves 0.1
  spacing + the gamma=1 refusal column, nudges centre to valid 1.05. No
  tolerance weakened. Only 2 real single-box scrapes needed re-target:
  `_param_spacing`->`charts[0].param_spacing`; F010 mutation now patches
  `surrogate_module._in_exclusion_ball` (the load-bearing guard both
  `envelope`+`in_domain` share via `_farfield_raw_chart`), not `in_domain`.
  Property shims (gamma_grid etc.->charts[0]) already cover the rest.
- TEST 12/13 added: `_multichart_fixture()` (lru_cached) builds a 4-chart
  surrogate (pos_tube/pos_ff/sad_tube/sad_ff) from synthetic smooth tensors
  via TubeChart/FarFieldChart.from_values — NO engine calls, runs instant.
  Overlap band engineered via ff eta_overlap_min=0.02 < tube eta_max=0.05
  (eta in (0.02,0.05] double-matches -> tube priority wins). Negative-theta
  saddle wedge theta_grid=[-0.39,-0.09] exercises `_theta_into_frame` unwrap
  (query theta=2pi-0.19 -> -0.19). Provenance MUST use lists not tuples
  (json round-trip value-equality). refused_points always (n,3) via
  `_normalize_refused` (None->empty(0,3)) so npz allow_pickle=False safe.
  Self-falsification: dataclasses.replace(tube, eta_max=0.025) flips overlap
  selection tube->ff. No-double-serve proved by isolating each chart in a
  1-chart surrogate. Full suite: 33 passed, 1 skipped (timing opt-in), ~5min.
