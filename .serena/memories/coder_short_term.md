# Coder Short-Term Observations

- INS-1-001/002 fix (Build 3 C6 inspector fixup): Added _LOBE_ETA_MAX=0.05
  constant to test_lensing_surrogate_lobe.py and passed to both
  _saddle_lobe_admissions call sites. In scripts/measure_tube_fraction.py,
  removed dead dataclasses.replace (config no longer has eta bounds) and
  passed eta_max/eta_floor as explicit kwargs to _build_tube_chart and
  _tube_heldout_samples. Removed unused `import dataclasses`.


- WP1 (Build Step 3 C6): Replaced fixed eta_max/eta_floor constants with
  curvature-relative f_max/f_floor in surrogate_training.py. TrainingConfig
  fields renamed (eta_max→f_max, eta_floor→f_floor). Per-arc absolute
  eta_max=f_max*R_c computed in _train_band_charts and threaded to
  _build_tube_chart, _tube_heldout_samples, _saddle_lobe_admissions, and
  _interior_admission. Foot-of-normal skip guard deleted, replaced by
  assertion (f_max < 0.5). Test files (exterior_windows, caustic_cusps,
  surrogate_training, exterior_admission, ppgo_bandsplit) still reference
  the OLD config.eta_max field — they will need updating by Test Dev.
