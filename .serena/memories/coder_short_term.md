# Coder Short-Term Observations

- WP1 Born Residual Chart wiring: Created cogwheel/lensing/born_residual_chart.py
  (frozen BornResidualChart dataclass with covers/evaluate methods, lazy-cached
  RegularGridInterpolator on (gamma, rho, log_w) grid). Wired into likelihood.py:
  added FARFIELD_KERNEL_SUM to channels import, added `import math` + `import types`,
  added born_residual_chart=None optional kwarg to __init__ (stored as attribute,
  rides pickle in __dict__), extended get_init_dict (pop when None, raise
  NotImplementedError when set — same pattern as amplification_surrogate). Replaced
  fact-4 slot's bare `return None` with the Born residual serve path: guards
  (born_chart is None, caustic_rho ValueError/LensDomainError, rho<=1.0, covers()),
  SimpleNamespace adapter duck-typing born_carrier_from_partition, deferred import
  of born_carrier_from_partition inside the slot, carrier+residual=f_total, far-field
  reconstruction via reconstruct_farfield with FARFIELD_KERNEL_SUM definition and
  geom.t_min. Both files parse clean (ast.parse + import OK). The slot only fires
  when amplification_surrogate is not None AND surrogate.serve() declines
  (served=False), so born_residual_chart=None is byte-identical to HEAD.
  UNVERIFIED: runtime correctness of the reconstruction telescoping (carrier +
  residual - ppgo gives the correct far-field envelope for reconstruct_farfield).


- WP1 Build 1e-gamma: Added `_log_reach_gamma_axis` function (lines 1273-1331)
  to cogwheel/lensing/surrogate.py as a peer of `_uniform_axis`. Function places
  gamma nodes equispaced in log(caustic_reach) space via 200-point fine sweep +
  np.interp inversion, handles both increasing (positive parity) and decreasing
  (saddle) log-reach monotonicity. Replaced all 3 gamma-axis `_uniform_axis` calls:
  from_engine (line 2865), from_lobe_engine (line 3051), and _train_band_charts
  (surrogate_training.py line 3776). Added import to surrogate_training.py line 60.
  `_uniform_axis` left unchanged for s, d, rho_lobe axes. Test file
  test_lensing_surrogate_lobe.py line 1973 deliberately uses _uniform_axis for gamma
  (testing uniform behavior) — NOT changed per WP scope.
  Both files parse clean (ast.parse OK). No LSP diagnostics.

- WP1 Build 7: Added Part 0 resolution paragraph to _GHOST_SEPARATION_MIN
  comment block in channels.py (lines 207-217, 11 new comment lines).
  Updated COVERAGE_DESIGN.md table entry from SUSPECT to OK. No behavioral
  change — value 0.7 unchanged, all 22 test references untouched.

- INS-7-001 FIX (Build 6 C5, pass 4): Updated 3 test files to accommodate
  the new decay gate (_GHOST_DECAY_IM_THRESHOLD = 0.4):
  (a) test_lensing_ghost_gate.py: ADMIT_CONFIGS[0] offset 0.60→0.65 for
      margin; TRAIN_SERVE_ADMIT_CONFIG matches.
  (b) test_lensing_born.py: SADDLE_GHOST_THETA_RANGE (0.02,0.6)→(0.20,0.6)
      to exclude near-axis configs with Im(tau_c)<0.4.
  (c) test_lensing_exterior_windows.py: GhostGateTestCase/TagContractTestCase
      rho 2.0→1.65 (theta=45°); WindowSeamReconstructionTestCase rho
      1.2,theta=30→1.65,theta=45; MidWindowGhostTestCase rho (1.9,2.1)→
      (1.6,1.7); SelfFalsificationTestCase also patches _GHOST_DECAY_IM_THRESHOLD;
      test_near_axis_ghost_degrades_the_label converted to verify decay-gate
      REFUSAL (the pinned limitation it documented is now FIXED).
  UNVERIFIED: exact Im(tau_c) values at new configs (computed from F027 data
  and ghost_gate ADMIT_CONFIGS annotations; not runtime-verified).

- INS-6-002/003/004 FIX (Build 6 C5, pass 3)