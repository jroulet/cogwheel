# Coder Short-Term Observations

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