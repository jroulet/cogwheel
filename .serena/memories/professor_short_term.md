# Professor Short-Term Observations

## Ghost decay gate domain review (2026-08-02)

- **test_lensing_ghost_decay_gate.py**: 18/18 pass (14s). All five specified
  tests certified:
  1. **Decay refusal** (near-axis, gamma=1.6, theta=0.02): Im(tau_c)=0.044 < 0.4
     threshold → GhostDomainError. Separation=0.94 > 0.7 → separation gate ADMITS.
     The two gates are provably independent.
  2. **Well-decayed admit** (gamma=1.5, y=(2,2)): Im(tau_c)=0.825 > 0.4 → admits.
     |G| decays from 9.07e-2 to 7.52e-5 across band (factor ~800×, exponential).
  3. **Few-images refusal**: real_images=[] raises GhostDomainError (no images to
     separate from).
  4. **Protective refusal (self-falsification)**: forcing ghost into refused config
     worsens residual by 37.5% (norm_with/norm_without = 1.375). The gate IS
     protective, not overprotective.
  5. **Train/serve skew impossibility**: both w-grids (0.5..10 and 2..10) give
     identical ADMIT decision on the same config; both give identical REFUSE on
     the near-axis config. The gate is provably w-independent (no frequency in
     the criterion Im(tau_c) >= constant).

- **test_lensing_ghost.py**: 31/31 pass + 1 xfail (9.3s). Ghost primitives,
  selection, guards, self-falsification all clean.
- **test_lensing_chang_refsdal_ghost_frame.py**: 12/12 pass (3.5s). Frame
  conventions bit-identical.
- **test_lensing_ghost_gate.py**: 12/13 pass (12s). 1 FAILURE in
  `GhostSeparationConstantReachableRedTestCase.test_raising_constant_to_two_refuses_an_admit_config`
  — pre-existing issue: ADMIT_CONFIGS[0] has sep=2.58 > 2.0 so patching MIN to 2.0
  doesn't flip it. This is a test design issue in the SEPARATION gate reachable-red
  (not the new decay gate). Not a regression from the current build.

- **Physics**: The decay gate `Im(tau_c) >= 0.4` is the correct criterion.
  Near a principal axis of the Chang–Refsdal macro matrix, the critical point
  approaches the real axis (Re axis of the Fermat surface), Im(tau_c) → 0, and
  the ghost contribution e^{iw tau_c} oscillates rather than decaying — it is
  NOT a small correction to the kernel sum. The threshold 0.4 = 2.0/5.0 derives
  from requiring at least _FARFIELD_WINDOW_RADIANS of attenuation at the chart
  band floor w ~ 5. The criterion is a pure lens-configuration property (no w)
  so train/serve skew is structurally impossible.

- Heavy full-sampling validation is operator-deferred.
