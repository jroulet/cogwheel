# Professor Short-Term Observations

## Ghost gate orthogonality witness review (2026-08-02)

- **test_lensing_ghost_gate.py**: 17/18 pass, 1 pre-existing failure (17s).
  All 5 orthogonality-witness tests (`GhostGateOrthogonalityWitnessTestCase`)
  certified PASS:
  1. **Decay gate passes**: Im(tau_c)=0.502 > 0.4, margin=0.102 > 0.05 minimum.
  2. **Separation gate refuses**: GhostDomainError raised with "separation" in
     message, confirming the code reached past the decay gate.
  3. **Separation independently below**: sep=0.600 < 0.7, margin=0.100 > 0.05.
  4. **Disabling sep gate admits**: with _GHOST_SEPARATION_MIN=0.0, the config
     admits (decay gate is the only remaining guard), proving independence.
  5. **Scatter diagnostic**: plot produced showing non-nested refusal regions.
     Green star at (Im(tau_c)=0.50, sep=0.60) is in the "orthogonal quadrant"
     (decay passes, sep refuses), distinct from the refuse-both (red) and
     admit-both (blue) clusters.

- **Physics certification**: At positive parity (gamma < 1), Im(tau_c) and
  separation are strongly correlated — both driven by the geodesic distance
  from the caustic in the Fermat potential. Sweep confirms NO positive-parity
  config exists with Im(tau_c)>=0.4 AND sep<0.7 simultaneously. At saddle
  parity (gamma=5.0 > 1), the critical-point topology changes (deltoid
  caustics, two lobes) and the coupling breaks: the ghost can decay (high
  Im(tau_c), meaning large imaginary part of the Fermat delay at the complex
  stationary point — exponential suppression e^{-w*Im(tau_c)}) while remaining
  spatially close to a real image (low separation, meaning the single-saddle
  expansion has overlapping contributions and is INVALID). The separation gate
  is therefore independently load-bearing: without it, the decay gate alone
  would wrongly admit configs where the stationary-phase approximation fails.

- **Pre-existing failure**: `test_raising_constant_to_two_refuses_an_admit_config`
  — ADMIT_CONFIGS[0] has actual separation=2.012, so patching threshold to 2.0
  doesn't flip it. Test design issue (comment says "sep~1.98" but measured 2.01).
  Not a regression from the current build; unrelated to the orthogonality witness.

- **test_lensing_ghost_decay_gate.py**: 18/18 pass (8s). All decay gate tests
  clean. The two test files together certify the two gates are independent,
  non-redundant, and each load-bearing for its own failure mode.

- Heavy full-sampling validation is operator-deferred.
