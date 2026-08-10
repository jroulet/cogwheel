# Professor short-term observations — 2026-08-10

## Ghost-gate exterior admission test review

All 39 tests across DT-1 through DT-8 passed. Key findings:

1. **Physics gate correct.** The `_exclude_ghost_dominated` function correctly distinguishes three cases:
   (a) ghost nonexistence → retainable (no ghost to contaminate KERNEL_SUM)
   (b) ghost exists + Im(tau_c) >= 0.4 → retainable (ghost decays: exp(-w*0.4) ≤ 0.135 at w=5)
   (c) ghost exists + Im(tau_c) < 0.4 → excluded (unsubtractable oscillatory ghost dominates)
   Only case (c) fires the exclusion — matching the spec precisely.

2. **Threshold at Im(tau_c)=0.4 physical.** At the chart floor w≈5, exp(-5*0.4)=exp(-2)=0.135 — a genuine small correction. At the failure config (Im~0.013), exp(-5*0.013)=0.94 → ghost is unsubtracted. Threshold derived from _FARFIELD_WINDOW_RADIANS/5.0.

3. **Multi-gamma probing works.** At the band edge gamma_lo=0.46, Im(tau_c) drops below 0.4 while at gamma_mid=0.5 it's above — the band-edge probe catches what the single-gamma probe misses. Correct.

4. **Center probe catches straddling tiles.** A tile whose 4 corners all have Im(tau_c) >= 0.4 but center has Im(tau_c) < 0.4 is correctly excluded by the 5-point probe.

5. **All self-falsifications pass.** Each positive assertion has a companion test that patches _GHOST_DECAY_IM_THRESHOLD to 0.0 or mocks the gate, confirming the assertions have teeth.

6. **End-to-end wiring verified.** The ghost_drop_count counter flows from _farfield_exterior_tiles through _train_band_charts into the exterior region report's `ghost_excluded_tiles` key.

7. **Saddle branch tested.** DT-3 and GhostDominatedSelfFalsificationTestCase:test_saddle_parity_ghost_exists_below_threshold cover saddle parity where ghost either doesn't exist or exists with Im(tau_c) below threshold.
