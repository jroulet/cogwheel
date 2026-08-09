## 2026-08-09 — Professor review: lensing exterior admission (cusp exclusion) build verdict

### Tests reviewed
- `ExcludeNearCuspBandEdgeTestCase`: Band-edge checking is load-bearing. Corner dist: 0.1399 at γ=0.25 (excluded, d=0.15), 0.1689 at γ=0.30 (not excluded), 0.1973 at γ=0.35 (not excluded). r_caustic is SMALLER at lower γ → same (ρ,θ_c) maps proportionally closer to cusp vertex. gamma_band=None → no exclusion. Physics correct.
- `DeltoidCuspSourceAnglesTestCase`: γ=1.5 returns [0.0, 0.4317] rad — 2 distinct angles, off-axis present, D₂-folded correctly. γ=0.5 returns [] (no deltoid for positive parity). Physically distinct from _cusp_source_angles. Physics correct.
- `FarfieldTilesCuspExclusionTestCase`: Strict subset filtering, excluded tiles verified near cusp vertices. Backward compat preserved. Self-falsification via mock and d_exclude=0.0. Physics correct.

### Verdict: PASS
All 12 tests pass (0 failures). Numeric distances independently verified. No physics concerns. Heavy full-sampling validation deferred to operator out-of-band.
