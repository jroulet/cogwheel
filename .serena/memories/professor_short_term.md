# Professor Short-Term Observations

## Log-reach gamma axis review (2026-08-02)

- **test_lensing_log_reach_gamma.py**: 23/23 pass (17 structural/self-falsification
  in 3.0s without engine; 6 engine-backed in 18.9s with COGWHEEL_TRAIN_TIER=1).
  All three spec classes green.

### Spec 2 (structural) verification:

1. **Length**: Both positive and saddle arrays return exactly 7 elements. ✓
2. **Strict ascending**: All diffs > 0 for both parities. ✓
3. **Endpoint pinning**: `arr[0] == lo`, `arr[-1] == hi` to 1e-14. ✓
4. **Log-reach round-trip**: max error 1.17e-5 (near-wall), well within 5e-4 tol. ✓
   Interior positive: ~1.4e-6. Saddle (1.02,1.40): also within tolerance.
5. **Clustering direction**:
   - Wall band (0.90,0.98): last_gap=0.006548 < first_gap=0.021991 ✓
   - Saddle (1.02,1.40): first_gap=0.007447 < last_gap=0.304815 ✓
6. **Log-reach uniformity confirmed**: diffs in log-reach space are uniform to
   ~5e-5 relative (0.14831±0.00001 for wall band).
7. **Self-falsification**: all 5 mutation detectors fire correctly.

### Spec 1 (comparative accuracy) verification:

- **Coverage**: 30/30 held-out points served by both charts.
- **Max uniform eps**: 0.003518 (at gamma=0.97282 — the expected near-wall spike).
- **Max log-reach eps**: 0.000831 (at gamma=0.90756 — away from wall, low absolute).
- **Improvement ratio**: 0.2364 (need < 0.7) — 76% improvement, far exceeding 30%.
- **Absolute bar**: 0.000831 << 0.05 (tube_eps_max). ✓
- **Physics interpretation**: The uniform grid's worst error peaks at gamma≈0.97
  where caustic reach varies steeply (diverges as gamma→1); log-reach concentrates
  nodes there (last gap 0.0065 vs first gap 0.022), equalizing the interpolation
  error across the band.
- Diagnostic plot generated: `cogwheel/tests/output/log_reach_gamma_comparative_eps.png`.

### Spec 3 (regression guard) verification:

- **Coverage**: 20/20 held-out points served.
- **Max eps**: 0.001160 (at gamma=0.36255, near band edge).
- **Bar**: 5e-3. PASS (0.00116 < 0.005). ✓
- **Interpretation**: Interior band (0.35–0.65) is far from the wall, so caustic
  reach varies smoothly; log-reach placement is nearly uniform here (gamma diffs
  range 0.044–0.053, close to uniform 0.050). No degradation from the new placement.

### Overall assessment:

The `_log_reach_gamma_axis` implementation is correct and achieves its design goal.
The 200-point fine sweep + np.interp inversion achieves adequate round-trip accuracy
(max 1.2e-5 on the steepest band). The clustering direction follows the physics:
caustic reach diverges at gamma=1, so nodes crowd the wall side for both parities.
The comparative gain (76% improvement in max eps) is physically expected — the
nonlinear coordinate change compensates the steep caustic-reach gradient.

- Heavy full-sampling validation is operator-deferred.
- Part 0 mechanical tests (13/13) still passing — no regression.
