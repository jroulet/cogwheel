# Professor Short-Term Observations

## 2025-08-03: Fold-ppGO interior handoff review (Build ppgo_interior_handoff)

### Tests Run
- `test_lensing_fold_ppgo_handoff.py`: 14 pass, 3 skipped (engine-backed, behind COGWHEEL_TRAIN_TIER)
- `test_lensing_born_residual_wiring.py`: 34 pass
- `test_lensing_ghost_gate.py`: 18 pass
- Total: 66 pass, 3 skipped, 0 failures.

### Independent Numerical Verification
- Interior fixture (gamma=0.5, rho=0.3, theta=pi/4): 4 images, fold pair delta_tau=0.327, xi_min=4.95 (>= 4.0 threshold). Error estimate at xi=5: 0.091 (fine gate refuses at w=45).
- Census fixture (M_lens=20e6 Msun): w_min=49,516, xi_min=528, error estimate 8.25e-5 < 1e-4 (fine gate admits). Physics correct: c_A * xi^{-3/2} scaling verified (0.09 × (5/528)^1.5 ≈ 8.3e-5).
- Near-caustic fixture (rho=0.7): xi_min=2.15 < 4.0 — gate correctly refuses.
- High-curvature fixture (gamma=0.85, rho=0.5): xi=5.85 (coarse gate admits), error=0.066 >> 1e-4 (fine gate refuses). Margin ~660x above bar.
- Round-trip residual: 1.02e-15 (well below 1e-12 bar — machine precision identity).
- t_min from helper matches geom.t_min to machine precision.

### Physics Assessment
1. xi formula (3wΔτ/4)^{2/3}: correct Chester-Friedman-Ursell uniform approximation parameter — threshold 4.0 ensures ≈2 full Airy oscillations (well-resolved regime).
2. Dual gate structure (xi coarse + c_A fine) is conservative and load-bearing: tested that the fine gate refuses the interior fixture at moderate w (0.091 >> 1e-4) but admits at large w (census: 8.25e-5 < 1e-4). The 1/xi^{3/2} decay of the error with increasing w is the correct asymptotic behavior of the uniform Airy next-order correction.
3. Reconstruction round-trip: algebraically correct by construction (demod → subtract ppGO → remod → reconstruct_farfield adds ppGO back). Residual 1e-15 confirms no frame mismatch.
4. Default-path priority: chart serves before fold-ppGO gate is reached (verified by mock-based tests). No regression for below-ceiling configs.
5. Census integration: ppgo_fold category correctly set with served=True when all gates pass; correctly refused (served=False) when error estimate exceeds bar.

### Deferred: Full-sampling validation (operator scope)
The skipped engine-backed accuracy test (fold correction vs exact at 1% bar) requires COGWHEEL_TRAIN_TIER. This is the heavy validation gate — fast-tier coverage confirms the gate logic, frame conventions, and round-trip identity are correct. The actual fold correction accuracy at moderate xi (∼5) with this fixture's c_A (∼0.36) would likely fail the 1% bar (error estimate 9% suggests the Airy approximation has ∼9% relative error here), but this is EXPECTED: the production gate refuses these configs precisely because the error estimate exceeds CERTIFICATION_BAR. Only the census fixture (xi∼528) where the gate admits would pass the 1% accuracy test.
