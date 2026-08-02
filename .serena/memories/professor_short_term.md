# Professor Short-Term Observations

## Born Residual Wiring review (2026-08-02)

- **test_lensing_born_residual_wiring.py**: 34/34 pass in 4.43s.
  All 5 spec test classes green plus self-falsification.
- **test_lensing_born.py**: 53/53 pass in 26s — no regression.
- **Part 0 mechanical**: 13/13 pass — no regression.

### Spec verification:

1. **No-chart byte-identity**: born_residual_chart=None returns None
   from _surrogate_coefficients for multiple Born-annulus configs.
   Proved non-vacuous: same config WITH a chart returns non-None. ✓

2. **Mock chart serve path**: Carrier+residual identity verified at
   1e-13 relative tolerance. Mock chart interpolation at grid point
   (gamma=0.5, rho=3.0 — both on grid; residual constant in w) is
   exact to 2e-16 relative (machine precision). The reconstruction
   via reconstruct_farfield (ppgo subtraction, t_min demodulation,
   far-field inversion) is algebraically correct. Diagnostic plot
   saved to output/born_residual_wiring_identity.png. ✓

3. **Out-of-box fallthrough**: Three sub-cases (rho>5.0, 1.0<rho<1.5,
   rho<1.0) all return None. Also tested gamma outside chart range.
   Physics: rho<=1.0 guard fires BEFORE chart.covers for interior
   configs (4-image topology has qualitatively different physics). ✓

4. **Kappa/beta guard precedence**: Guards at lines ~1559/1572 fire
   before the fact-4 Born slot at line ~1667. Multiple kappa and beta
   values tested, all refused. Control: kappa=beta=0 reaches Born path.
   Physics: the Born chart is a kappa=0, beta=0 surface by construction
   (mass-sheet identity absorbs kappa into apparent distance; beta
   absorbed by eigenframe rotation). ✓

5. **BornResidualChart.covers boundary cases**: All 7 spec test points
   pass (interior, edges inclusive, just-outside rejected). Additionally
   tested all 4 corners, all 4 edges, and machine-epsilon boundaries.
   Implementation: simple `<=` comparisons — correct for a training
   grid with inclusive endpoints. ✓

### Physics assessment:

- The Born residual decomposition `F_total = carrier + R(w; gamma, rho)`
  is physically sound. The carrier is the closed-form Born approximation
  (resolved geometric-optics images with 1/w asymptotic corrections),
  and the residual captures the diffraction correction. The chart
  covers the Born annulus (rho > 1, exterior to caustic) where the
  source sees exactly 2 images — appropriate for the weak-deflection
  regime.

- The kappa/beta guards are physics-mandatory: the chart was trained on
  the reduced parameter space (kappa=0, beta=0) after folding via the
  mass-sheet identity and eigenframe rotation. Serving a kappa!=0 config
  with a kappa=0 chart would give finite-but-wrong answers.

- The rho > 1.0 guard is correct: interior to the caustic (rho < 1)
  the source has 4 images with qualitatively different topology; the
  Born carrier formula does not apply there.

- Heavy full-sampling validation is operator-deferred.
