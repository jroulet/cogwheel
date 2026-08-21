## low_w chart cusp-fallback REVIEW (2026-08-21, OpenCode professor pass)

Ran test_lensing_low_w_diffractive_chart.py (47 passed, 62 s) + the cusp-serving
regression files test_lensing_airy_fold.py + test_lensing_fast_path.py
(150 passed/11 skip/2 xfail, 86 s). No diagnostic plots are emitted by the fast
tier — plot-based diagnostics live in the operator's bake/margin report.

VERIFIED physics:
- Non-vanishing cusp witness is (gamma'=0.8, rho=1.2, theta=0.2), b3=1.42e-15
  (measured, not the brief's rho=2.0). Pearcey |F_ref| min/max ~ 0.25, band
  spread ~3.9x << 1e3. Airy form correctly None there (b3->0).
- Decline/self-falsification witness is FAR-EXTERIOR rho=2.0 (NOT the brief's
  interior rho in [0.6,1)): its Pearcey `uniform = cluster_sum * P/P_asymp`
  collapses because `_matches_stationary` (tol = _CALIBRATION_TOL*spread + 1.0)
  stops matching the single exterior image once scaled_delay = w*(tau-tau_c)
  grows past the finite stationary values at w ~ 8 -> n_match 0 -> cluster_sum 0
  -> uniform == 0. |P| itself stays O(1) (|P/Pasym| -> 1.0); the collapse is
  cluster-classification, NOT a Pearcey P=0 zero.
- KEY MEASURED FACT: there are NO interior cusp cells (rho<1 with b3->0) at
  gamma'=0.8 — the full theta sweep at rho=0.5/0.7/0.9 shows b3 ~ 2 (all FOLD).
  b3->0 happens only on/outside the caustic (rho>=1). So the `fold_cusp_reference`
  docstring claim "interior cusp cells can hit P ~ 0" describes a case that never
  fires here (interior cells use the primary Airy path). Minor doc inaccuracy,
  not a physics error; the guard is correct regardless of collapse mechanism.
- Guard floor _NON_VANISHING_MIN_RATIO=1e-3 declines the far-exterior collapse
  to None -> exact engine (never NaN). Teeth proven (self-falsification green).
- Node-exact cusp re-modulation: Pearcey F_ref convention consistent between
  train (fold_cusp_reference) and serve (likelihood.py:2057 rebuilds the same
  F_ref) to NODE_EXACT_TOL=1e-10. Fold/cusp |F_ref| continuity ~3.12x across the
  theta 0.2(cusp)->0.3(fold) handoff (tol 5.0), both forms genuinely visited.
- cusp_amplification refactor (geometry-> _cusp_uniform_geometry, controls->
  _cusp_controls) is DRY behavior-preserving; cusp-serving ladder tests still
  green, no regression.

BRIEF DRIFT (flag for future briefs): spec 1 said non-vanishing witness rho=2.0
(actually declined) and spec 4 said interior decline cell (doesn't exist). The
implementation is the MEASURED-corrected version; brief was written on the
pre-measurement expectation. Verdict PASS.
