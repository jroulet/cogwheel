---
date: 2026-08-20
section: Backlog
---

- **Diffractive-certificate calibration extended into the deep interior
  (INS-3-001); smoke re-baked; PROVISIONAL coefficients pasted**
  `[→ spec]` — the stranded fence-build revision (0d35d4f) left the
  two-sided near-fold fence in place but the calibration grid still
  started at `r = 0.3`, so NO bake sampled the deep interior
  (gamma <= 0.3 needs `r < 0.22`); the un-calibrated fit over-served
  there (measured: gamma=0.2/rho=0.3 -> fit 60 vs true 34, 1.77x;
  gamma=0.3/rho=0.3 -> 2.93x; gamma=0.3/rho=0.5 -> 2.72x;
  gamma=0.5/rho=0.3 @ cusp -> 2.54x) and `min(w_fit,
  _DIFFRACTIVE_FIT_CEILING)` clipped it to 60, quietly re-serving the
  interior where the series is NOT honest. This build completes the
  INS-3-001 owner ruling: CALIBRATE the interior, don't decline it and
  don't clip it.

  WHAT SHIPPED (code): the calibration grid (`_unfenced_grid_points` in
  `scripts/fit_diffractive_certificate.py`) now reaches `r = 0.1` on
  both scales — full `linspace(0.1, 1.3, 7)`, smoke interior-inclusive
  (deep-interior cells gamma in {0.2, 0.3} x r in {0.1, 0.2}, one
  interior anchor gamma=0.5/r=0.3, one near-exterior caustic-feature
  anchor gamma=0.5/r=0.9, one smooth-exterior anchor gamma=0.2/r=0.9).
  Fresh provisional coefficients pasted into `_diffractive.py`:
  de-rate `_DIFFRACTIVE_FIT_DERATE = 0.85` (the clamp, down from
  0.844967), new poly/harmonic/caustic coefficients, provenance SHA
  362c58e (526.9 s). Smoke margins now: grid 178/178 conservative and
  tight, off-grid 44/44 both, excluded-shell 49/227 grid rows (21.6%).
  `w_low_fit`'s docstring rewrites the conservativeness contract: the
  de-rate is the SOLE margin (it alone guarantees no over-serve on the
  calibration grid and its held-out off-grid midpoints; extrapolated
  off-grid points MAY over-serve), and the `min(., CEILING)` clip is a
  hard oracle-domain cap (`W_CEILING_SCHWINGER`, no oracle above 60), a
  no-op wherever the fit is calibrated. Corner witness re-pinned
  CORNER_R 1.05 -> 1.1 (the full-branch radii change dropped r=1.05);
  ratio prose re-baked (~1.01x). Honest-ceiling range updated ~4-34 ->
  ~4-41 in both code and tests.

  REMAINING (driver, post-build): the smoke coefficients are PROVISIONAL
  — the driver MUST re-bake at `--scale full` and paste the emission
  block verbatim. The gamma = 0.5 deep interior (`r ~ 0.11-0.28`) is
  EXTRAPOLATION from the smoke anchors at r=0.3/0.9 (the full grid adds
  gamma=0.5 at r=0.1) and its residual ~1.12x cusp-direction over-serve
  is why `TestWLlowFitDeepInteriorHonestServe` stays RED BY DESIGN
  (`_BRUTE_ACCURACY_REASON`) until the full bake lands — the exact-heavy
  honest-serve suite flips green with ZERO test edits then.
  `_CALIBRATED_GAMMA` constant retired; the fast-tier
  `TestWLlowFitDeepInteriorServedByFit` structural pin widened to all of
  `_DEEP_INTERIOR_GAMMAS` (0.2, 0.3, 0.5) with the calibration qualified
  by bake in the class docstring.

  SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph (Rung P) was corrected to
  the shipped fitted-surface mechanism (see
  `2026-08-20_diffractive_certificate_interior` spec-changelog
  fragment); the next build — the low-w near-fold analytic serve
  (todo.d stem `lensing_low_w_near_fold_serve`, which this fence build
  was the prerequisite for) — was completed 2026-08-20 by the
  chart-serve build, whose completion record is
  `[[2026-08-20_low_w_near_fold_chart_serve]]`.
