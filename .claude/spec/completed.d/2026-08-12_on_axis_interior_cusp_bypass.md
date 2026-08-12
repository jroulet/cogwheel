---
date: 2026-08-12
section: Backlog
---
## On-axis interior cusp serving gap — closed (fold detection + degenerate bypass)

Implemented in commit `b80d1d6` (`feat(lensing): serve on-axis interior cusp
sources via the Pearcey arm`).

- **`_airy_fold._merging_fold_pair` detects 3-image cusp cluster**: added
  `_CUSP_TIE_EPS = 1e-12`; when the best saddle delay has `tie_count >= 2`
  (two saddles at the same delay, the degenerate symmetric pair on the cusp
  symmetry axis), returns None instead of certifying a 2-image fold form against
  a 3-image reality. This routes the node to the cusp arm as the next rung.

- **`cusp_amplification` interior degenerate bypass**: the calibration bypass
  extends to interior degenerate clusters — an on-axis interior source
  (`rho < 1`) whose first-order control projection degenerates to 1 stationary
  point while `len(images) > 2` and the fold arm has declined. The
  self-calibrating ratio `P/P_asymp` is accurate there (measured rel-err
  1.5e-3 at w=100 vs exact). Exterior sources (`rho > 1`) keep the certificate
  enforced (measured 52% error for exterior on-axis source — bypass does NOT
  apply there).

- **`operator.py` mpmath band**: the uniform-asymptotic rung is now offered
  BEFORE the exact engine in the mpmath band (60 < w <= 150), matching the
  declared SERVING LADDER order. On-axis sources in this band are now served
  by the cusp arm (fast), closing the w=150 exact-refusal gap.

ACCEPTANCE met (second OR condition): the fold arm correctly detects the
3-image cluster and declines to the cusp arm; the cusp arm serves the interior
degenerate cluster via bypass; all 221 affected tests pass.
