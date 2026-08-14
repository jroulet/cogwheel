---
date: 2026-08-13
section: Lensing
---

- **RESOLVED: re-gated the census `ppgo_fold` mirror to the interior rung**
  (build `fold_exterior_ghost` WP-3, commit 1805bfd; see
  [[2026-08-13_fold_exterior_ghost]]). `surrogate_census.characterize_sample`
  no longer classifies with the retired `xi_min`-based fold gate; it now
  gates on `image_count == 4` and
  `geometry.ppgo_error_estimate(...) * _PPGO_INTERIOR_SAFETY <=
  CERTIFICATION_BAR`, single-sourced from `likelihood`/`geometry` rather
  than re-typed — matching the interior rung `_surrogate_coefficients`
  actually serves (build `ppgo_interior_certificate`,
  [[2026-08-13_ppgo_interior_certificate]]). Inspector INS-2-001 accepted
  deferral is closed; the classification skew (dry-run census counts vs
  what production serves) is gone.
