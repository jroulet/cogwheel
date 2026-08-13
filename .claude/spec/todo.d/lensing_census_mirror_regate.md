---
section: Backlog
---

- **RE-GATE THE CENSUS `ppgo_fold` MIRROR TO THE NEW INTERIOR RUNG** `[→ spec]`
  — `surrogate_census.characterize_sample` (~L468) still classifies with the
  OLD xi-based fold gate and claims to "Mirror _surrogate_coefficients", but
  the likelihood interior rung was re-gated (build ppgo_interior_certificate,
  2026-08-13) to the exact 4-real-image predicate + c3 certificate. The census
  dry-run counts now skew vs what likelihood serves. Inspector INS-2-001,
  escalated and ACCEPTED as deferred. Fix: gate on `image_count == 4` and
  `geometry.ppgo_error_estimate(...) * _PPGO_INTERIOR_SAFETY <=
  CERTIFICATION_BAR`, update the mirroring comment, and add the one canonical
  pin in the census suite. Classification skew only — no wrong serving.
