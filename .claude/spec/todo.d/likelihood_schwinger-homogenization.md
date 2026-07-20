---
section: Backlog
---
- [ ] **Homogenize the engine on the Schwinger evaluator (Build 8
  program)** `[→ spec]` — owner directive (2026-07-19): "make it
  homogeneous, use the surrogate to speed it up, and do what is needed
  to make the w range higher." Target architecture: Schwinger as THE
  single wave evaluator on both parities; the legacy
  hypergeometric/operator-series path demoted to ORACLE duty (byte-level
  regression gate on the overlap domain — owner-set gate); the
  surrogate (see `likelihood_envelope-surrogate.md`) as the production
  speed layer trained on Schwinger; resolved high-w served by geometric
  optics + per-image relative binning; the narrow unresolved-high-w
  near-caustic corner served by the fold/cusp uniform (Airy) patch
  (v-plane evaluator demoted to not-needed-unless-Airy-falls-short).
  Interim state to be dissolved (Build 7a, deliberate): legacy
  bit-frozen wherever it certifies, Schwinger only for saddle parity
  and as the strong-shear refusal fallback (w <= 60, gamma' > 0).
  Measured constraints driving the ordering: warm Schwinger
  30-125 ms/point vs the 9.8 ms full lnlike (surrogate is
  load-bearing); rescued strong-shear SACR-C envelope shows a 0.94-nat
  interpolation discrepancy vs brute force (rescued-node envelope
  accuracy gate is a Build 7b precondition).
