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
  near-caustic corner left as a NAMED refusal / exact fallback in THIS
  build — all fold/cusp uniform-asymptotics (Airy/Pearcey) serving
  belongs to the subsequent cusp fast-serving build
  ([[likelihood_cusp-fast-serving]]; scope fence set 2026-07-20 so the
  two builds do not collide), with the v-plane evaluator demoted to
  not-needed-unless-the-uniform-patch-falls-short.
  Interim state to be dissolved (Build 7a, deliberate): legacy
  bit-frozen wherever it certifies, Schwinger only for saddle parity
  and as the strong-shear refusal fallback (w <= 60, gamma' > 0).
  Measured constraints driving the ordering: warm Schwinger
  30-125 ms/point vs the 9.8 ms full lnlike (surrogate is
  load-bearing); rescued strong-shear SACR-C envelope shows a 0.94-nat
  interpolation discrepancy vs brute force (rescued-node envelope
  accuracy gate is a Build 7b precondition).

  OWNER DESIGN SEED (2026-07-20, for the Build 8b full-box tiling):
  fit the surrogate in CAUSTIC-ADAPTED coordinates so the non-smooth
  locus sits at a fixed place — (gamma, eta, theta, log w) with
  (eta, theta) from `nearest_caustic_point` puts every caustic at the
  eta = 0 plane for all gamma; fitting in u = sqrt(eta) (the known
  fold exponent) makes the interpolant smooth through the transition.
  One near-caustic tube chart + far-field raw charts beats tiling
  axis-aligned boxes around a curved surface. Cusp neighborhoods need
  their own patches (2/3-power scaling; Airy -> Pearcey) — this is
  where the surrogate and the fold/cusp uniform-asymptotics (Airy)
  programs CONVERGE: the uniform form IS the known structure at the
  fixed transition. Query-time remapping is nearly free
  (geometry_partition already returns caustic_distance).
