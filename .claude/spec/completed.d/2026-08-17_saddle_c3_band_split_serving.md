---
date: 2026-08-17
section: Backlog
---

- **c3 band-split serving SHIPPED — the dead rung serves 14.09% of the
  prior and total chart demand fell 72.25% → 53.30%** `[→ spec]` —
  build `c3_band_split_zero_refusal` (commit 6958f0c; census mirror
  re-gate follow-up commit). The 672-point calibration is live via
  per-draw band-split: analytic zero-envelope above the closed-form
  split point w_split = w_ref·(S·est/bar)^(1/3) (exact cube-root
  inversion of the certificate's w⁻³ law; est-None stays the whole-draw
  coalescence refusal), exact engine stitched below via the shared
  `_band_split_mask` + `_engine_envelope_below_split` (Born refactored
  onto the same helper byte-identically). Null-split identities
  byte-exact at both boundaries; in-band accuracy overlap vs the DD
  engine < 1e-3 (certificate currency; true remainder ≤ 5e-5).
  MEASURED (re-gated 10k demand census, seed 0,
  `.claude/handoff/demand_census_post_c3_regate_10k.json`): saddle_c3
  0.32% → 14.09%; the sibling ceiling rung (same build) took
  ppgo_above_ceiling 0.00% → 15.87%; wave_refused 12.03% → 2.13%
  (residual tracked in [[lensing_wave_refused_to_zero]]);
  engine_residual 72.25% → 53.30%. Census route taxonomy: saddle_c3
  counts every c3-served draw (whole-band and split), per-draw w_split
  recorded for the tiling design's below-split demand sizing.
