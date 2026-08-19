---
date: 2026-08-18
bump: patch
---
Recorded the trained-floor band split in the `born_residual_chart` contract
description (serving behavior only; no field, producer, or consumer change).

- Previously a dense w band whose low edge fell below the chart's trained
  `log_w_grid` floor failed `covers()` outright and sent the WHOLE draw to the
  exact engine, discarding coverage the chart genuinely held over the upper
  part of the band.
- The `_born_residual_analytic` intercept now treats that low-edge escape as a
  second-tier band split instead of a refusal: the chart serves its trained
  sub-band (Born carrier + residual) and `_engine_envelope_below_split` hosts
  only the untrained remainder below the trained floor. This nests inside the
  existing certified-ppGO split (Born carrier + residual at or below `w_trust`,
  bare point-mass ppGO above), so a single band can now carry three tiers.
- Off-axis `gamma`/`rho` box misses still refuse rather than cubic-extrapolate;
  the split applies to the w axis only, where the untrained region is bounded
  and engine-hostable.
- The null-split path is byte-exact against the pre-split serving values, so
  draws entirely inside the trained log-w range are unchanged.
