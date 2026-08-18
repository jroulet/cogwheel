---
date: 2026-08-18
bump: patch
---

Post-commit doc sync (733b7ef): the serve-route demand census paragraph
still described `0.32% saddle_c3` as unreachable and pointed at
`todo.d/lensing_saddle_c3_band_split_serving`, a fragment already
retired to `completed.d/2026-08-17_saddle_c3_band_split_serving.md`
(dangling plain-text reference, not `[[...]]`-linked so invisible to
the wiki-link checker). Corrected in place: the sentence now states the
fix (build `c3_band_split_zero_refusal`) and the re-gated numbers
(saddle_c3 0.32% -> 14.09%, ppgo_above_ceiling 0.00% -> 15.87%,
wave_refused 12.03% -> 2.13%, engine_residual 72.25% -> 53.30%, since
further reduced to 24.10% by the low-w diffractive rungs). No SPEC
content changed beyond this correction.
