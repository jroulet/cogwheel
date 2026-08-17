# Build: c3 band-split serving + wave_refused to zero

## Mission

Two dead serving paths, one shared fix: band-split serving on the Born
w_trust-split architecture. (1) The saddle c3 rung is DEAD under the
physical prior as a whole-band intercept — its calibrated admitting
floors are unreachable from the physical 20 Hz band floors (Fact 6), so
the census's saddle_c3 route sits at 0.32%. Band-split serving revives
the 672-point calibration: serve the analytic channel sum ABOVE the
per-draw certificate floor, engine/chart BELOW — exactly the
w_trust-split the Born intercept already implements. The certificate
itself yields the per-draw split point: the smallest w where
`S * ppgo_error_estimate(w) <= bar`; the estimate has w**-3 shape, so
closed form to invert or bisect, cheap. (2) Simultaneously, wave_refused
MUST GO TO ZERO (owner directive 2026-08-17): 12.03% of prior mass is
deterministic refusal (above-150 nodes both arms decline ->
SchwingerCertificationError, lnL = -inf), yet above w = 150 the physics
gets EASIER. Fix (a): `_ppgo_above_ceiling` gates on the BAND FLOOR, so
a draw whose above-150 nodes are individually resolved fails the
whole-band gate — make the serve PER-NODE (resolved above-ceiling nodes
analytic, low band engine/chart: band-split, same architecture). Fix
(b): the residual armless population (arms declining in the high-w
near-caustic corner) is covered by consolidating the arm-extension work
of [[lensing_ppgo_extrapolation_beyond_engine_reach]] (refined form)
and the saddle-envelope-negligible route — do not duplicate it.

## Facts (measured; SHA 6a3f43c unless noted)

1. c3 gate (`cogwheel/lensing/likelihood.py`,
   `_saddle_farfield_analytic_serves`): admits iff
   `est = ppgo_error_estimate(images, source, matrix, w_lo)` is not None
   AND `min_sep >= _SADDLE_FARFIELD_MIN_IMAGE_SEP` (= 0.05) AND
   `_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR`
   (S = 20.0, bar = 1e-3), evaluated at the band FLOOR w_lo — whole-band
   admit-or-refuse. `None` est (divergent mu/c3, merging pair) is the
   primary coalescence discriminator and must stay a refusal.
2. Above-ceiling intercept (`likelihood.py`, `_ppgo_above_ceiling`):
   fires only when `dense_w.max() > W_CEILING_SCHWINGER_QD` (= 150);
   whole-band gate is `if w_lo * min_delta_tau < RHO_END: return None`
   (RHO_END = 4.0, operator.py). Physical w_lo <= 8.67 makes this nearly
   unsatisfiable: census route ppgo_above_ceiling = 0.00%. On None the
   caller falls to the exact engine -> SchwingerCertificationError.
3. Refusal sites: `_schwinger.f_schwinger` raises
   `SchwingerCertificationError` on `w > W_CEILING_SCHWINGER_QD` and on
   paired N/2N disagreement; `operator.py` wave wrappers offer
   `_uniform_arm_value` (fold Airy, then cusp Pearcey) first, then
   re-raise via the lowest-index refuser — ONE node kills the draw.
4. Born precedent to copy (`likelihood.py`, `_born_residual_analytic` +
   `_ppgo_band_split`): w_trust from the certified map (never a
   constant); ceiling `min(parity_wall, cell_ceiling)`; `band_split =
   w_trust is not None and w_lo < w_trust < w_hi`; `below_mask =
   dense_w <= w_trust`; envelope zeroed above (E_ff = 0 telescopes to
   the bare image-kernel sum, FARFIELD_KERNEL_SUM gauge); test-pinned
   byte-exact null-split identity: no split -> whole-band serve.
5. Census (.claude/handoff/demand_census_corrected_10k.json, 10k draws,
   seed 0, 20-1024 Hz, verified at 6a3f43c): wave_refused 1203/10000 =
   12.03%, saddle_c3 32 = 0.32%, born_analytic 15.40%, engine_residual
   72.25%. ALL wave_refused draws sit in the w_hi > 150 band; top cell
   wedge_interior gamma 0.55-0.9 at 4.29% (429 draws). Records carry
   per-node `node_route_kinds`; mixed served/refused lists confirm the
   per-node defect directly.
6. Admitting floors (2026-08-17 demand-census audit, fragment record):
   c3 admits at w_lo >= ~28 (rho 0.3) / ~20 (rho 1.5) / ~8.7 (rho 2.5);
   physical band floors w_lo = 2.476e-3 * M <= 8.67 (M <= 3500).
   Residual demand concentrates at w <= 60 (corrected census), so the c3
   split shrinks saddle-side table need to [w_lo, min(60, split)].

## Scope

IN: c3 band-split serving in the saddle far-field rung (per-draw split
point from the certificate; analytic channel sum above, engine/chart
below; Born-style below_mask + null-split identity); per-node
above-ceiling serving replacing `_ppgo_above_ceiling`'s whole-band w_lo
gate; consolidation of the arm-extension work the wave_refused fragment
names (refined ppgo-extrapolation + saddle-envelope-negligible routes)
so the armless high-w near-caustic corner serves; band-split machinery
factored once (DRY with the Born/surrogate split arithmetic, not a
third copy); fast synthetic tests.
OUT: tiling/training campaigns; the tube representation (separate
build); the deltoid redesign; any change to lobe/wedge serving;
raising/moving W_CEILING_SCHWINGER_QD or the certificate constants.

## Acceptance

- Re-run `scripts/serve_route_census.py` (same config: 10k, seed 0):
  wave_refused reads ZERO, or only a measure-zero named-refusal set,
  explicitly enumerated in the completion record.
- saddle_c3 route becomes LIVE: nonzero fraction, serving the
  above-floor bands (report the new fraction).
- Byte-exact null-split identity: a draw whose split point falls below
  its band floor serves identically to today (the Born precedent's pin).
- Full fast suite green.
- Values-not-paths pins, tolerance-based, with parsimony: one canonical
  pin per invariant; re-point existing pins rather than adding parallel
  ones; report added-vs-retired counts.

## Constraints

Branch claude-dev only. Closes BOTH
`todo.d/lensing_saddle_c3_band_split_serving.md` and
`todo.d/lensing_wave_refused_to_zero.md` with completed.d records; both
are `[→ spec]` — spec_changelog.d fragment with `bump:`; render
fragments after writing. No new measured constants without provenance
(SHA + how measured); split points come from the certificate, never
hardcoded. In-build tests fast/synthetic; census re-run and bulk sweeps
are driver post-build steps. Escalate on surprise rather than iterate —
in particular, if the analytic sum above the split misses the engine
reference anywhere the certificate admits, STOP: that falsifies the
certificate calibration, not the plumbing.
