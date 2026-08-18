# Build: Born far-field completion, build 2 — the two-image GO carrier

## Mission

Owner directive 2026-08-17, both parities; build-2 redirect 2026-08-18.
Build 1 proved the single-image Born series dead at ANY order: the
omitted second image's beat, sqrt(|mu2|/|mu1|), floors the remainder
w-independently — the far-field carrier must contain BOTH images (the
beat-free lesson one octave up). So the far field is served by the
TWO-IMAGE GO carrier — the SAME resolved sum `select_branch`'s
geometric branch serves (bind it, never re-derive) — admitted by a
certificate calibrated on the measured ~1/w law. THREE CLOSURES:
(1) the GO-carrier far-field rung, both parities, saddle keyed on
rho_lobe (`_lobe_serves` fold), targeting what the geometric gate
DECLINES in the far field; (2) residual-chart REPRESENTATION settled
now — residual against the SAME carrier, ONE definition for rung and
chart (DRY; the optimal-representation clause), training stays in the
campaign; (3) Born chart floor band-split (direction (a), mechanical)
reviving the ~6% box-covered population. Census re-gated IN-BUILD;
the honest 44.62% ledger is the baseline.

## Facts (measured at 1fe35a6 unless noted)

1. Honest ledger (`demand_census_post_born_10k.json`, 10k seed 0,
   20-1024 Hz, build-1 mirror shipped 3f3c57d): ppgo_above_ceiling
   15.87%, saddle_c3 14.09%, born_analytic + born_carrier_only 0.00%,
   diffractive 8.12% analytic + 15.17% engine-hosted, engine_residual
   44.62%, wave_refused 2.13%.
2. Build-1 verdict: the carrier-only certificate
   |delta| = hypot(a0, w*b1/2)/q2r NEVER admits under the prior — est
   min 0.0195 / median 0.547 vs the 5e-5 bar over all 1461 Born-gate
   draws. The honest `trained_band_escape` mirror exposed born_analytic
   14.61% -> 0%: box-covered draws never fit the trained log-w FLOOR
   (prior band width 3.936 vs trained 2.485) — direction (a)'s
   population, engine-served today.
3. Corrected-carrier probe (`corrected_carrier_remainder_probe.md`,
   540 engine calls, harness byte-identical to `born_amplification`,
   recorded b830b0e): admission at the 1e-3 bar EMPTY (0/500 nodes,
   both parities); census beat amplitude sqrt(|mu2|/|mu1|) min/median
   = 2.87e-2/1.37e-1 (29x/137x over bar); a0 breaks the exact w->0
   limit. No polynomial in 1/q2r represents a second stationary point.
4. Two-image GO carrier, same probe: F_2GO = sum_i sqrt|mu_i| *
   exp(i*w*tau_i - i*pi*n_i/2); rel error vs the engine decays ~1/w,
   crossing 1e-3 at w ~ 15-60 and reaching 2e-4-2e-3 at w=60 on
   census-realistic geometries (q2r 8-90, both parities). This probe
   table is the certificate's calibration set.
5. Saddle far-field demand (lobe-gauge probe, 9f331dd): 1267 draws =
   12.67% of prior, 2-image lobe-EXTERIOR, rho_lobe p50/max = 5.2/20.2
   vs rho_outer 1.25-2.40; w_hi p50/p99/max = 4.61/30.7/38.0, w_lo >=
   0.026, no w_split; invisible to the origin-gauge `caustic_rho` gate.
6. The geometric branch already serves resolved draws:
   `operator.select_branch` returns 'geometric' iff w*delta_min >= 4.0
   AND cancellation_exp > 48 AND eta >= 0.3, serving
   `geometric_amplification` — the same two-image sum. This rung
   targets ONLY what that gate declines in the far field (eta/L legs,
   the low-w unresolved band).
7. Precedents: `_band_split_mask` / `_engine_envelope_below_split`
   (`likelihood.py`); `serve_route_census.classify_draw`;
   `BornResidualChart` (reconstruct = carrier + R); lobe frame
   `_to_lobe_fixed` / `_lobe_cusp_axis_map`.

## Scope

IN: the Professor's plan-time certificate on the ~1/w law (per-draw
admission floor w_go(draw) by inversion, the c3 pattern) with a
MANDATORY reachability check BEFORE any WP is written: quote from
probe + census which (w, geometry) cells pass the bar under the prior,
and enumerate which far-field population is NOT already
geometric-served (Fact 6) and what the carrier + certificate adds vs
that gate — if the gap is better closed by certifying/relaxing the
`select_branch` legs for the far field, the plan says so instead (no
double-building); if the admitted region is small, weight shifts to
the residual charts, with numbers. The serving rung in
`likelihood.py`, BOTH parities, saddle admission keyed on rho_lobe;
the carrier BOUND to `geometric_amplification`, never re-derived.
Residual-chart representation only (training in the campaign):
residual against the SAME carrier, lobe-gauge cusp-adapted coordinates
(saddle), existing coordinates (astroid annulus). Born chart floor
band-split: trained chart serves [trained floor, w_trust],
diffractive/engine below (nested `_band_split_mask`). Census mirror
for every new route (taxonomy law: GO-carrier-served = analytic ONLY
with a real certificate; residual-chart demand stays engine side until
charts exist) + the 10k engine-free re-run IN-BUILD; fast synthetic
tests.
OUT: the low-w diffractive rungs and the beat-free tube (untouched);
chart TRAINING and the tiling design (campaign steps); the 2b
arm-extension (wave_refused 2.13%); the corrected-carrier serve
(direction (b), superseded by Fact 3); widening any existing
certificate constant or gate leg without its own derivation.

## Acceptance

- Census re-run in-build (same config, 10k seed 0): far-field engine
  hosting shrinks to the certificate-refused + chart-pending
  population ONLY — quote new-route fractions, the served share of the
  12.67% saddle far-field, and the revived floor-band share.
- GO-carrier serving accuracy vs the exact engine <= the certificate
  bar EVERYWHERE admitted (cheap DD-band oracle, w <= 60 overlap);
  tolerance from the certificate, never tuned. Escalate-not-iterate.
- Byte-identity for every population the new rungs do not touch —
  astroid chart box, low-w rungs, geometric gate (null-split pins).
- Full fast suite green.
- Parsimony: one canonical pin per invariant; re-point existing pins;
  report added-vs-retired counts.

## Constraints

Branch claude-dev only. Certificate-gated serving: the admission bound
comes from the derivation calibrated on Fact 4's probe table — never a
per-draw tuned constant. Escalate-not-iterate: if the carrier misses
the engine ANYWHERE the certificate admits, STOP — that falsifies the
certificate, not the plumbing; never widen bars. If acceptance is met,
closes `todo.d/lensing_born_farfield_completion.md` (completed.d
record); `[→ spec]` — spec_changelog.d fragment with `bump:`; render
fragments. In-build tests fast/synthetic; bulk sweeps stay post-build.
