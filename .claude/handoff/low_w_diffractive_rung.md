# Build: low-w diffractive analytic rung (serve the band bottom)

## Mission

Owner directive 2026-08-17: engine fall-through below the far-field
w-floor is UNACCEPTABLE — the diffractive bottom must serve
analytically, BOTH parities. As w -> 0, F -> 1 (unlensed limit); the
long-wavelength expansion of F(w) = (w/2pi i) INT e^{i w tau(x)} d^2x
gives corrections in powers of w with coefficients that are pure
geometry (moments of the Fermat delay) — equivalently the
single-scattering/Born expansion in the lens potential. The Professor
derives the exact form and its truncation certificate at plan time —
the load-bearing step. The rung serves w below a per-draw analytic
ceiling w_low(draw), admitted by its own certificate (leading OMITTED
term <= bar — the c3 pattern at the opposite band end), NO measured
constants. ONE RUNG, THREE CLOSURES: (1) kill the F070 fall-through at
the astroid far-exterior diffractive sliver; (2) cover as much of the
12.67% saddle far-field low-w demand as the certificate admits;
(3) shrink every c3/Born band-split's below-split engine hosting to
[w_low, w_split].

## Facts (measured at c751215 unless noted)

1. F070 serve-site refusal (`likelihood.py`,
   `_surrogate_coefficients`, commit 8dfb8ca): after chart admission,
   `if served and definition in _FARFIELD_KERNEL_FAMILY: w_floor =
   farfield_w_floor(geom.delays, geom.real_mask); if
   float(chart_w.min()) < w_floor: served = False` — correct refusal
   (the kernel-sum envelope diverges below the floor; it once served
   468x max|F| wrong), then falls through Born -> raw-ppGO interior ->
   exact engine.
2. The floor is physics: `channels.farfield_w_floor` = (RHO_END/2) /
   min_{a!=b real} |tau_a - tau_b|; `inf` when < 2 real images.
   `surrogate_training._farfield_region_w_floor` takes the region max
   at the inner admission edge; every exterior tile is clipped to
   [w_floor, w_trust]. Window i below the floor has a named label
   (`FARFIELD_DIFFRACTIVE`, "subtract nothing", F -> 1 as w -> 0) but
   NOTHING serves it — no tiles, no analytic rung.
3. Low-w structure in the codebase: `_born.py` — the leading Born
   correction carries w in the NUMERATOR, vanishing as w -> 0 (lead
   term w-independent); `_hyp1f1.point_mass_g_derivatives` is the exact
   shear-free closed form G_PM(w,s) = C(w)*1F1(1 - iw/2; 1; -iws/2), an
   independent w -> 0 cross-check; `geometry._gaussian_moment_table`
   (Isserlis) is the only moment-table precedent. NO
   Fermat-delay-moment machinery exists — the Professor derives fresh.
4. Demand (regate census `demand_census_post_c3_regate_10k.json`, 10k
   seed 0, 20-1024 Hz, commit b097ce1, re-verified at c751215): routes
   ppgo_above_ceiling 15.87%, saddle_c3 14.09%, born_analytic 14.61%,
   engine_residual 53.30%, wave_refused 2.13%. Band bottoms: w_lo
   p1/p50/p99 = 0.026/0.47/8.18; 6242/10000 draws have w_lo < 1. All
   1409 saddle_c3 splits: w_split p1/p50/p99/max = 1.79/14.5/40.4/51.6
   — every below-split node is engine-hosted today.
5. Saddle far-field (lobe-gauge probe, 9f331dd,
   `saddle_residual_lobe_gauge_probe.md`): 1267 draws = 12.67% of prior
   are 2-image lobe-exterior far-field, ALL in the w_hi <= 60 band,
   w_hi p50/p99/max = 4.61/30.7/38.0, w_lo down to 0.026, no w_split
   (c3 declined end-to-end).
6. Oracle reachability: the entire below-floor band sits deep inside
   the engine-reachable range (DD band w <= 60 is the cheap oracle;
   engine to 150) — full overlap window for value acceptance, unlike
   c3's above-ceiling extrapolation.
7. Mirror-fidelity lesson (c3 build): code shipped at 6958f0c with the
   census classifier stale; the acceptance instrument needed a driver
   follow-up (b097ce1) before the revival was measurable. This build
   re-gates the census IN-BUILD (engine-free, ~minutes).

## Scope

IN: the Professor's plan-time low-w expansion + truncation certificate
(coefficients pure geometry; per-draw ceiling w_low from certificate
inversion — the c3 closed-form precedent); the serving rung in
`likelihood.py`, BOTH parities, replacing the F070 fall-through (serve
below w_low, refuse-and-fall only above); band-split integration so
c3/Born below-split hosting becomes engine on [w_low, w_split] only
(reuse `_band_split_mask` / `_engine_envelope_below_split`, no third
copy); `serve_route_census` gains the rung (route taxonomy +
`classify_draw` waterfall mirroring production) and the 10k engine-free
census re-run IN-BUILD; fast synthetic tests incl. the F -> 1 anchor.
OUT: the deltoid coordinate redesign (5c, sequenced after this build's
census re-run); tube trainer resolvable-subarc trim; tiling design and
training campaigns; the 2b arm-extension (wave_refused 2.13% -> 0);
raising W_CEILING_SCHWINGER_QD or any existing certificate constant;
diffractive-bottom CHART training (the rung is analytic; the
`FARFIELD_DIFFRACTIVE` label is untouched).

## Acceptance

- Value accuracy vs the exact engine wherever the certificate admits
  in the overlap band (cheap DD-band oracle, Fact 6): relative error
  <= the certificate's bar, tolerance from the certificate, not tuned.
- Census re-gated in-build (same config, 10k seed 0): the new route is
  LIVE — report its prior fraction, the fraction of the 12.67% saddle
  far-field it covers, and the per-draw w_low >= w_split coverage of
  the 1409 saddle_c3 splits (the deltoid-descope input).
- The F070 site SERVES below the floor instead of falling through, and
  is byte-identical above w_low (null-split identity, the Born/c3 pin).
- Full fast suite green.
- Parsimony: one canonical pin per invariant; F(w -> 0) = 1 as an
  EXACT anchor; re-point existing pins; report added-vs-retired counts.

## Constraints

Branch claude-dev only. Analytic only, certificate-gated: w_low comes
from the derivation, NEVER a measured constant. Closes
`todo.d/lensing_low_w_diffractive_analytic_rung.md` with a completed.d
record; `[→ spec]` — spec_changelog.d fragment with `bump:`; render
fragments after writing. Escalate-not-iterate: if the truncated
expansion misses the engine reference ANYWHERE the certificate admits
in the overlap band, STOP — that falsifies the derivation or
certificate, not the plumbing; never widen bars or add fudge factors.
In-build tests fast/synthetic; the engine-free census re-gate is
in-build (Fact 7); bulk oracle sweeps remain driver post-build steps.
