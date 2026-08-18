# Build: Born far-field completion, both parities (carrier-only beyond the chart)

## Mission

Owner directive 2026-08-17 ("changes needed for both parities"),
superseding the deltoid coordinate redesign. THE LAW: far from the
caustic the physics gets EASIER — the residual against
`born_lead_carrier` decays as 1/|y'|^2 — so beyond the residual
chart's certified reach the carrier serves ALONE at all w, admitted by
its own truncation certificate (leading omitted term <= bar; the c3
pattern in the spatial direction), to the prior edge and beyond;
`covers()` is never a refusal-to-engine. TWO DEFECTS CLOSED:
(1) astroid — the Born gate falls through to the engine outside the
shipped chart's box; (2) deltoid — the gate keys on origin-gauge
`caustic_rho > 2`, the wrong distance on the saddle (lobes don't
enclose the origin); the saddle rung keys on rho_lobe instead (the
`_lobe_serves` D2 fold onto the +y1 lobe). The shipped low-w rungs
serve/host the band bottom; the target is the above-w_low remainder.

## Facts (measured at 733b7ef unless noted)

1. The Born gate (`likelihood.py`, `_born_residual_analytic`) refuses
   (`return None`) when: `born_chart is None`; `lens['kappa'] != 0.0 or
   lens['beta'] != 0.0`; `lens['gamma'] == 0.0`; `rho <= 2.0 or not
   born_chart.covers(lens['gamma'], rho)`; post band-split,
   `host_mask.any() and not born_chart.covers(gamma, rho, chart_w)`.
   Every `None` falls through to the exact seed engine (line ~3142).
   `covers()` is axis-aligned box containment; the shipped `gamma_grid`
   is astroid-only (0.05-0.9): saddle and beyond-box astroid queries
   are COVERAGE-refused, never certificate-refused. The buried
   surrogate-path twin (line ~2198) gates at `rho <= 1.0`.
2. The carrier (`chang_refsdal/_born.py`): `born_lead_carrier` =
   `morse * sqrt(|mu_macro|) * exp(1j*w*phi_geo)`, BOTH parities
   SHIPPED (exact literal `morse = -1j` on the macro saddle). Small
   parameter 1/|y'|^2 = 1/Q2r; leading OMITTED term `a0/Q2r +
   1j*(w/2)*b1/Q2r + O(w^2/Q2r^2)` — the certificate's raw material —
   with `a0`, `b1` closed-form (`_born_factors`) but POSITIVE-PARITY
   ONLY ("not derived on the macro saddle"); `w` in the NUMERATOR makes
   the certificate a joint (w, distance) bound. Plan-time Professor
   derivation: the certificate FORM, both parities (a saddle a0/b1
   analogue or a bound not needing it), inverted to an admission
   predicate — plus Fact 6's question.
3. Saddle demand (`saddle_residual_lobe_gauge_probe.md`, 9f331dd):
   1267 draws = 12.67% of TOTAL prior are 2-image lobe-EXTERIOR
   far-field, rho_lobe p50/p90/max = 5.2/9.6/20.2 vs rho_outer caps
   1.25-2.40, all `w_hi <= 60`, w_hi p50/p99/max = 4.61/30.7/38.0.
   Origin-gauge rho saturates near 2 there — Born never fires.
4. LOW-W OVERLAP (`wp3_low_w_census_10k.json`, 10k seed 0, regate at
   733b7ef): diffractive_analytic 14.27%, diffractive_engine_hosted
   14.93%, engine_residual 24.10% (2410, down from 53.30% at b097ce1);
   the 2410's composition is the target. The wp3 json is summary-only,
   so the 2410's saddle-vs-positive split is UNMEASURED; at b097ce1
   (per-draw records, `demand_census_post_c3_regate_10k.json`) it was
   1720 saddle / 3610 positive of 5330.
5. Lobe machinery to REUSE (`surrogate.py`): the `_lobe_serves` D2
   fold (`abs(y1_eig)`, `abs(y2_eig)` onto the trained +y1 lobe),
   `_to_lobe_fixed` / `_lobe_boundary_radius` / `_lobe_cusp_axis_map`.
   A gamma > 1 `_saddle_farfield_analytic` rung exists (zero-envelope
   resolvable case) — coexist, don't duplicate.
6. OPTIMAL-REPRESENTATION CLAUSE (owner, 2026-08-17: a shrunken region
   is not an excuse for a mediocre chart): annulus charts store the
   RESIDUAL AGAINST THE SAME CARRIER THE RUNG SERVES (the astroid
   `BornResidualChart` pattern, reconstruct = carrier + R; ONE carrier
   definition shared by rung and chart, DRY), in lobe-gauge
   cusp-adapted coordinates. The two-carrier reference question (lead
   carrier vs non-vanishing two-carrier combination — the subleading
   macro image interferes as cos(w*dtau_macro)) is the LOAD-BEARING
   plan-time Professor derivation — spend the plan review there.
7. Mirror-fidelity (the c3 stale-instrument lesson):
   `serve_route_census.classify_draw` mirrors the production waterfall;
   re-gate IN-BUILD (engine-free, ~minutes). Rung-taxonomy law (low-w
   ledger precedent): carrier-only counts ANALYTIC side ONLY when
   certificate-gated; anything engine-computed stays engine side.

## Scope

IN: the Professor's plan-time derivations (Facts 2 and 6); the
carrier-only serving rung in `likelihood.py`, BOTH parities —
`covers()` declines route to the certificate gate, never straight to
the engine; the saddle rung keyed on rho_lobe via the `_lobe_serves`
D2 fold (existing helpers, no new coordinates); the ONE shared carrier
definition + reconstruct = carrier + R contract the later annulus
charts consume (representation settled NOW); `serve_route_census`
taxonomy for the new route(s) + the 10k engine-free census re-run
IN-BUILD; fast synthetic both-parity value pins.
OUT: annulus chart TRAINING (campaign) and the demand-sized tiling
design (sequenced after this build's census); tube trainer
resolvable-subarc trim; the 2b arm-extension (wave_refused 2.13% -> 0);
any low-w rung change (`_low_w_diffractive_serve`, `diffractive_w_low`,
their certificate constants).

## Acceptance

- Carrier-only accuracy vs the exact engine everywhere the certificate
  admits in the reachable overlap (w <= 60, the cheap DD-band oracle):
  relative error <= the certificate's bar, not tuned. BOTH parities.
- Astroid Born serving byte-identical inside the chart box (the null
  identity — carrier-only activates only beyond it).
- NO `covers()` refusal anywhere in the far field: a beyond-reach draw
  serves carrier-only or is CERTIFICATE-refused into the engine.
- Census re-gated in-build (10k seed 0): report the carrier-only
  fraction (new analytic route), the residual-annulus fraction still
  engine-side, the astroid beyond-box closure, Fact 4's parity split.
- Escalate-not-iterate on any certified-band miss (Constraints).
- Full fast suite green.
- Parsimony: one canonical pin per invariant; re-point existing pins;
  report added-vs-retired counts.

## Constraints

Branch claude-dev only. Certificate-gated analytic serving only:
admission comes from the derivation, NEVER a measured constant. Closes
`todo.d/lensing_born_farfield_completion.md` with a completed.d record;
`[→ spec]` — spec_changelog.d fragment with `bump:`; render fragments.
Escalate-not-iterate: a miss anywhere the certificate admits falsifies
the derivation or certificate, not the plumbing — STOP; never widen
bars or add fudge factors. In-build tests fast/synthetic; the census
re-gate is in-build; bulk oracle sweeps are driver post-build steps.
