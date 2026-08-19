# Build brief — diffractive certificate: serve to the honest ceiling

## Mission

`diffractive_w_low` (Rung P, positive parity) refuses vastly more than its own
honest verifier requires.  It proposes a closed-form candidate and then only
ever searches DOWNWARD: `honest_error <= CERTIFICATION_BAR` returns the
candidate verbatim, an over-reaching candidate is root-found down, and nothing
ever searches UP.  A conservative candidate is therefore shipped as-is.

Make the certificate return the ceiling its own verifier actually supports, and
let the serving ladder use it.  This is a certificate + routing build: NO new
charts, NO training, NO campaign.

## Measured facts (measured at 1a06ef3, engine-free, census
`.claude/handoff/demand_census_post_born_10k.json`, 10k draws)

1. Shipped `w_low` vs the honest ceiling at the SHIPPED truncation order 8,
   over astroid-side `engine_residual` draws (median ratio, sample per bin):
   gamma < 0.05 -> **405x** (n=40); 0.05-0.1 -> **67.0x**; 0.1-0.3 -> **23.3x**;
   0.3-0.6 -> **5.1x**.  For gamma in [0.6, 1) the shipped certificate returned
   `None` on every sampled draw.
2. At gamma < 0.05 the honest ceiling covers the WHOLE band on **77.5%** of
   residual draws; the shipped certificate covers 0% of them.
3. Whole-band / partial coverage at order 8 across astroid residual demand:
   exterior **36.4% whole + 44.1% partial**; tube 6.4% + 26.4%;
   wedge_interior 1.8% + 8.6%.  Whole-band conversion alone is ~8.7% of
   residual demand (44.62% -> 40.74% of all draws); the PARTIAL share is the
   larger prize and is realized by band-splitting at the certified ceiling.
4. The honest verifier is the N/2N tail ratio over `_operator_terms` evaluated
   at `2 * order` — the SAME quantity the shipped `relerr` closure already
   computes inside `diffractive_w_low`.  Bind it; do not re-type it.
5. Median honest ceiling vs median band top: exterior 9.21 vs 10.05, tube 10.61
   vs 7.95, wedge_interior 5.28 vs 25.13.  The exterior ceiling lands AT the
   band top — that region is close to fully analytic.
6. `born_carrier_only` fired **0 times in 10 000 draws**: the carrier
   certificate refuses 801/801 exterior residual draws on the accuracy bar,
   median overshoot **1.26e4x**, and the estimate does NOT improve with rho
   (median est 0.46 at rho in [2,4) vs 1.36 at rho > 100).  This matches the
   recorded lead-carrier-dead finding (second-image beat).  DO NOT attempt to
   revive the carrier-only rung in this build.

## Scope

IN:
- `cogwheel/lensing/chang_refsdal/_diffractive.py`: `diffractive_w_low` and
  its root-find helper.  The certificate must bracket UP from the candidate
  and return the largest w whose honest tail ratio clears
  `CERTIFICATION_BAR`, and must still refuse when even the band floor fails.
- Truncation order: scan it on the fixture set (8 / 12 / 16 / 24) and record
  whether reach rises (convergent) or saturates/falls (asymptotic).  Change
  the default order ONLY if the scan shows a clear win at acceptable cost;
  record the measurement either way.
- The consumers of that ceiling: `likelihood._diffractive_bottom_ceiling` /
  the low-w diffractive serve, so a draw certified across its whole band
  routes analytic, and a partially certified draw band-splits at the ceiling
  (analytic below, the previous behaviour above).
- `cogwheel/lensing/serve_route_census.py`: update the mirror by BINDING the
  production predicates (mirror-fidelity law — never re-type a gate).

OUT (do not touch):
- Rung S / the macro saddle: the series diverges at every order — established,
  not re-litigated here.
- The Born carrier-only certificate (fact 6) and the Born residual chart.
- Any chart coordinate or representation change; any training or campaign run.

## Acceptance

- MONOTONE: on a fixture set spanning the gamma bins above, the new ceiling is
  `>=` the old one for every point, and strictly greater where fact 1 predicts
  a gain.
- TIGHT, NOT ARBITRARY: at the returned ceiling the order-M truncation agrees
  with the 2M-order series within `CERTIFICATION_BAR`; at 1.5x the returned
  ceiling it does NOT.  Both directions pinned.
- HONEST AGAINST THE ENGINE (the real oracle): on ~12 fixture points spanning
  the gamma bins, `diffractive_amplification` at the returned ceiling agrees
  with the exact engine within `CERTIFICATION_BAR`.  A dozen calls at 0.0903
  s/call is the whole engine budget for this build — no sweeps.
- VALUES, NOT PATHS: assert the served amplification against an oracle and a
  tolerance, not which branch produced it.
- LEDGER: re-run the 10k serve-route census and report the new
  `engine_residual` share and the `diffractive_analytic` share.  Expect
  engine_residual to fall from 44.62%.  Report added-vs-retired test counts.

## Constraints

- Branch `claude-dev` only.  Slow tiers stay gated.  In-build tests FAST.
- Spec/TODO workflow applies: this is a behavior change — `[→ spec]`.
