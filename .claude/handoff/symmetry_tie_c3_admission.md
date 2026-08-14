# Build: certificate admission for 2-image saddle exteriors (c3-led, ghost-optional)

## Mission

HEAD's macro-saddle far-field gate (`likelihood._saddle_farfield_analytic_serves`,
~:538) refuses on two wrong questions: (a) a scalar `rho <
_SADDLE_FARFIELD_RHO_FLOOR = 2.0` floor — a proxy the measured failure
geometry does not respect; (b) a pairwise delta_tau resolution leg whose tie
filter is `delta_taus > 0`, so a symmetry-tied mirror pair (delta_tau == 0
exactly, images SPATIALLY far apart) falls through to min() of the wrong set
and refuses. Measured: the on-axis tied witness and its 0.03-off-axis
neighbour have IDENTICAL error curves (<1% apart) — the tie changes nothing
physical; the refusal is asking the wrong question.

Replace both with admission on the measured error law: the zero-envelope
serve's true remainder decays as `w^-3` with the SAME SHAPE as the c3 term
`geometry.ppgo_error_estimate` already computes (parity-blind, per-image).
Admit a 2-image saddle exterior draw iff a safety-factored certificate at the
draw's band floor clears the production bar; discriminate genuine coalescence
from symmetry ties by IMAGE SEPARATION, not delay coincidence.

## Measured facts (do NOT re-derive; each cost real engine time)

All at SHA `9f331dd` unless noted (production code untouched since; only SDK
scripts/spec moved). Raw data: `scripts/calibration_pilot_followup.json`;
generator: `scripts/calibrate_saddle_exterior_certificate.py` (thin caller of
production geometry + the pairing-validated tier-1 test helpers; its pairing
gate ran green before every claim below). Contract currency everywhere:
pointwise `err(w) = |F_serve - F_exact| / |F_exact|`, production bar
`p90 <= 1e-3 AND max <= 1e-2` over `w in [8, W_CEILING_SCHWINGER = 60]`
(_schwinger.py:119 — the whole band is cheap double-double, ~43 ms/eval).

1. **The remainder is w^-3 everywhere measured.** Power-law fits on 24-node
   curves at 8 witnesses spanning connecting region (rho 0.3, incl. exact
   axis), the gamma=1.2 ridge (angle 1.41, scales 2.0/2.6), transverse cone,
   generic: k = 2.90-3.82 with R^2 0.96-1.000, ALWAYS beating exponential
   (R^2 0.83-0.94). The omitted physics has the c3 SHAPE, not the ghost's
   exp(-a w) shape. Contract-pass w floors (smallest w such that [w, 60]
   passes the bar): connecting g1.2 12.40 / g1.5 8.0 / g2.0 8.0; ridge
   16.12 (s=2.0) / 11.36 (s=2.6); transverse 8.0; generic 8.0. The
   connecting g1.2 curve has a non-monotone knee at w~14.8 before the tail.
2. **c3 alone is optimistic by a bounded, region-dependent factor.**
   c3/actual shortfall: up to 9.4x connecting region, <=1.6x
   ridge/transverse, <=5.9x generic (pointwise, 672 (config, w) points).
   Because remainder and c3 share the w^-3 shape, the ratio is
   w-stable — that is the boundedness argument a safety factor stands on.
3. **The ghost term is unavailable exactly where the serve fails hardest.**
   `geometry._ghost_candidates` rejects on `Re(root) <= 0` (the principal-log
   branch domain of `_ghost_delay`); measured: the ENTIRE gamma=1.2 ridge,
   the connecting region, and the transverse cone are outside the branch
   domain, while 74% of generic draws (rho in [1.2, 4]) are inside. The
   non-image quartic pair there IS a genuine complex-conjugate stationary
   pair (lens-eq residual 1e-14..1e-16) — the ghost EXISTS but the shipped
   continuation cannot reach it. Branch-corrected continuation is OUT of
   scope (real derivation risk); exactly ON a principal axis the pair
   collapses to a spurious real double root (residual O(1)).
4. **Where the full c3+ghost certificate evaluates it is NOT interior-like
   conservative:** 25% optimistic pointwise (min ratio 0.169), though every
   optimistic witness has err_max 1.9e-6..5e-4 — zero false admits at any
   practical bar on the measured set. Near the lobe axis (angles 0.02-0.15)
   the ghost term over-refuses by 23-1140x (Im tau_c small, true err tiny).
5. **Eta/rho scalar floors are dead ends, measured twice:** the production-
   metric eta boundary is ~2.90 (gamma=1.2, angular ridge at ~1.41 rad —
   NOT the transverse direction; error oscillates with radial scale along
   the ridge), so any safe scalar floor (x1.15 -> 3.35) swallows the entire
   audited real-use band (eta 1.0-2.5, T1 witness 1.994). Fix-agent-attested
   per-gamma boundaries (docstring provenance only, no captured stdout):
   g1.2 ~2.90, g1.5 ~2.2, g2.0 ~1.5.
6. HEAD state (verified, not assumed): no `_SADDLE_ETA_FLOOR`, no
   `_SADDLE_TIE_EPS` anywhere; the dead build's gate/census/test work was
   fully reverted (checkpoint d5672fa6cd98 is NOT an ancestor). The live
   rung computes `rho = caustic_rho(...)` at ~:2136 and passes it down.
7. Oracle edge, do not chase: one gamma=1.1 draw raised
   SchwingerCertificationError at w=60 (paired-rule disagreement 3.279e-10
   vs 3e-10, near the parity wall). Record-and-skip is the right handling.

## Scope

IN:
- **The certificate.** Professor adjudicates its exact form FROM THE
  MEASURED SET (no new scans): the c3-led candidate is
  `admit iff S * ppgo_error_estimate(real_images, y, matrix, w_lo) <= bar`,
  with S covering the measured max shortfall (9.4x) with explicit headroom,
  mirroring the interior methodology (state median/p99/max of the ratio
  distribution; false-admit unacceptable — Professor asymmetry). The ghost
  term MAY be added where the branch domain admits it (it only tightens),
  or omitted for uniformity — decide and say why. Never a new scalar fence.
- **The tie/separation discriminator.** Tied pairs (delta_tau <= ~1e-12)
  with min image separation >= a floor DERIVED from the mirror-pair
  geometry (separation = 2|x|; measure the population, do not pin) are
  symmetry-tied -> certificate path; below it, genuinely merging -> keep
  refusing. Two-sided flip test at the boundary.
- Wire into `_saddle_farfield_analytic_serves` replacing the rho floor and
  the delta_tau tie hole; verify with a serve-path trace whether the same
  refusal exists on the positive-parity exterior path (report; touch only
  if traced).
- CHECKPOINT (driver): verify whether `_saddle_grid` consults this rung at
  all or only `_positive_parity_grid` does. If the saddle node path never
  reaches it, SAY SO — the gate must feed what the saddle path actually
  serves; the gap is a finding, not a silent assumption.
- Census mirror moves in the same build (served == counted).
- Fast decision-level tests + ONE calibration evidence report: re-run
  `scripts/calibrate_saddle_exterior_certificate.py --followup` against the
  shipped gate (~2 min, priced) and show zero false admits with the chosen
  S; witnesses below their measured contract-pass w floors REFUSE, above
  them SERVE.

OUT: branch-corrected ghost continuation (fact 3 — future fragment);
charts/training; the certified-map rho<1 API guard relaxation (5/7, its own
build); `_ppgo_above_ceiling`; lobe interiors; slow tiers; ANY in-build scan
design — the calibration set is handed in, agents may re-RUN the script for
verification only.

## Acceptance

- Connecting-region, ridge, transverse and generic witnesses (both the
  anchor set and fresh draws) SERVE analytically iff the certificate clears
  the production bar at their band floor, with measured error vs the
  parity-correct oracle within the bar — REPORTED per witness, including
  every fact-1 witness at a w_lo above its measured pass floor serving and
  below it refusing.
- The accidental-tie (coalescing) witness still refuses; the separation
  discriminator flips two-sided at its derived boundary.
- Zero false admits over the full measured calibration set with the shipped
  S; the ratio-distribution table (median/p99/max, % optimistic before
  S) appears in the evidence report with the S derivation.
- Existing saddle + exterior suites green; no scalar rho/eta floor remains
  reachable in the 2-image exterior gate.

## Constraints

Branch claude-dev; spec/TODO fragments per CLAUDE.md; values-not-paths;
pairing gate before any oracle claim; geometric sanity (place every number
against the astroid/deltoid picture); briefs discipline — measurement
belongs driver-side, and the last build DIED at error_max_turns iterating a
scan in-build: if any WP believes it needs a new measurement, escalate,
do not iterate.
