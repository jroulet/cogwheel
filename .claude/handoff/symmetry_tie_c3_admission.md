# Build: serve symmetry-tied exteriors via the c3 certificate

## Mission

The connecting region (origin-side saddle exterior) and both parities'
symmetry axes are refused by the w-resolution leg because their image pairs
are DELAY-TIED BY SYMMETRY (delta_tau == 0 exactly, mirror pair x -> -x).
That gate asks the wrong question: delay coincidence by symmetry is NOT
coalescence — the tied images are SPATIALLY FAR APART, each an isolated,
healthy stationary point whose per-image expansion never consults its
partner's delay. Result: measurably-accurate ppGO regions fall to the exact
engine (a failed serve by the owner's standing rule).

Replace the question with the right object: admit symmetry-tied 2-image
exteriors on `geometry.ppgo_error_estimate` — the per-image, parity-blind
c3 certificate that already gates the 4-image interior rung — plus its
ghost term, instead of the pairwise delta_tau leg.

## Measured facts (SHA at launch; F069-safe oracles as noted)

1. Connecting region (saddle, rho <= 0.4, incl. the axis): raw ppGO err
   1e-5..1e-6 for w >= 25 (2026-08-13 audit, saddle oracle
   `_saddle_mass_sheet_map` + f_schwinger, pairing validated 0.0e0 incl.
   kappa=0.2). The shipped certified map's saddle rho<0.5 cells agree:
   w_cert 27.7/19.2/15.9 (artifact read) and an independent re-measurement
   reproduced w_cert 16-28.
2. The tie discipline (saddle build, in flight at launch time) makes
   delta_tau <= 1e-12 pairs NON-resolving — deliberate refusal. This build
   adds the c3 path as an ALTERNATIVE admission for exactly that case; the
   tie discipline stays for accidental (coalescing) ties, discriminated by
   IMAGE SEPARATION: a tied pair with min image distance >= a derived floor
   is symmetry-tied (serve via certificate); below it, genuinely merging
   (keep refusing). Derive the separation floor from geometry (the mirror
   pair's separation is 2|x| — measure the population), do not pin.
3. `geometry.ppgo_error_estimate(real_images, source, matrix, w_min)` is
   shipped, parity-blind, and its interior calibration measured ratio MAX
   0.980 / 0% optimistic (1248 points). Its EXTERIOR behavior needs the
   ghost term: on 2-image censuses add
   |ghost_kernel(w_min).kernel| * exp(-w_min * Im tau_c)
   (GhostAbsentError impossible on a 2-image census — treat as internal
   error; GhostDomainError = unavailable -> REFUSE. NOTE: exactly ON a
   principal axis the ghost reconstruction is UNAVAILABLE by coordinate
   collapse (F073 work, `GhostAbsentError`/`GhostDomainError` split in
   geometry.py) — measure how wide the unavailable band is; if it is the
   measure-zero axis only, serve the certificate WITHOUT the ghost term
   there IFF the ghost bound extrapolated from neighboring off-axis points
   is negligible (state the margin), else refuse and record the band.)
4. Positive-parity axes: exterior on-axis fold band has b3 ~ 0 (fold
   refuses structurally) and tied pairs; same c3 admission applies. Born
   rung owns rho > 2 once wired (F077) — do not duplicate its domain.

## Scope

IN:
- SUPERSEDE THE SADDLE ETA-FLOOR LEG ENTIRELY (added 2026-08-14, driver):
  the `_SADDLE_ETA_FLOOR` gate that build saddle_admission_predicates
  ships is a SCALAR PROXY for the certificate this build installs — eta
  stands in for the two closed-form pieces of the zero-envelope remainder
  (the per-image C-series that `geometry.ppgo_error_estimate` bounds, and
  the omitted ghost `|ghost_kernel| * e^{-w Im tau_c}`). Replace the eta
  leg with direct c3+ghost certificate admission for ALL 2-image saddle
  exteriors (not just tied pairs); retire the eta floor to a cheap sanity
  backstop or delete it (decide from the calibration, state why).
  CALIBRATION DATA EXISTS: the currency-corrected
  `scripts/measure_saddle_eta_floor.py` scan (production-contract
  p90/max per witness over the band, per gamma) is exactly the
  exterior-certificate calibration set — reuse it, do not re-scan; the
  certificate's safety factor comes from its ratio distribution, interior
  methodology (median/p99/max, 0-optimistic bar).
- The c3+ghost admission path for 2-image exteriors with symmetry-tied
  pairs, wired wherever the delta_tau resolution leg refuses ties:
  the saddle serve gate (post-saddle-build shape) and, if the same
  refusal exists on the positive-parity exterior serve path, there too
  (VERIFY with a serve-path trace before editing; report if not).
- CHECKPOINT (from the driver): verify whether `_saddle_grid` consults the
  arm ladder / the new ppGO+ghost rung at all, or only
  `_positive_parity_grid` does. If the saddle node path never reaches the
  rung, say so explicitly — this build's gate must then feed whatever the
  saddle path actually serves, and the gap is a finding, not a silent
  assumption.
- Safety factor derived from a measured exterior-certificate calibration
  (mirror the interior methodology: ratio distribution over a config x w
  grid vs the parity-correct oracle; state median/p99/max; factor with
  headroom over max).
- Census mirror moves in the same build (served == counted).
- Fast decision-level tests + ONE calibration evidence REPORT.

OUT: charts/training, the certified-map API guard relaxation (separate,
now-correctly-priced follow-up), `_ppgo_above_ceiling` (F076 distance gate
is its own item if not already absorbed), lobe interiors, slow tiers.

## Acceptance

- Axis and connecting-region witnesses (both parities where applicable,
  derived fixtures) SERVE analytically with measured error vs the
  parity-correct oracle under 1e-4 at w <= 60, REPORTED; the accidental-tie
  (coalescing) witness still refuses; the separation discriminator's
  boundary has a two-sided flip test.
- Existing saddle + exterior suites green; the tie-discipline tests from
  the saddle build stay green (the c3 path is an ADDITION for the
  far-separated case, not a relaxation of the merging case).

## Constraints

Branch claude-dev; fragments; values-not-paths; pairing gate before any
oracle claim; geometric sanity (place every number against the
astroid/deltoid picture); frequency-independent admission is NOT required
here (this rung is not a label oracle for charts — but if the Professor
rules otherwise, follow the ruling and say so).
