# Green-field audit — the serving stack designed from scratch (2026-07-23)

Given: every measurement of the past week. Question: what would we build
today, and which queued work does that design NOT contain?

## The from-scratch design

ONE fact organizes everything measured: wave effects are irreducible
only in a COMPACT region — the caustic neighborhood in caustic units,
at low-to-moderate w. Everything outside it is analytic:

- serving rungs (all analytic, zero training): real-image ppGO above
  the measured trust floor; the ghost-pair term in the fold annulus;
  Airy/Pearcey arms in their windows; (candidate) a Born rung at the
  deep-diffraction floor, expanded about the exact macro limit
  `F(w -> 0) = 1/sqrt((1-kappa)^2 - gamma^2)` (NOT about 1: `F -> 1` holds
  only at `gamma = kappa = 0` — F009, `.claude/spec/FINDINGS.md`).
- the trained core: charts ONLY over (rho = caustic-scaled radius,
  theta_c, gamma, w in [w_floor, w_cert(cell)]) — ONE coordinate
  system for interior AND exterior, per-lobe frames for the saddle.
  Two labels, tagged: (a) outside + non-crown interior: F minus
  everything-analytic (real kernels + gated ghost); (b) near-critical
  interior (and plausibly ALL interior — Professor call): the SACR-C
  switched envelope. Component grids: analytic terms cost zero nodes;
  the remainder gets convergence-measured sparse grids.
- certification: the existing measured map (already caustic-scaled),
  with measured-support discipline on both axes (8h-b1, running) and
  the monotone-outward theorem as the one missing piece for
  prior-universality.

## What the fresh design does NOT contain (= cuts to the queue)

1. **Two sequential representation migrations.** The frozen plan does
   the ghost label on RAW-coordinate exterior tiles (8h-b2), then
   moves interiors to caustic coordinates (8h-b3) — two retrains, two
   re-pilots. The fresh design has ONE representation: caustic-fixed
   coordinates on BOTH sides of the caustic, labels and coordinates
   land together, ONE retrain. MERGE 8h-b2+8h-b3 into a single
   "caustic-fixed core" build (WP3 ghost machinery survives verbatim —
   it is coordinate-independent physics; WP6/7/8 survive; WP4/5 fold
   into them with grids now caustic-fixed).
2. **Mass strata.** They exist only because serving was whole-band: a
   chart had to contain a draw's entire 1.7-decade w band, so charts
   were replicated across log-mass windows. With per-node band-split
   serving (live since 8h-a), a chart's w-window can simply be
   [w_floor, w_trust(region)] — which CONTAINS every draw's chart
   segment by construction. One chart family per region, NO strata:
   tile count drops by the strata multiplicity (~3x) and the
   whole-band containment logic in serving reduces to a range check.
   The strata-trim machinery (8h-a WP3, 8h-b1 WP2 consumer) remains
   correct but becomes vestigial once charts adopt fixed w-windows —
   do not extend it further.
3. **The interior far-field-style label as a separate family** — IF
   the Professor confirms SACR-C envelope serves the whole interior
   (measured hint: far-field-style label fails 6e-2 even at mid-gamma),
   the interior has ONE label, not two.
4. **Raw-coordinate exterior tiles and every artifact built from them**
   (v3 charts, pilot charts): already understood as disposable; no
   further work invests in them.
5. **NOT cut** (survives any redesign): tubes and arms (already
   caustic-fixed, certified), the certification map + measured-support
   discipline, the eps gate + registration machinery, the ladder
   census, the refusal vocabulary, band-split dispatch, the ghost
   physics (WP3), the SDK verification spine.

## The one unmeasured input the design needs

The LOW-W FLOOR: does the deep-diffraction wave correction vary on the
caustic scale (then it lives inside the compact core — done) or the
Einstein scale (then universality needs the Born rung)? One cheap probe
(~20 exact evals, minutes) decides it, and decides whether w_floor is
a config or a physics constant.

## Sequencing consequence

- 8h-b1 (running): unaffected — measurement-support discipline is
  coordinate-independent. Its WP2 consumer work is retained but not
  extended (see cut 2).
- REPLACE queued 8h-b2 + 8h-b3 with ONE build: "8h-B core" =
  WP3(ghost, verbatim) + caustic-fixed charts both sides (WP6/7/8
  texts) + both labels + fixed per-region w-windows (strata removed)
  + component grids. Width demands a split: WP3 alone is
  self-contained (pure geometry, no consumers) -> 8h-b2' = WP3 +
  low-w-floor probe; 8h-b3' = the coordinate/label/window core.
- Then: calibration re-pilot -> qd (8h-c, BEFORE the campaign per the
  standing ruling: saddle ceilings/windows and labels become final,
  so ONE campaign suffices; stall contingency stands) -> map
  extension sweep -> ONE campaign -> census -> 100%.
