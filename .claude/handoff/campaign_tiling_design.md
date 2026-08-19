# Build: demand-sized tiling design + campaign cost estimate (7a step 2)

## Mission

Owner doctrine (`todo.d/lensing_training_campaign.md`, binding):
analytics serve first; charts exist ONLY where the analytic ladder
cannot serve; every axis sized by the problem's own scales, never a
bare count. ORDER-OF-WORK STEP 2 — DEMAND-SIZED TILING: tile the
residual only and PRICE it. Deliver the tile/node plan and THE
CAMPAIGN COST ESTIMATE for owner pre-campaign review; the campaign RUN
is step 4 (driver-monitored, after review), NOT this build.
The stopped first launch (332-chart gamma slice, killed at 4%) failed
the bar three ways, on record in the fragment and FORBIDDEN here:
(1) one 0.04-wide gamma band per parity (~4%/~7% of the prior);
(2) 7 gamma nodes per sliver band (a count against the wrong measure);
(3) 82% of the budget training the astroid exterior that the Born
intercept / c3 certificate / certified map already serve — tables
PREEMPTING the analytics. Each must be structurally impossible.

## Facts (measured at d491e71 unless noted)

1. Honest demand map (`demand_census_post_born_10k.json`, 10k seed 0,
   20-1024 Hz, committed 3f3c57d): ppgo_above_ceiling 15.87%,
   saddle_c3 14.09%, diffractive_analytic 8.12%,
   diffractive_engine_hosted 15.17% (ALL saddle — Rung S hosting),
   engine_residual 44.62%, wave_refused 2.13%. The JSON PREDATES the
   band-split revival (born_analytic reads 0; the Born completion
   measured +3.430% revived) — honest ledger engine_residual ~41.19%
   (2026-08-18_born_farfield_completion.md).
2. engine_residual composition (same JSON, 4462 draws): wedge_interior
   2682 / exterior 801 (= the census's born_chart_demand split) / tube
   781 (near_caustic_tube) / lobe_interior 198; 4235 astroid / 227
   saddle; 3451 draws at w_hi<=60, 989 in 60<w_hi<=150, 22 above 150
   — above-60 need is the above-ceiling rungs', NOT chart demand.
3. Saddle split in the LOBE gauge (`saddle_residual_lobe_gauge_probe
   .md`, 9f331dd, post-c3-regate census): of 1720 saddle residual
   draws, 1267 = 12.67% of prior are genuine 2-image lobe-EXTERIOR
   far-field — all w_hi<=60 (p50/p99/max 4.61/30.7/38.0, w_lo to
   0.026, no w_split), rho_lobe p50/p90/max 5.2/9.6/20.2 vs rho_outer
   caps 1.25-2.40; lobe_interior 1.18%, tube_shell 1.84%,
   exterior_shell 1.17%, guard 0.34%; post-Born these draws sit mostly
   under diffractive_engine_hosted.
4. w-need edges: all 1409 saddle_c3 splits carry w_split p1/p50/p99/max
   = 1.79/14.5/40.3/51.6 (`wp3_low_w_census_10k.json`, 733b7ef);
   below-split hosting is engine on [w_low, w_split] only (low-w
   rung); every edge under the DD ceiling 60.
5. Predictor (`cogwheel/lensing/tiling_census.py`, 7873a45):
   engine-free per-(region x parity x band) node counting via the
   production tilers, `_LABELS_PER_NODE=8`, `_SECONDS_PER_LABEL=0.09`,
   `_self_estimate` cross-check, Q1-Q4; REPORT EVIDENCE, never a gate;
   UPPER BOUND (`ppgo_trim_modeled: False`).
6. The 50h/2M precedent (`campaign_cost_estimate_7a.md`, 2026-08-15;
   `tiling_census_production_postF081.json`, 5aedd5a): 252,000 nodes
   x 8 = 2,016,000 calls; two independent estimates agreed to 0.4%
   (census model 181,440 s; smoke currency 36,672 calls / 3,311.8 s =
   0.0903 s/call, DD band); astroid exterior = 205,800 nodes = 82%.
   STOPPED. Fragment expectation: a SMALL FRACTION of the 2M calls.
7. Representations settled, each with its home: tube = beat-free
   r = E/F_ref + parity-gated trainer trim (completions 2026-08-17/18);
   f_max=0.40 / f_floor=0.08 both parities, DENSITY-not-constants
   closes the bar gap — flagged bands astroid gamma 0.10-0.40, saddle
   ~1.1; the lever: F083 eps 4.3e-3 at n_theta=10 on the core sub-arc
   vs 0.108 at n_theta=7 on the full trimmed span
   (`f_constants_decision.md`, 5ceb2b3, measured 77da2e6); far-field/
   annulus residual charts store vs the two-image GO carrier,
   lobe-gauge cusp-adapted on the saddle (Born completion);
   saddle-bottom anchor charts vs the EXACT F(w->0) = -1j*sqrt(mu_macro)
   (campaign fragment — retires Rung S's 15.17%); wedge/lobe interiors
   keep existing machinery. Deltoid far-field coordinate: standing
   SEPARATE redesign build (`deltoid_farfield_redesign.md`, Q2
   cusp-in-tile); it "trains nothing until it lands" — this design
   prices the region and flags the dependency.

## Scope

IN: a tiling-plan module + machine-readable plan artifact — demand-sized
tile enumerations per region x parity x band with per-tile node counts
and w-ranges, a THIN consumer of the production tilers plus the demand
map (the tiling census's own pattern; NO hand counts); an engine-free
10k census refresh at build HEAD (honest post-band-split routes,
reconciled to the ~41% ledger — minutes); gamma bands from the
caustic-relative measure, wide away from the wall, full residual reach
per parity; per-region w axes bounded by the measured edges (Fact 4
splits, Fact 3 far-field w_hi<=38 — never a blanket [w_floor, 60]);
per-band n_theta sized on the trimmed span, linear in span (Fact 7);
far-field annulus extents from the certificate/serving handoff
boundaries (c3/geometric/low-w rungs, prior edge rho_lobe ~20), NOT the
old rho_outer caps (Fact 3); re-probe of the f_constants outlier
(gamma=0.10, f_max=0.28); THE COST ESTIMATE: total calls x measured
0.0903 s/call = wall-clock with per-region breakdown, cross-checked vs
`_self_estimate` and the census aggregate, written to a NEW handoff
artifact for owner review AND quoted in the completion record;
census/tiling-census mirrors updated IFF the plan changes what they
count; fast synthetic tests.
OUT: RUNNING the campaign (step 4, owner-gated); any chart training;
the deltoid far-field coordinate redesign itself (Fact 7); serving-
ladder changes; step-3 pre-train checklist items (f_floor sweep,
N_map=501, tube 2-D gamma map decision, band-merge packing); the 2b
arm-extension (wave_refused 2.13%); raising any ceiling or constant.

## Acceptance

- No-explosion: every axis's node count traceable to a measured scale
  (a span over a resolution), never a bare count; per-region totals
  justified against the refreshed demand map; each forbidden failure
  mode demonstrably absent (full-reach gamma coverage; caustic-relative
  gamma measure; astroid-exterior tiles confined to residual cells).
- Cost estimate with per-region breakdown; census-aggregate and
  `_self_estimate` cross-checks reported; total a small fraction of the
  2M precedent — if NOT, STOP and escalate (a demand-sizing falsifier,
  never a knob to tune).
- Every planned label node DD-band (w<=60): no tile needs QD/mpmath.
- Flagged density bands (astroid 0.10-0.40, saddle ~1.1) carry per-band
  n_theta allocations closing the measured eps gaps (Fact 7 lever).
- Full fast suite green. Parsimony: one canonical pin per invariant;
  report added-vs-retired counts.

## Constraints

Branch claude-dev only. ENGINE-FREE build — the design and the estimate
use geometry, tilers, and the measured per-call currency; zero wave
evaluations. The campaign fragment `todo.d/lensing_training_campaign.md`
STAYS OPEN (steps 3-4 remain): completed.d record for this step only,
spec_changelog.d fragment with `bump:` if SPEC is touched; render
fragments. In-build tests fast/synthetic.
Escalate-not-iterate: an exploding count, demand untraceable to census
cells, or an estimate near the old 2M is a design falsifier — stop and
report, never widen bands or shave nodes to force the number.
