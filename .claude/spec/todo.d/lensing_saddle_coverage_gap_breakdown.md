---
section: Backlog
---

- **THE SADDLE COVERAGE GAP IS 1236/10000 DRAWS AND FOUR DISTINCT CAUSES, ONE
  OF WHICH DOMINATES** `[→ spec]` — measured 2026-08-12 by the driver with
  `scripts/census_dry_run.py --n-samples 10000 --seed 42`, then instrumented
  per rejection branch. Structural only: no engine calls, no trained NPZ, so
  it re-runs in ~1 min and needs no training.

  Whole-prior structural coverage **87.61%**. Of the 1239-draw gap, 1236 are
  saddle and 3 near-caustic. Route: origin `rho > 1` serves via the Born rung
  FIRST; only `rho <= 1` reaches `_classify_saddle`. Of 3700 saddle draws,
  1958 are Born-served and **1742** reach the lobe path.

      901 (51.7%)  rho_lobe > rho_outer            beyond the charted shell
      494          SERVED lobe_exterior
      177 (10.2%)  in band, admits_exterior()=False
      149 ( 8.6%)  rho_lobe < 1, admits()=False
       12          SERVED lobe_interior
        9 ( 0.5%)  no lobe admission for band

  The four rejection branches sum to 1236 and the two SERVED counts match the
  census's own rows, so the instrumentation is faithful to the classifier.

  ## Cause 1 (901) — the charted shell does not reach the corridor

  Gap sources sit at `rho_lobe` p50 ~4.9, p90 ~8.5, max ~11, while
  `rho_outer` is only **3.3-3.5** over most of the saddle range (3.533 at
  gamma=1.1, 3.528 at 1.3, 3.390 at 1.5, 3.301 at 1.59). Since
  `rho_lobe = |y - centroid| / r_deltoid(theta)` and the corridor sits
  ~1.0-1.7 from the `+y1` centroid (measured `|centroid|` 0.998 -> 1.668 over
  gamma 1.1 -> 1.59) while `r_deltoid` is a few tenths, corridor sources land
  squarely in the uncovered 3.5-11 band.

  This is the INTER-LOBE CORRIDOR as a number, confirming the standing
  prediction in [[lensing_saddle_forensics]] item (e): the corridor has no
  natural centroid, so neither lobe's polar frame is right for it.

  SUB-POPULATION, 144 draws: their band has `rho_outer <= 1` (measured as low
  as **-4.147** at gamma=1.005), so the exterior band `(1, rho_outer]` is
  EMPTY BY CONSTRUCTION and no lobe-exterior chart can ever serve them.
  `rho_outer = 1 + _SOURCE_BOX_CORNER - coordinate_radius_min` goes negative
  when `coordinate_radius_min` blows up near the parity boundary. Decide
  whether that formula is right near gamma=1 or whether the region needs a
  different rung; a negative width should probably refuse loudly rather than
  silently yield an empty band.

  ANSWERED 2026-08-12 — the answer SPLITS by frequency. The question was whether a
  source at `rho_lobe ~ 5-11` is physically far-field, since `rho_lobe`
  divides by the small `r_deltoid`. Measured on 780 of this population: the
  PHYSICAL distance to the nearest caustic point is `eta` p50 **0.971**, and
  **99.2% have `eta >= 0.3`** (`ETA_MIN_GEOMETRIC`) while only 0.7% are
  genuinely near-caustic. They are far from the caustic and do not need a
  chart at all. BUT frequency decides what can serve them: w > 60 (216 draws,
  27.7%) are already served by the saddle stationary-phase arm and only the
  INSTRUMENT mislabels them, while w <= 60 (564 draws, 72.3%) genuinely need
  chart coverage, so extending `rho_outer` remains on the table for THAT
  sub-population. Full measurement in
  [[lensing_saddle_gap_is_a_routing_failure_not_coverage]].

  ## Causes 2 and 3 (177 + 149) — admission predicates refuse

  177 sources lie inside the charted exterior band yet fail
  `admits_exterior()`; 149 lie inside a lobe yet fail `admits()`. Both are the
  `caustic_cloud` nearest-distance `>= eta_max` tube-shell exclusion doing its
  job — the cusp/fold neighbourhoods are deliberately carved out. The question
  is not whether to remove the carve-out but WHICH named rung serves the
  carved region: the Pearcey cusp arm and the Airy fold arm exist for exactly
  this, and `_classify_saddle` never consults them.

  ## Cause 4 (9) — was 292, and was an INSTRUMENT artifact

  `_saddle_lobe_admission` used a FIXED-width gamma band. A fixed band can
  straddle a change in the caustic's fold-arc count, which makes
  `band_caustic_structure` raise `CausticTopologyError` ("Split the band" —
  measured counts [10, 6, 6] across (1.001, 1.05)). The census swallowed that
  and returned no admission, reporting as uncovered a region production covers
  fine: production bisects via `stable_gamma_bands`, which yields 16 stable
  sub-bands there (one measure-zero sliver dropped), each producing 2 lobe
  admissions.

  FIXED 2026-08-12: the census now mirrors production's bisection.
  "No admission" fell 292 -> 9.

  BUT ONLY 41 DRAWS BECAME SERVED (coverage 87.20% -> 87.61%). The other ~250
  moved into OTHER rejection branches — `rho_lobe > rho_outer` rose 721 -> 901,
  `admits_exterior` refusals 155 -> 177, `admits` refusals 109 -> 149. So this
  was a MISATTRIBUTION, not a phantom gap: the draws were always uncovered,
  the instrument just blamed the wrong cause. Worth remembering when a fix
  moves a number — check whether the population was served or merely
  reclassified.

  ## Acceptance

  Re-run the same probe and report the same six-way breakdown. Each cause
  closes independently; quote PER-CAUSE counts, never just the total, so a fix
  that moves one and not the others is visible.
