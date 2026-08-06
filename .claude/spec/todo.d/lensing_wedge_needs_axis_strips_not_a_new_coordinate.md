---
section: Backlog
---

- **THE WEDGE PATH WORKS — the failure is thin strips along the reflection
  axes; exclude them and it beats `ffin` 5:1** `[→ spec]` — measured
  2026-08-06. SUPERSEDES the diagnoses in
  [[lensing_wedge_charts_fail_the_eps_bar]] (blamed the coordinate) and
  [[lensing_wedge_fails_only_at_the_cusp_axis]] (blamed cusp diffraction).

  ## The decisive measurement

  Same band, same radial tile (`r = 0.455 +- 0.089`), same `n_theta = 7`, only
  the angular tile POSITION differs:

  | angular tiles | eps, tile TOUCHING an axis | eps, INTERIOR tile |
  |---|---|---|
  | 4 | 1.29e-1 | **3.82e-4** |
  | 8 | 6.48e-2 | **3.27e-4** |
  | 12 | 3.96e-2 | — |

  Interior tiles are **~340x more accurate** than axis-adjacent ones at
  identical width and node count. They reach the `ffin` baseline (3.42e-4) at
  just FOUR angular tiles, and barely improve beyond that (3.82e-4 -> 3.27e-4)
  — i.e. they are already at the noise floor, fully converged.

  Axis-adjacent tiles converge at exactly FIRST ORDER (each halving of width
  halves eps: 3.93e-1, 2.34e-1, 1.29e-1, 6.48e-2, 3.96e-2). At first order,
  matching `ffin` by global refinement would need ~116x more tiles — hopeless.
  The whole failure is in the strips.

  ## What this means

  `4 angular x 5 radial = 20 charts` at `ffin` accuracy, against `ffin`'s
  **106** for the same region — a genuine **5x** saving, comfortably better
  than the 4x the exact D2 fold alone predicts (the wedge also needs fewer
  radial tiles: the radial direction is superb, 1.31e-4 at FIVE nodes).

  ## What it is NOT (four hypotheses killed by measurement)

  - NOT the coordinate. Radial converges at 1.3e-4 on 5 nodes; transverse at
    p ~ 3.6. The owner's `d_tau^(2/3)` matches wedge `r` to ~10% (they are
    affinely related: `distance = r_caustic(theta) * (1 - r)`), so neither is
    at fault and neither is better.
  - NOT a wiring bug. Node exactness is 6.33e-16.
  - NOT gamma or w resolution. Refining gamma 7->13 leaves eps unchanged to 4
    digits; refining w from 40 to 168 nodes likewise.
  - NOT cusp diffraction. The locus is `theta_wedge -> pi/2` at `r ~ 0.45`,
    nowhere near the cusp POINT at `r = 1`.

  The interference beat between the two delay-degenerate images IS real
  (`d_tau ~ 0.021 per degree` off the axis, ~2.8 cycles across the span at
  `w_max`) but is NOT the binding constraint: at 30 nodes/cycle a cubic spline
  should reach ~1e-6, and axis-adjacent tiles are still at 4e-2 and converging
  at first order. First order means a genuine non-smooth feature on the axis,
  whose mechanism is still unidentified.

  ## Work

  - Exclude a strip of half-width `delta` at `theta_wedge = 0` and `= pi/2`;
    tile the remainder with ~4 angular x 5 radial. The minimum `delta` for
    `ffin` parity is being measured; interior tiles at `delta >= 0.196 rad`
    already pass, so the exclusion costs at most ~25% of the quadrant and
    probably far less.
  - Route the excluded strips to a rung that covers them, and VERIFY it serves
    them to tolerance — an exclusion that nothing covers converts an accuracy
    bug into a coverage hole.
  - `_wedge_serves` must refuse inside the same strips via the SAME predicate
    the tiler uses. Train/serve skew is the recurring bug class here.
  - Report the eps DISTRIBUTION (p50/p90/max) and the WORST-SAMPLE LOCUS in
    the training report. The max-metric summary hid this localisation for a
    full day; the argmax location identified it in ten minutes once printed.
  - SEPARATE QUESTION worth its own investigation: what IS the non-smooth
    feature on the reflection axis? Two images are exactly delay-degenerate
    there (mirror pair), and the SACR-C switch keys on `|tau_a - tau_c|`, so a
    switch or carrier-assignment change across the degeneracy is the obvious
    suspect — but the four hypotheses above were all obvious too, and all
    wrong. Measure before theorising.

  ACCEPTANCE: with axis strips excluded, wedge interior charts reach median
  eps at or below `ffin`'s 3.42e-4 at ~20 charts for the region; the strips
  are served to tolerance by a named rung; and the worst-sample locus is no
  longer the reflection axis.
