---
section: Backlog
---

- **THE SACR-C ENVELOPE IS NEGLIGIBLE FOR ~97% OF THE SADDLE GAP AT HIGH w —
  but the gate that would license using that does not exist yet** `[→ spec]` —
  measured 2026-08-13 by the driver at HEAD 88174b6.

  ## The ppGO-extrapolation idea, as filed, does NOT work

  [[lensing_ppgo_extrapolation_beyond_engine_reach]] proposed splining the
  demodulated, scaling-stripped residual on the grounds that it is `w`-flat by
  construction. Measured on a representative gap source
  (`gamma=1.30`, `y=(0.50,0.20)`, `eta=0.594`, 2 images), `w` in [6, 58]:

      Re(E), Im(E) decay ~4 orders of magnitude (1.5e-2 -> ~1e-6)
      while OSCILLATING: 5 sign changes in Re, 7 in Im
      2nd differences 0.33-0.37 of scale on a 16-node log grid
      log-log fit of |E|: slope -3.74 but residual rms 0.66 -- not a power law

  Dividing out a smooth analytic scaling cannot flatten an oscillation. The
  residual is neither flat nor a clean power law, so that extrapolation route
  is refuted as filed.

  ## What the decay actually licenses

  The envelope becomes NEGLIGIBLE against the total. `|E| / |F_total|` over
  30-40 draws sampled from the REAL gap population (far from the caustic,
  origin-`rho <= 1`, beyond `rho_outer`):

      w = 24   p50 3.60e-05   p90 1.47e-04   93.3% below 1e-3
      w = 40   p50 4.98e-06   p90 2.76e-05   93.3% below 1e-3
      w = 58   p50 2.97e-06   p90 1.17e-05   96.7% below 1e-3

  So above `w ~ 24-40` the switched analytic channels alone reproduce the
  total to well under the 1e-3 chart bar, for almost the whole population.
  No chart, no extrapolation, no engine call.

  THIS WOULD COVER THE EXPENSIVE POPULATION ENTIRELY. The 216 mpmath-band
  draws (`60 < w <= 148`, ~85-120 s/call, F061) all sit above the crossover,
  as do the ~112 that `w*|y| > 58` makes unchartable at any price
  ([[lensing_saddle_gap_is_a_routing_failure_not_coverage]]). They would stop
  needing a chart rather than needing a cheaper one.

  ## THE BLOCKER: no working gate

  1 of 40 draws at `w = 58` is envelope-DOMINATED at `|E|/|F| = 4.17e-01`,
  five orders of magnitude off the median. Serving it from the analytic
  channels alone would be catastrophically wrong.

  `eta` does NOT discriminate it. Measured: the outlier sits at `eta = 0.509`
  while the 39 good draws span `eta` 0.165 to 1.437 — good draws exist on
  BOTH sides of it. Image count does not either (all 2). The outlier's other
  coordinates are `gamma = 1.59` (near the prior edge, 1.6) and `|y| = 1.438`.

  A rung without a gate keyed on what actually degrades it is exactly the
  `_airy_fold` failure mode: its `xi` self-certificate could not see distance
  from the caustic and read 1.2e-2 where the true error was O(1) (F028,
  confirmed against GLoW in F032; no amplitude refinement removes it, F033).
  Do not ship this rung on a 97% success rate.

  ## Next measurement, before any build

  Characterise the outlier population with a larger sample — but note the
  instrument cost: sampling at `w > 60` enters the mpmath band and costs
  ~100 s per draw, so a 40-draw sweep there is ~1 hour. Sample in the
  double-double band (`w <= 58`) where the same physics is visible at
  ~0.2 s/call, and spot-check above 60 with a handful.

  Specifically: is the outlier a `gamma -> 1.6` prior-edge effect, an
  unresolved-image case that `w * delta_min` would catch, or something else?
  `select_branch`'s resolution leg (`w * delta_min >= RHO_END = 4.0`) is the
  obvious candidate and was NOT measured here.

  ## Process note

  A build was launched to adjudicate the `w`-flatness question and killed
  after 54 minutes of Professor deliberation with zero loggable output. That
  was a briefing error: "is X true of the physics" is a MEASUREMENT, not a
  design decision, and the driver answered it in ~3 minutes of probing. Send
  builds decided approaches; settle empirical questions first.
