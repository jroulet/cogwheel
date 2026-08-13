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

  ## The exception is a GAUGE artifact, and the fix is to re-gauge

  1 of 40 draws at `w = 58` is envelope-DOMINATED (`|E|/|F| = 4.17e-01`). It
  is cleanly separated by the EXISTING resolution criterion — measured over
  180 (source, w) pairs:

      w*delta_min   n    |E|/|F| p50    p90       MAX
          [0,4)     6      2.74e-02   5.93e-01   6.71e-01
          [4,8)    12      4.78e-05   1.68e-04   2.38e-04
         [8,16)    19      3.31e-05   1.50e-04   9.65e-03
        [16,32)    45      1.30e-05   6.74e-05   1.11e-04
        [32,64)    55      6.27e-06   2.49e-05   4.40e-05
       [64,inf)    43      1.26e-06   3.14e-06   7.15e-06

  WHY, and it is not what it looks like. The excluded population is FARTHER
  from the caustic than the served one (`eta` p50 **1.169** vs 0.992), with
  **0.0%** inside the Airy fold fence, and `r(eta, delta_min) = -0.118` —
  uncorrelated. These are not merging images.

  `delta_min = |tau_a - tau_c|` measures a real image's delay against the
  CRITICAL CARRIER, which for a far-from-caustic source is a VIRTUAL
  reference: the delay at the nearest caustic point, where no image sits. A
  real image's arrival time landing near it is a coincidence of geometry.
  The SACR-C switch is `S_a = smootherstep(w*|tau_a - tau_c|, 0.5, 4)`, so a
  small argument turns the switch OFF and hands that channel to the envelope
  BY CONSTRUCTION. The failure threshold matching `RHO_END = 4.0` exactly is
  the tell: this is the switch, not the physics.

  ## Plan: re-gauge, do not re-chart

  SACR-C is an EXACT ALGEBRAIC GAUGE, not an asymptotic expansion — `E`
  absorbs whatever the trials miss, and NOTHING requires `tau_c` to be the
  critical-point delay. `F` is reproduced for any choice.

  So for the far-from-caustic saddle region (`eta >= ETA_MIN_GEOMETRIC`,
  where no fold arm is needed), choose `tau_c` to MAXIMISE the minimum switch
  argument instead of parking it at the critical point. Worked example, the
  outlier: real delays `[0, 0.9333]`, `tau_c = -0.0443` gives
  `delta_min = 0.044` and `w*delta_min = 2.57`; placing `tau_c` between the
  images (~0.4) gives `delta_min ~ 0.4`, hence `w*delta_min ~ 23` at `w = 58`
  — comfortably above `RHO_END`, every switch on, envelope negligible.

  The region is then served by the switched analytic channels with NO chart,
  NO extrapolation and NO engine call, and nothing falls back to exact.

  WHAT TO SETTLE BEFORE BUILDING:
  - the `tau_c` choice must be a FUNCTION of the source, identical at train
    and serve time, or the stored envelope is in a different gauge than the
    one served (this is the train/serve skew class of bug already recorded
    for the ghost gate);
  - near the caustic the critical-point `tau_c` is the RIGHT choice — the
    merging pair's delays converge to it and the envelope handles them
    smoothly. So this is a REGIONAL gauge switch, and the handover between
    the two conventions needs its own continuity check;
  - confirm empirically that re-gauging actually drops `|E|/|F|` for the
    outlier, rather than moving the weight to a different channel. That is a
    direct measurement and must be made before any build.

  ## Process note

  A build was launched to adjudicate the `w`-flatness question and killed
  after 54 minutes of Professor deliberation with zero loggable output. That
  was a briefing error: "is X true of the physics" is a MEASUREMENT, not a
  design decision, and the driver answered it in ~3 minutes of probing. Send
  builds decided approaches; settle empirical questions first.
