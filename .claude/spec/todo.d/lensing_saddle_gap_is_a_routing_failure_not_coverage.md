---
section: Backlog
---

- **THE DOMINANT SADDLE GAP SPLITS 28/72 BETWEEN A ROUTING MISLABEL AND A
  GENUINE COVERAGE HOLE** `[→ spec]` — measured 2026-08-12. Population: the
  780 draws beyond `rho_outer` (Cause 1 of
  [[lensing_saddle_coverage_gap_breakdown]], excluding the 144 with a
  degenerate band).

  ## They are all FAR from the caustic

  Physical distance to the nearest caustic point
  (`geometry.nearest_caustic_point(...).distance`):

      eta   p10 0.617   p50 0.971   p90 1.380   max 2.682
      eta >= 0.3 (ETA_MIN_GEOMETRIC): 99.2%
      eta <  0.3 (genuinely near-caustic): 0.7%

  `rho_lobe` p50 is 5.70 but `|y|` p50 is only 0.732 — the large normalised
  radius is an ARTEFACT of dividing by the small `r_deltoid`. These sources
  sit near the origin, between the lobes, a full unit from the caustic.

  ## But frequency decides what can serve them, and it splits the population

      w   p10 7.2   p50 28.0   p90 104.4   max 147.5
      w >  60   216 (27.7%)   saddle stationary-phase arm serves, FAST
      w <= 60   564 (72.3%)   Schwinger exact, SLOW

  `channels.py` does NOT call `select_branch` for a saddle host — not because
  the saddle lacks geometric routing, but because `cancellation_exponent` is
  positive-parity-only by design and **the operator's saddle arm owns the
  per-node routing internally**: resolved AND above the `w <= 60` ceiling ->
  stationary phase, otherwise Schwinger (`channels.py` ~L666-700).

  ## Two different fixes, do not conflate them

  **216 draws (w > 60) — INSTRUMENT.** The engine already serves these
  quickly via the saddle stationary-phase arm. The census does not model that
  internal routing, so it labels them `exact_engine`. Teach
  `census_dry_run.py` the saddle operator's own gate and they become served
  with no library change at all.

  **564 draws (w <= 60) — GENUINE.** Far from the caustic but low frequency,
  where diffraction still matters and Schwinger is the CORRECT evaluator, just
  slow. A fast path here means interpolation, i.e. chart coverage. So
  extending `rho_outer` to reach `rho_lobe ~ 11` IS on the table for this
  sub-population — unlike for the `w > 60` group, which needs no chart.

  ## Correction worth keeping

  An earlier pass of this analysis concluded "routing failure, not coverage —
  this rules out extending `rho_outer`". That was drawn from the `eta`
  measurement ALONE, before checking frequency. `eta >= 0.3` says the source
  is far from the caustic; it does NOT say a fast rung exists at that source's
  `w`. The two-leg conclusion was right for 28% of the population and wrong
  for 72%. Measure every leg of a gate before concluding which side of it a
  population lands on.

  ## Acceptance

  Split reporting by `w` band, not just by cause. The instrument fix should
  move ~216 draws with zero library change; anything more means it is
  crediting coverage that does not exist.
