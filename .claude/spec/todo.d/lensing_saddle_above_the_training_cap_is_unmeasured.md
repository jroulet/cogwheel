---
section: Backlog
---

- **THE WHOLE SADDLE REGION ABOVE `w = 148` IS UNMEASURED, AND THE CENSUS
  CANNOT SEE IT** `[→ spec]` — owner-flagged 2026-08-13, driver-measured.

  `_SADDLE_W_CEILING = 148` is a TRAINING cap, not a prior bound, and
  `scripts/census_dry_run.py::_draw_prior` samples `w ~ LogU(5, 148)` to
  mirror it. So every coverage number measured from that census — including
  the 87.61% in [[lensing_saddle_coverage_gap_breakdown]] — describes
  `w in [5, 148]` ONLY. It is not whole-prior coverage.

  The prior reaches much further. With `M_L in [10, 3500]` Msun over a
  20-1024 Hz band and `w = 8 pi G M_L (1+z_L) f / c^3`:

      M_L =   10 Msun  ->  w(1024 Hz) =   1.3
      M_L = 1000 Msun  ->  w(1024 Hz) = 126.8
      M_L = 3500 Msun  ->  w(1024 Hz) = 443.8

  The `w = 148` cap is hit at `M_L ~ 1167` Msun at the top of the band, so
  **~19% of the log-uniform lens-mass prior** has part of its frequency band
  above the cap. On a log-`w` prior over `[5, 444]`, roughly **24% of saddle
  draws sit above `w = 150`** and ~0.3% land in the `(148, 150]` sliver.

  ## What serves each band (measured: the partition evaluates fine at w = 149, 200, 400)

      w <= 148          trained charts + the tier-1 analytic rung + engine
      148 < w <= 150    NO CHART (cap is 148) and NO stationary phase
                        (needs w > 150) -- Schwinger mpmath only, ~100 s/call
      w > 150, resolved stationary-phase arm, fast
      w > 150, unresolved  refuses -> SchwingerCertificationError -> lnL = -inf

  The `(148, 150]` sliver exists purely because the TRAINING cap and the
  STATIONARY-PHASE threshold do not meet. Two constants, two apart, chosen
  independently: `_SADDLE_W_CEILING` is "2 below W_CEILING_SCHWINGER_QD" by
  its own comment, which guarantees the gap rather than closing it.

  ## Why this matters for the tier-1 rung

  Tier 1 is `w`-UNBOUNDED by construction: the delays and kernels come from
  the geometry, which is `w`-independent; only the Schwinger EVALUATION is
  capped. Verified — the partition returns finite kernels and a finite total
  at `w = 149, 200, 400`. So tier 1 plausibly serves BOTH the `(148, 150]`
  sliver AND the unresolved-above-150 population that currently REFUSES.

  If so the rung is worth substantially more than its brief claims, since a
  refusal is a hole in posterior support, not merely a slow evaluation.

  ## Measure before claiming any of it

  1. What fraction of saddle draws above `w = 150` are UNRESOLVED
     (`w * delta_min < RHO_END`) and therefore currently refused? That is the
     population tier 1 would rescue.
  2. Does tier-1 accuracy hold above the ceiling? `|E|/|F|` shrinks with `w`
     (measured p50 3.6e-05 at w=24 -> 3.0e-06 at w=58), so it should IMPROVE,
     but there is NO exact oracle above 150 to certify against — the same
     bind as any above-ceiling claim. State the extrapolation honestly rather
     than certifying what cannot be checked.
  3. Fix the census sampler so `w` spans the real prior, not the training cap.
     Until then no census number is whole-prior coverage, and every one of
     them should say so.
