---
section: Backlog
---

- **THE DOMINANT SADDLE GAP IS ALL GENUINE COVERAGE, AND 24% OF IT CARRIES
  ~99.5% OF THE ENGINE COST** `[→ spec]` — measured 2026-08-12. Population:
  the 901 draws with `rho_lobe > rho_outer`, Cause 1 of
  [[lensing_saddle_coverage_gap_breakdown]].

  ## They are far from the caustic, and NONE reach the geometric arm

      eta   p10 0.617   p50 0.971   p90 1.380   max 2.682
      eta >= 0.3 (ETA_MIN_GEOMETRIC): 99.2%

  `rho_lobe` p50 is 5.70 while `|y|` p50 is only 0.732 — the large normalised
  radius is an ARTEFACT of dividing by the small `r_deltoid`. These sources
  sit near the origin, between the lobes, a full unit from the caustic.

  But being far from the caustic does NOT reach a fast rung here. The saddle
  takes stationary phase ONLY when resolved AND
  `w > W_CEILING_SCHWINGER_QD = 150` (`operator.py` module docstring, ~L105).
  Measured `w` over this population: p10 7.2, p50 28.0, p90 104.4,
  **max 147.5** — so ZERO draws clear 150 and none are geometric-served.
  Every one of the 901 is a genuine coverage gap.

  ## Cost weighting inverts the priority

      w <= 60          564 (62.6%)  Schwinger double-double  ~0.2 s/call
      60 < w <= 150    216 (24.0%)  Schwinger MPMATH        ~85-120 s/call
      degenerate band  121 (13.4%)  see the rho_outer todo

  MEASURE AGAINST THE SERVING BUDGET, NOT AGAINST EACH OTHER. The warm
  single-thread lnlike is **9.8 ms** (ratio layer); the exact-engine crown is
  751 ms. Per call:

      double-double   200 ms   =     20x the ENTIRE lnlike budget
      mpmath      100,000 ms   =  10204x the ENTIRE lnlike budget

  A single such node does not fit inside an evaluation — it IS the
  evaluation. Amortised over all draws at the measured hit rates:

      dd      5.64% x 200 ms   =   11.3 ms/draw  =   1.2x budget
      mpmath  2.16% x 100 s    = 2160.0 ms/draw  =   220x budget

  So BOTH populations must be served. mpmath is worse by ~190x and is
  correctly first, but the 564 double-double draws are NOT cheap: on their
  own they more than double the cost of every likelihood evaluation. An
  earlier version of this note called them "worth ~1.9 minutes" — that is
  wall-clock over a 10k census sample, which is the wrong denominator
  entirely. Cost belongs against the budget a draw must fit in.

  ## HALF THE EXPENSIVE POPULATION CANNOT BE CHARTED AT ALL

  A chart node requires the engine to evaluate, and the 1F1 kernel refuses
  above the product ceiling `_DD_PRODUCT_MARGIN = 58` (`w * |y| <= 58`;
  training grids cap `w_max` so no node exceeds it). Measured over the 216
  mpmath-band draws (`|y|` p50 0.667, `w` p50 91.0):

      w * |y|   p10 10.5   p50 61.2   p90 188.8   max 394.1
      w*|y| <= 58   48.1%   a chart CAN be trained (at ~100 s/node)
      w*|y| >  58   51.9%   kernel refuses -- NO training node can exist

  So ~112 of the 216 are not a tiling problem at all: no amount of chart
  extent reaches them, because the producer cannot generate a node there.
  They need [[lensing_ppgo_extrapolation_beyond_engine_reach]] — train where
  the engine is cheap and extrapolate in `w` with the known analytic scaling
  divided out — or another named rung. The remaining ~104 are chartable but
  cost ~100 s per node, so a chart over them must be sized deliberately, not
  tiled at the density used below the ceiling.

  This also sharpens the ppGO fragment: it was filed as a general idea, and
  this is its concrete motivating population.

  ## Corrections worth keeping — three, all from partial measurement

  1. First pass called this a ROUTING failure needing no charts. That rested
     on `eta` alone: `eta >= 0.3` says a source is far from the caustic, NOT
     that a fast rung exists at its `w`.
  2. Second pass split it 28/72 using `w > 60` as the geometric threshold.
     WRONG CONSTANT — 60 is `W_CEILING_SCHWINGER`, the double-double/mpmath
     boundary INSIDE the wave branch. The geometric threshold is
     `W_CEILING_SCHWINGER_QD = 150`. Three distinct ceilings exist and none
     implies the others (F019); picking the wrong one turned the most
     expensive sub-population into an imagined "already served fast" one.
  3. The corrected reading: nothing here is served fast, and the population I
     had twice dismissed is the one that matters most.

  Read a threshold out of the code that OWNS the decision before classifying a
  population by it.

  ## Acceptance

  Report by `w` band and by COST, not by count. A fix that serves the 564 and
  not the 216 removes ~0.5% of the engine time this region costs.
