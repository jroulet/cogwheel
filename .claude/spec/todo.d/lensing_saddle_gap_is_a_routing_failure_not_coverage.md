---
section: Backlog
depends_on: [lensing_saddle_coverage_gap_breakdown]
---

- **THE DOMINANT SADDLE GAP IS A ROUTING FAILURE, NOT MISSING COVERAGE — the
  sources are FAR from the caustic** `[→ spec]` — measured 2026-08-12.

  For the 780 sampled draws in the largest gap population (`rho_lobe` beyond
  `rho_outer`, excluding the degenerate-band sub-population), the PHYSICAL
  distance to the nearest caustic point (`geometry.nearest_caustic_point`
  `.distance`) is:

      eta   p10 0.617   p50 0.971   p90 1.380   max 2.682
      eta >= 0.3  (ETA_MIN_GEOMETRIC)  99.2%
      eta <  0.3  (genuinely near-caustic)  0.7%

      0.0-0.1     1   0.1%
      0.1-0.3     5   0.6%
      0.3-1.0   415  53.2%
      1.0-99    359  46.0%

  Meanwhile `rho_lobe` p50 is 5.70 and `|y|` p50 is only 0.732. So the large
  normalised radius was an ARTEFACT of dividing by the small `r_deltoid`:
  these sources are physically close to the origin but ~1.0 away from the
  caustic — THREE TIMES the engine's own far-from-caustic threshold.

  They do not need a chart. They are in the regime the geometric branch
  exists to serve. They reach `exact_engine` only because
  `_classify_saddle` never consults a far-field rung.

  ## Root cause: origin-`rho` is the wrong discriminator for an off-origin caustic

  `census_dry_run.py::_classify` tries `origin rho > 1 -> born` FIRST, and only
  then routes `gamma >= 1` into the lobe path. For the macro saddle the two
  deltoids sit OFF the origin, so a source can be physically far from BOTH
  lobes while having a small origin radius — it is between them. Origin-`rho`
  cannot express that, which is the same defect already fixed for the chart
  COORDINATE (origin-polar retired for lobe-local by `LobeExteriorChart`,
  commit 4c7dc92) but NOT yet for the serve GATE.

  ## Consequence for the fix direction

  This rules OUT the two options that looked most natural before measuring:
  extending `rho_outer` from ~3.4 to ~11, or giving the corridor its own
  chart coordinate. Both would build charts for a region that does not need
  charting. The fix is to re-key the gate off CAUSTIC DISTANCE (and the
  resolved-image conditions) rather than origin `rho`.

  ## What is NOT yet measured — do this before implementing

  `eta >= ETA_MIN_GEOMETRIC` is NECESSARY but NOT SUFFICIENT.
  `select_branch` also requires `w * delta_min >= RHO_END = 4.0` (resolved)
  and `L > L_MAX = 48` (strongly cancelling). This probe measured only the
  `eta` leg. Measure the fraction of these 780 that clear the FULL gate
  before claiming they are servable — the honest claim today is "they are far
  from the caustic", not "they are geometric-servable".

  Note the engine's own `eta` leg is live on BOTH parities and was measured
  separately (F034: saddle p90 8.95e-1 -> 4.54e-3 with worst case 484x over
  15% of resolved draws), so the saddle `eta` gate is already trusted
  machinery — this is about CONSULTING it, not building it.

  ## Acceptance

  The saddle `exact_engine` bucket drops by the measured servable fraction,
  and the per-cause breakdown in
  [[lensing_saddle_coverage_gap_breakdown]] shows the drop in
  `rho_lobe > rho_outer` specifically, not just in the total.
