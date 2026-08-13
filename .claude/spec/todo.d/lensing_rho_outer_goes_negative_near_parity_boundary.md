---
section: Backlog
---

- **`rho_outer` GOES NEGATIVE NEAR THE PARITY BOUNDARY, SILENTLY EMPTYING THE
  EXTERIOR BAND** `[→ spec]` — measured 2026-08-12. The lobe-exterior charted
  band is `rho_lobe in (1, rho_outer]`, with

      rho_outer = 1.0 + _SOURCE_BOX_CORNER - coordinate_radius_min

  (`census_dry_run.py::_saddle_stable_subbands`, mirroring the production
  packing path). `coordinate_radius_min` comes from
  `_st._coordinate_radius_bounds(band, parity)` and blows up as `gamma -> 1`,
  so `rho_outer` falls below 1 and then negative:

      gamma = 1.005   rho_outer = -4.147
      gamma = 1.02    rho_outer =  0.940
      gamma = 1.03    rho_outer =  2.235
      gamma = 1.10    rho_outer =  3.533   (healthy)

  An interval `(1, rho_outer]` with `rho_outer <= 1` is EMPTY. So for those
  bands no lobe-exterior chart can ever serve anything, whatever the
  admission says — and nothing anywhere reports this. It is not caught, not
  logged, and not refused; the band simply produces no exterior coverage and
  the draws land in the gap under a different label. MEASURED: 144 of the
  1742 saddle draws reaching `_classify_saddle` sit in such a band.

  This is a THEORETICAL bug, not a tuning issue: a negative-width interval is
  not a small band, it is an ill-posed one, and the formula that produced it
  is being evaluated outside its domain of validity.

  ## What to settle

  1. Is the formula right near `gamma = 1`? `coordinate_radius_min` diverging
     at the parity boundary may be correct physics (the deltoids degenerate
     as `det A -> 0`) while the SUBTRACTION that turns it into a normalised
     outer radius is simply not meaningful there.
  2. Whatever the answer, a computed `rho_outer <= 1` must REFUSE LOUDLY
     (named error or an explicit recorded skip) rather than silently yield an
     empty band. The repo's own convention is a named refusal over a silent
     degradation — this currently fails that convention.
  3. Decide what serves `gamma` just above 1. It may be that no chart should:
     as `det A -> 0` the macro magnification diverges and the exact engine or
     a dedicated near-boundary rung may be the honest answer. `gamma = 1` is
     already a measure-zero NAMED refusal; the question is the neighbourhood.

  Related: this is a sub-population of Cause 1 in
  [[lensing_saddle_coverage_gap_breakdown]], but it has a different fix — the
  other 757 draws there are a genuine reach question, these 144 are a
  degenerate band definition.
