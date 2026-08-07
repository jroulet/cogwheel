---
section: Backlog
---

- **`_train_band_charts` cannot be run for ONE region, so every measurement
  reimplements it** `[housekeeping]` — identified 2026-08-06 after it caused a
  concrete error.

  `_train_band_charts(box, config, rng, outdir, parity, label, band,
  structure, charts, chart_reports, ppgo_map)` does an entire band — tube,
  exterior, and interior — in one call. There is no way to say "just the
  interior". Since the exterior costs ~39 min/band and the interior ~2, any
  interior measurement that used the real entry point would spend 95% of its
  time on the part not under test.

  So every probe reassembles the pipeline by hand: reading the tile-dict shape
  from `surrogate_training.py:4583`, calling `_wedge_interior_tiles` and
  `_build_wedge_chart` directly, and recomputing eps. That is the
  oracle-tautology trap by default rather than by accident — the measurement
  agrees with the DRIVER'S READING of the code, not with the code.

  ## It has already cost real errors

  - A probe hand-rolled the subdivision split (halve `r`, split `theta` at the
    u-midpoint) from the docstring instead of calling
    `_subdivide_wedge_tile`. It happened to agree — `theta_split = 1.248904`
    both ways — but only luck made that true, and the hand-rolled version also
    applied the DRIVER's eps metric rather than the gate that actually
    registers charts.
  - A second probe transcribed tile bounds from a printed table rounded to 4
    decimals, overshooting `pi/2` by 1e-4, which made
    `_wedge_cusp_axis_map` return a silently complex array. Taking the tiles
    FROM the tiler fixed it.

  ## Work

  - Give the training entry point a region filter (e.g.
    `regions=('interior',)`) so a measurement can invoke the SHIPPING path for
    one region at its own cost. Everything downstream — admission, tiling,
    build, gate, subdivision, reporting — then runs exactly as production
    does.
  - This is the cheap structural fix that makes "the oracle must call shipping
    code" the DEFAULT rather than a discipline the driver has to remember.

  ACCEPTANCE: an interior-only band run completes in interior-scale time
  (~minutes, not ~40), produces the same charts and chart_reports as the full
  path restricted to that region, and the wedge probes in
  [[lensing_subdividers_are_single_level]] can be re-expressed as calls to it.
