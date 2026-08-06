---
section: Backlog
---

- **THE INTERIOR WEDGE CHART IS BUILT, SERVE-WIRED, TESTED — AND NEVER
  TRAINED** `[→ spec]` — audited 2026-08-06. `InteriorWedgeChart` (added
  2026-08-03, `ff06b8a`) has 20 references in `surrogate.py`, a live
  `select_chart` branch via `_wedge_serves`, a `_WedgeCausticMap` transform
  table, `from_wedge_values`, `_assemble`, and two dedicated test files that
  pass. `grep 'InteriorWedgeChart('` across `cogwheel/` and `scripts/` finds
  exactly ONE construction and it is in a test. `surrogate_training.py` never
  mentions the class.

  So the astroid interior is trained the OLD way: `_train_band_charts` calls
  `_farfield_interior_tiles` and builds `FarFieldChart` tiles carrying the
  `INTERIOR_SACR_C` label (the `ffin` charts — 106 of the 165 charts in the
  2026-08-05 production attempt). `select_chart` will happily dispatch to a
  chart type nothing produces.

  ## Why the wedge chart is the right home for the astroid interior

  1. **`(s, d)` degenerates inside.** The far-field coordinate needs a UNIQUE
     nearest-caustic foot; on the medial axis (astroid centre and diagonals)
     the foot is ambiguous, which is why `_FARFIELD_MEDIAL_AXIS_TOL = 1e-6`
     and the near-tied-foot rejection exist. `(r, theta_wedge)` is global
     inside: `r = 0` centre, `r = 1` caustic boundary, no ambiguity.
  2. **4-fold symmetry.** `theta_wedge = atan2(|y2|, |y1|)` in `[0, pi/2]`
     covers a QUARTER of the interior; `r_caustic` is 4-fold symmetric so the
     fold is exact. Potentially ~4x fewer interior charts.
  3. **The DD cap becomes exact.** `w * |y| < 58` becomes
     `w * r * r_caustic < 58`, known at each grid point — the class docstring
     claims this eliminates the DD bottleneck for high-w draws at small `|y|`.

  The ENVELOPE is unchanged: the wedge chart already declares the
  `tau_c`-demodulated `INTERIOR_SACR_C` label. This is a COORDINATE change,
  not a label change — chart class and `envelope_definition` are orthogonal.

  ## The macro-saddle path is ALREADY WIRED — copy it

  The saddle interior is complete end to end and is the working template:
  `_saddle_lobe_admissions` (2276) -> `_lobe_interior_tiles` (2340, called at
  4162) -> `_build_lobe_chart` (2874, called at 4364) -> `from_lobe_engine`,
  producing `LobeInteriorChart` in lobe-local `(rho_lobe, theta_local)` with
  `rho_lobe = |y - centroid| / r_deltoid(theta_local)`, cusp-aligned tiles
  that never straddle one of the lobe's three cusp rays or the lobe-local
  `+-pi` seam, and admission that also excludes the inter-lobe corridor.

  `LobeInteriorChart` landed 2026-07-28 WITH its training path;
  `InteriorWedgeChart` landed 2026-08-03 WITHOUT one. The astroid interior is
  the ONLY region of the four whose intended chart class is not trained --
  tube, exterior (both parities) and saddle lobe interiors are all wired.

  So this is a TRANSCRIPTION of a working pattern, not a design problem:
  `_build_lobe_chart` is the template for `_build_wedge_chart`, and
  `_lobe_interior_tiles` is the template for the wedge tiler (both lay uniform
  radial rows over a normalised radius and cusp-align the angular axis).

  ## Work

  - an engine-backed constructor (the class has `from_wedge_values` only;
    `FarFieldChart` has `from_engine`) evaluating at `(gamma, r, theta_wedge)`
    nodes through `_from_wedge_fixed`;
  - `_build_wedge_chart` in `surrogate_training.py`;
  - a wedge tiler over `r in [0, 1)` x `theta_wedge in [0, pi/2]`, cusp-aligned
    (the astroid cusps sit at the wedge edges);
  - swap the positive-parity interior call at `_train_band_charts` (~4239) and
    RETIRE the `ffin` path — one authoritative representation per region;
  - the macro-saddle interior is NOT in scope: `LobeInteriorChart` owns it.

  ## NO `ffin` SURVIVES — retire the whole path, do not leave it dark

  `ffin` charts exist ONLY for the astroid interior, and the wedge chart
  covers exactly that region (both are positive-parity-only:
  `_interior_admission` raises `ValueError` for `parity != 1`). So after the
  swap NOTHING produces a `FarFieldChart` carrying `INTERIOR_SACR_C`, and
  these become dead and must be DELETED in the same build rather than left
  reachable:

  - `_farfield_interior_tiles` and `_interior_admission` (unless the wedge
    tiler genuinely reuses the directional-admission geometry -- if so, move
    it, do not leave a second caller);
  - the `definition=INTERIOR_SACR_C` branch of `_build_farfield_chart`;
  - the `child_definition = INTERIOR_SACR_C` branch of
    `_subdivide_farfield_tile` (~3675);
  - the "INTERIOR `FarFieldChart`" branch of `_heldout_eps` (~2988) and its
    `max|E|` normalization special case;
  - any load/serve branch reconstructing an interior-tagged `FarFieldChart`.

  The `INTERIOR_SACR_C` LABEL itself stays -- it is the envelope both
  `LobeInteriorChart` and `InteriorWedgeChart` carry. What retires is the
  pairing of that label with `FarFieldChart`.

  ACCEPTANCE: no `FarFieldChart` carries `INTERIOR_SACR_C` after the swap;
  a grep for `INTERIOR_SACR_C` in `surrogate_training.py` finds it only on the
  wedge and lobe paths;
  interior held-out eps no worse than the `ffin` baseline at equal or lower
  chart count; a medial-axis query that the `ffin` path refused now serves;
  the D2 fold is exercised (a query and its three mirror images serve
  identical values).

  DIVISION OF LABOUR after this lands — astroid interior:
  `InteriorWedgeChart`; saddle lobe interiors: `LobeInteriorChart`; exterior
  both parities: `FarFieldChart`; near-caustic: `TubeChart`.
