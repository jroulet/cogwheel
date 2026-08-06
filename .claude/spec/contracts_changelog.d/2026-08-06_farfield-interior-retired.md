---
bump: patch
---

### FarFieldChart no longer serves the astroid interior

Commit `034fcf7` wired `InteriorWedgeChart` into training and retired the
`ffin` path: `_train_band_charts` now builds the astroid interior via
`_wedge_interior_tiles` / `_build_wedge_chart` -> `from_wedge_engine`
instead of `_farfield_interior_tiles` / `_build_farfield_chart`, and every
`INTERIOR_SACR_C`-labelled `FarFieldChart` branch (`_build_farfield_chart`,
`_subdivide_farfield_tile`, `_heldout_eps`) was deleted rather than left
reachable.

`lens_amplification_surrogate`'s description corrected: the parenthetical
"(exterior far-field and interior SACR-C alike)" on the `FarFieldChart`
record sentence was accurate before this build and is not now -- no
`FarFieldChart` record carries `INTERIOR_SACR_C` any more. The
`INTERIOR_SACR_C` label itself is unchanged and still documented on the
`LobeInteriorChart` and `InteriorWedgeChart` record descriptions in the
same field.
