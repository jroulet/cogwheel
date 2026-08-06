---
section: Backlog
---

- **The wedge's cusp-adapted `u` map is stored in fields named for ARC LENGTH**
  `[→ spec]` — naming debt knowingly incurred 2026-08-06, to be retired with
  [[lensing_r_caustic_should_root_find_not_scan]] in the same follow-up build.

  After the cusp-axis change the wedge chart's angular spline coordinate is
  `u = d^(2/3)` (angular distance to that tile's cusp), NOT arc length. But it
  is stored in the existing fields `theta_to_s` (a 2xN array
  `[theta_fine, u_fine]`) and `s_grid`, and validated by the SHARED
  `_validate_theta_to_s`.

  Reusing the fields was the right call FOR THAT BUILD: it keeps the serve path
  coordinate-agnostic (`_evaluate_chart` reads the stored map and splines in
  whatever it holds, so no serve change was needed at all), and the
  `axis_schema` tag `wedge_caustic_relative_v2` disambiguates the semantics at
  load. Nothing is incorrect.

  It is nonetheless the exact failure mode this repo has already been bitten
  by, recorded in [[lensing_farfield_name_spans_three_regimes]]: **a name that
  records a symbol's FIRST USE rather than its role**. `FarFieldChart` came to
  span intermediate field, far field and interior; `theta_to_s` now holds
  something that is not `s`. The next reader who trusts the field name will be
  wrong, and the `axis_schema` tag only protects the LOADER, not the reader.

  ## Why it is not a one-line rename

  `_validate_theta_to_s` is SHARED by the tube, lobe-interior and far-field
  arc-length maps, which legitimately do hold arc length. So retiring the name
  for the wedge means either

  - a wedge-specific validator plus wedge-specific field names
    (`theta_to_u` / `u_grid`), leaving the arc-length users untouched; or
  - a neutral name for the shared machinery (`theta_to_axis` / `axis_grid`)
    with the meaning carried entirely by `axis_schema`.

  The second is DRYer and matches how the loader already works, but touches
  every chart class and their serialized field names — i.e. it is a schema
  change for tube/lobe/far-field too, and would need its own version bump on
  each. Prefer the first unless the second is being done anyway.

  ## Work

  - Pick one of the two options above and apply it.
  - Whichever is chosen, the `axis_schema` tag remains the authority on
    semantics; the rename is for READERS, and must not become a second source
    of truth about what the axis means.
  - Land with the `r_caustic` root-find so the coordinate layer is touched
    once, not twice.

  ACCEPTANCE: no field or validator in the wedge path is named for arc length;
  the arc-length users keep theirs; serve remains coordinate-agnostic; and a
  stale artifact still hard-refuses on `axis_schema`, not on a field name.
