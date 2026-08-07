---
bump: patch
---

### Wedge interior angular axis is cusp-adapted (u = d**(2/3)), not arc length

SPEC.md's "Microlensing engine" `InteriorWedgeChart` prose described the
wedge's angular spline axis as an ARC-LENGTH map (`s_fine` = cumulative
trapezoid of `geometry.caustic_speed`), which is now wrong: this build
(`cogwheel/lensing/surrogate.py`, `surrogate_training.py`) replaced it with
`u = d**(2/3)`, `d` the angular distance to the NEAR astroid cusp. Arc
length made the cusp singularity WORSE (`caustic_speed` vanishes linearly
at a cusp, so `s ~ theta**2`); `u = d**(2/3)` is the exact cusp scaling of
`r_caustic` and absorbs it.

`_WEDGE_AXIS_SCHEMA` bumped `wedge_caustic_relative_v1 -> _v2`; v1 is
dropped from the known set (hard refuse, no migration — see the paired
`contracts_changelog.d` fragment for the `DATA_CONTRACTS.yaml` side).

Renamed the "ARC-LENGTH ANGULAR AXIS" subsection to "CUSP-ADAPTED ANGULAR
AXIS" and described: the caustic waist split (`theta_waist =
argmin_theta r_caustic(gamma, theta)`, new helper `_wedge_theta_waist`,
migrates up to ~30% from `pi/4` under shear), the per-tile `axis_origin`
(`'low'`/`'high'`), and `_wedge_cusp_axis_map` (uniform-in-`u` construction).
`(2, 2001)` array shape, `s_grid` field name, and the one-`np.interp`
serve-time contract are unchanged — only what the axis encodes changed.

Far-field (`s, d` fold-adapted) and tube (`s` = arc length along the fold)
angular/arc-length coordinate descriptions elsewhere in SPEC.md are
UNTOUCHED — this is wedge-only.

This is a documentation-accuracy correction of an existing SPEC.md
paragraph, not a claim about production coverage: the wedge interior's
eps-gate status (`todo.d/lensing_wedge_angular_axis_is_cusp_singular.md`,
which consolidates the earlier eps-bar and axis-strip fragments) is tracked
separately in `todo.d` and is NOT resolved by this fragment.
