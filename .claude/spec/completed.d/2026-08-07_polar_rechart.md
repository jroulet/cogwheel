---
date: 2026-08-07
section: Backlog
---

### Exterior re-charted in polar (rho, theta_c) — `(s, d)` retired

`lensing_exterior_should_chart_in_polar_not_sd` closed. The exterior bulk is
now charted in the tiler's own caustic-fixed polar `(rho, theta_c)`
coordinates instead of the bridged `(s, d)` far-field-smooth coordinate;
`ExteriorPolarChart` replaces `FarFieldChart` as the exterior chart class.

- Deletion (`0a31fcf`, post-strand cleanup, ~1064 lines from
  `cogwheel/lensing/surrogate.py`): `FarFieldChart`, `_FarFieldArcMap`,
  `_to_farfield_smooth`/`_from_farfield_smooth`, `_farfield_serves`,
  `_caustic_arclength_map`, and the `(s, d)` schema constants; stale
  `(s, d)` artifacts hard-refuse at load (`5859a78` fixes the last stale
  `FarFieldChart` reference and the `(s, d)` docstring).
- The polar coordinate is self-contained, single-valued and cusp-safe by
  construction: tile edges sit on cusp rays (`_exterior_polar_tiles`), so no
  tile straddles a cusp and no arc-length map is needed.
- BOTH parities chart: positive-parity astroid and macro-saddle exteriors,
  the latter with an additive scalar-reach rho (`rho = 1 + |y| -
  _caustic_reach`, `drho/d|y| = 1`) since the two deltoid lobes do not
  enclose the origin.
- Cusp carve-out added (`_CUSP_EXCLUSION_DISTANCE = 0.2` y-units,
  `_exclude_near_cusp`), sized by the separation-gate contour, wider than
  the Pearcey arm's `_CUSP_ARM_COVERAGE = 0.07` image-theta rad.
- `m_lens_range` override added to `PriorBox.from_prior_classes()` /
  `train()` (same build) so a per-region probe trains a single mass/w
  stratum through the production training path.
- `_KNOWN_ENVELOPE_DEFINITIONS` widened to accept interior tags (the loader
  validates against the union of far-field and interior SACR-C labels).

Confirmed by the driver probes
(`2026-08-07_driver_probes_exterior_wedge.md`) and the test migration
(`72f4b84`, `test_lensing_farfield_envelope.py` /
`test_lensing_interior_wedge_chart.py` / `test_lensing_prior.py`).
