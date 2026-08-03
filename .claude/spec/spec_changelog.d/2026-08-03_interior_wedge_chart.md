---
bump: patch
---

### Add InteriorWedgeChart to SPEC.md surrogate chart collection

SPEC.md row 55 (sampling/surrogate layer) now describes `InteriorWedgeChart`,
the positive-parity astroid-interior chart type added in surrogate.py. Covers
the caustic-normalised wedge-polar coordinates `(r, theta_wedge)`, D2 symmetry
fold, `_WEDGE_AXIS_SCHEMA = 'wedge_caustic_relative_v1'` tag, `_wedge_serves`
dispatch, `_WedgeCausticMap`, and `from_wedge_engine` training entry point.
