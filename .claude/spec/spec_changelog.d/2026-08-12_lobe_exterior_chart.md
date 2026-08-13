---
date: 2026-08-12
bump: patch
---

### Document LobeExteriorChart, retire ExteriorPolarChart for macro-saddle exterior

Post-commit sync for 4c7dc92 (WP2, deltoid exterior geometry fix), deferred
via INS-5-001 on two builds stranded at the tree gate.

SPEC.md described the macro-saddle (`gamma > 1`) exterior as an
`ExteriorPolarChart` in origin-centred caustic-fixed polar coordinates with
an additive scalar-reach `rho`. The build retired that path: the two deltoid
lobes sit off the origin and neither encloses it, so a single origin-ray
crosses EXT(2img) -> INT(4img lobe) -> EXT(2img), making origin-polar
coordinates topologically ill-posed for the saddle exterior. The macro-saddle
exterior is now charted per lobe as `LobeExteriorChart`, in the SAME
lobe-local `(rho_lobe, theta_local)` frame as `LobeInteriorChart` but over
the exterior shell `rho_lobe` in `(1, rho_outer]`, trained by
`from_lobe_exterior_engine` and served by `_lobe_exterior_serves`.

Updates: `ExteriorPolarChart` is now documented as positive-parity (astroid)
only; `LobeExteriorChart` added alongside `LobeInteriorChart` in the pipeline
table row and the "Key abstractions" far-field coordinate contract; the
inter-lobe-corridor sentence corrected (a corridor source is now served by
the canonical `+y1` lobe's exterior chart via the D2 reflection fold,
superseding the earlier "falls through to the exact engine" description);
the GATED-subdivider kind list clarified to `far-field, wedge, lobe-interior`
(lobe-exterior and tube charts have no subdivider); the lobe test-file
certified-by sentence extended to cover the new `LobeExteriorChart` test
classes.
