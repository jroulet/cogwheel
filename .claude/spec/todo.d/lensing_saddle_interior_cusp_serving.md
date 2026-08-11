---
section: Backlog
depends_on: []
---

- **Saddle deltoid interior cusp region is NOT served by the Pearcey arm**
  `[→ spec]` — measured 2026-08-11 by the driver (A/B at a8361be).

  The interior-cusp serving build made the ASTROID interior serve via the
  calibration bypass (3-stationary sources skip the per-image certificate),
  but the SADDLE deltoid interior is effectively unaffected.  For a deltoid
  cusp source (gamma=1.3, beta=0.37, src=(1.200,-0.173), rho=0.707),
  `_cusp_vertex` finds the deltoid tip but the source offset projects onto
  the HARD (radial) axis — the Pearcey controls land in the 1-stationary
  EXTERIOR regime, so `len(stationary_values) == 3` never fires and the
  calibration bypass is not reached.  Serving at w=40/60 is the
  PRE-EXISTING exterior Pearcey path (calibration-certified; forcing the
  certificate false refuses it).  At w>=80 the saddle deltoid interior
  refuses and falls to the exact engine.

  Root geometry: the astroid cusp soft axis points TOWARD the caustic
  interior (interior sources → 3 stationary points); the deltoid cusp soft
  axis is TANGENTIAL to the lobe ("interior" toward lobe centre is along
  the hard axis → 1 stationary point).  The build's new interior/exterior
  test configs are all gamma < 1, so the gap was not flagged.

  The ppGO fold-band gate IS parity-agnostic (nearest.distance is the fold
  arm's own admission, applies to both parities).

  ACCEPTANCE: the saddle deltoid interior (rho < 1, 4-image region) near a
  deltoid cusp vertex is served by a fast path (Pearcey arm, likely with a
  lobe-local coordinate like `surrogate._lobe_boundary_radius` /
  `_deltoid_cusp_axis_map`) at w >= 80 where it currently refuses;
  refusal-conservative; exterior saddle cusp sources unaffected; no live
  quadrature in the hot path.
