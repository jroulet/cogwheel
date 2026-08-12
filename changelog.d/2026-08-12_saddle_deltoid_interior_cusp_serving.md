---
date: 2026-08-12
---
### Saddle deltoid interior cusp sources now serve via the Pearcey arm (image-count discriminator)

The saddle (macro-saddle, `gamma > 1`) deltoid interior cusp region now
serves via the Pearcey arm at all w, where it previously refused at w >= 80
and fell to the exact engine. Changes in
`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`:

- **`_is_interior` is the image-count discriminator** (`len(images) >= 4`),
  parity-correct: the deltoid caustic does not enclose the origin, so the
  origin-based `r_caustic` interior check misclassified saddle corridor
  sources as interior. The image-count gate fixes both parities (astroid
  path byte-identical).
- **Interior bypass for the deltoid 1-stationary cluster**: for a deltoid
  cusp source the source offset projects onto the HARD (radial) axis (the
  deltoid cusp soft axis is TANGENTIAL to the lobe), so the Pearcey
  controls land in the 1-stationary EXTERIOR regime — the same
  self-calibrating `P/P_asymp` ratio justifies the bypass there as for the
  astroid interior degenerate cluster.
- **Cusp arm serves interior above-ceiling nodes in the mpmath band**: the
  serving ladder offers the uniform rung in the mpmath band, so interior
  nodes it previously refused are served.
- **6 stale refusal-contract fixtures re-pointed** to genuinely hard-core
  configs (2-image exterior sources where both arms decline and `F_op`
  raises `SchwingerCertificationError`); `_CUSP_TIE_EPS` allowlisted in the
  mechanical absorber guard.
