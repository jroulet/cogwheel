---
date: 2026-08-12
section: Backlog
---
# Saddle deltoid interior cusp serving — RESOLVED (commit 16aacc0)

The saddle (macro-saddle, gamma > 1) deltoid interior cusp region now serves
via the Pearcey arm at all w, closing the region that previously refused at
w >= 80 and fell to the exact engine.

## Resolution summary

- **`_is_interior` is the image-count discriminator** (`len(images) >= 4`),
  parity-correct: the deltoid caustic does not enclose the origin, so the
  origin-based `r_caustic` interior check misclassified saddle corridor
  sources as interior. The image-count gate fixes this on BOTH parities
  (astroid path byte-identical).
- **Interior bypass applies to the deltoid 1-stationary cluster**: for a
  deltoid cusp source the source offset projects onto the HARD (radial) axis
  (the deltoid cusp soft axis is TANGENTIAL to the lobe), so the Pearcey
  controls land in the 1-stationary EXTERIOR regime — the same
  self-calibrating ratio `P/P_asymp` justifies the interior bypass there as
  for the astroid interior degenerate cluster.
- **Cusp arm serves interior above-ceiling nodes in the mpmath band**: the
  serving ladder now offers the uniform rung in the mpmath band, so interior
  nodes it previously refused are served (not exact-engine fall-through).
- **6 stale refusal-contract fixtures re-pointed** to genuinely hard-core
  configs (2-image exterior sources where both arms decline and `F_op`
  raises `SchwingerCertificationError`). `_CUSP_TIE_EPS` allowlisted in the
  mechanical absorber guard.

## Recorded separately

The deltoid origin-rho misclassification in the ppGO/Born SERVING path is a
separate PRODUCTION bug, resolved in `288f37c`
(`lensing_saddle_origin_rho_assumption`).

## Acceptance

Saddle deltoid interior (`rho < 1`, 4-image region) near a deltoid cusp
vertex is served by the Pearcey arm at w >= 80 where it previously refused;
refusal-conservative (exterior saddle cusp sources unaffected); no live
quadrature in the hot path.
