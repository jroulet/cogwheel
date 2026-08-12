---
section: Backlog
depends_on: []
---

- **Origin-based `caustic_rho` / `r_caustic` misclassify the deltoid: the caustic does NOT enclose the origin for the saddle**
  `[→ spec]` — measured 2026-08-11 by the driver (user-flagged, Test Dev unaware).

  The deltoid (saddle parity, gamma > 1) is TWO disjoint 3-cusp lobes sitting
  OFF the origin on the shear axis.  The origin is NOT enclosed: for
  gamma=1.3, 190/360 origin rays MISS the caustic (`geometry.r_caustic`
  raises LensDomainError); for gamma=2.0, 322/360 miss.

  ## The measured bug

  `ppgo_map.caustic_rho(gamma, |y|)` uses a SINGLE scalar reach (the max
  caustic radius over angle) to define rho = |y|/reach.  For the saddle this
  is WRONG in the CORRIDOR between the two lobes (genuinely EXTERIOR, 2
  images) but with |y| < scalar_reach, so caustic_rho reports 0.18-0.29
  (< 1 = interior) while n_images = 2 = EXTERIOR.  Measured at gamma=1.3:
  - (0,0.3): n_images=2, caustic_rho=0.175 (WRONG interior)
  - (0.5,0): n_images=2, caustic_rho=0.292 (WRONG interior)
  - (0.3,0.3): n_images=2, caustic_rho=0.247 (WRONG interior)
  A scalar-reach rho cannot distinguish "inside the negative lobe" from
  "inside the positive lobe" from "in the corridor" at equal |y|.

  ## Users of origin-based caustic measures (audit needed per site)

  1. `cogwheel/lensing/likelihood.py:1396` `_ppgo_band_split` — classifies
     parity + rho for BOTH parities; saddle corridor misrouted.
  2. `cogwheel/lensing/likelihood.py:1681` — Born exterior vs fold-ppGO
     INTERIOR handoff gate `rho <= 1.0`; saddle corridor wrongly treated
     interior.
  3. `cogwheel/lensing/surrogate_census.py:285,394` — census rho
     classification, both parities.
  4. `cogwheel/lensing/surrogate_training.py:4820` — ppGO exclusion rho,
     both parities.
  5. `cogwheel/lensing/surrogate.py` `_to_caustic_fixed` — directional
     `r_caustic` for the astroid; the saddle already uses lobe-local
     coordinates (`_lobe_boundary_radius`, `_deltoid_cusp_axis_map`) which
     are CORRECT -- those paths are fine.
  6. `_pearcey_cusp.py` (this build's on-axis fix) — the direction-based
     `r_caustic` interior gate was already REPLACED by `len(images) >= 4`
     (image-count is parity-correct); this is the CORRECT discriminator.

  ## Fix direction

  The image-count discriminator (`len(images) == 4` interior, `== 2`
  exterior) is the parity-correct interior test (the census caps at 4
  images).  The origin-based scalar `caustic_rho` must NOT be used to
  classify saddle interior/exterior.  Audit each user above: either
  (a) branch on parity and use image-count for the saddle, or (b) use a
  lobe-local rho for the saddle (like the surrogate's lobe charts).  The
  ppGO/Born gates at likelihood.py:1396/1681 are the highest-risk
  consumers.

  ACCEPTANCE: for every saddle config, interior/exterior classification is
  parity-correct (matches n_images); the corridor between deltoid lobes is
  classified EXTERIOR; no serving path misroutes a saddle exterior source
  to interior-handling code.

  ## CONFIRMED PRODUCTION BUG (driver, 2026-08-11) — NOT just tests

  The ppGO map is SHIPPED (`cogwheel/data/certified_ppgo_map.npz`,
  schema 0.2.0) and covers the SADDLE (parity_codes=[0,1], gamma_edges up
  to 1.55).  Its rho design assumes rho=1 separates interior(4-image) from
  exterior(2-image) — TRUE for the astroid, FALSE for the deltoid.

  Measured: gamma=1.3 corridor source (0.5,0) has 2 images (EXTERIOR) but
  caustic_rho=0.292, so `_ppgo_cell_coords` returns
  ('saddle', 1.3, 0.292) → the shipped map CERTIFIES it:
  w_cert=19.16, w_trust=28.75.  `_ppgo_band_split` then returns w_trust
  and `_surrogate_coefficients` band-splits this EXTERIOR corridor source
  using an INTERIOR-cell certification — the served amplification is
  routed through the wrong ppGO cell.  rho=0.7 (inside a lobe, TRUE
  interior) is UNKNOWN; rho=2.0 (far exterior) certifies at w_cert=11.0.

  So the saddle ppGO certification itself is unsound wherever the deltoid
  corridor maps to rho<1.  FIX REQUIRED in production:
  `_ppgo_cell_coords` (likelihood.py:1356) must NOT use origin-based
  `caustic_rho` for the saddle — use lobe-local rho (the surrogate's
  `_lobe_boundary_radius` / deltoid geometry) or refuse saddle ppGO
  entirely pending a lobe-aware map.  Also audit `_ppgo_cell_ceiling`
  (likelihood.py:1447), `surrogate_census.py:285/394`, and
  `surrogate_training.py:4820` (the ppGO-exclusion rho).

  ## LIVE BY DEFAULT: the Born/fold-ppGO interior handoff (likelihood.py:1681)

  This path is LIVE without the ppGO map.  For a saddle corridor source
  (gamma=1.3, (0.5,0)): born_chart is None, caustic_rho=0.292 (wrongly
  interior), so `rho <= 1.0` fires the FOLD-PPGO INTERIOR handoff.  The
  handoff's `_merging_fold_pair` returns None (2-image exterior, no valid
  fold pair) yet `fold_ppgo_correction` returns a finite value
  (0.51+0.32j) — serving an EXTERIOR source through the INTERIOR branch.
  This misroutes the served amplification for saddle corridor draws on the
  default path.

  FIX (production, highest priority): in the fold-ppGO interior handoff
  and `_ppgo_cell_coords`, use the image-count discriminator
  (`len(images) == 4` = interior, `== 2` = exterior) instead of
  origin-based `caustic_rho` for the SADDLE.  For positive parity the
  caustic_rho scalar reach is correct (astroid encloses origin).  The
  fix must branch on parity: astroid keeps caustic_rho, saddle uses
  image-count (or lobe-local rho).
